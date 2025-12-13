using LinearAlgebra
using Flux
using NNlib
using Zygote
using CUDA

export HGT

"""
    AttnPool(D)

Graph-level attention pooling over K super-node embeddings U ∈ ℝ^{K×D}:

Given parameters q ∈ ℝ^D and Kᵖ, Vᵖ ∈ ℝ^{D×D},

    K' = U Kᵖ      (K×D)
    V' = U Vᵖ      (K×D)
    w  = softmax( q · K'ᵀ / √D )   (1×K)
    z  = w V'                     (1×D)

Returns `z` as a length-D vector. Works on CPU and GPU.
"""
struct AttnPool
    q  :: AbstractVector
    KP :: Dense
    VP :: Dense
end

# Row-wise Dense: U::K×D -> K×D′ 
_dense_rows_attn(d::Dense, U::AbstractMatrix) =
    permutedims(d(permutedims(U)))

function AttnPool(D::Int)
    q  = 0.02f0 .* randn(Float32, D)
    KP = Dense(D, D)
    VP = Dense(D, D)
    return AttnPool(q, KP, VP)
end

Flux.@layer AttnPool trainable = (q, KP, VP,)

function (p::AttnPool)(U::AbstractMatrix)
    K, D = size(U)
    length(p.q) == D ||
        throw(ArgumentError("AttnPool: q length $(length(p.q)) ≠ D = $D"))

    Kp = _dense_rows_attn(p.KP, U)             # K×D
    Vp = _dense_rows_attn(p.VP, U)             # K×D

    qrow   = reshape(p.q, 1, :)                # 1×D
    T      = eltype(U)
    scores = (qrow * permutedims(Kp)) ./ sqrt(T(D))  # 1×K
    w      = NNlib.softmax(scores; dims = 2)   # 1×K
    z      = w * Vp                            # 1×D

    return vec(z)                              # D
end


"""
    HGT(D; K = 16, heads = 1)

HEAL-style HGT block with `heads` attention heads and `K` learned queries
per head. For each graph in a microbatch, it:

  1. Takes node embeddings H_g (N_g×D),
  2. Applies per-head GCNs to get key/value node features,
  3. Forms K super-node embeddings via attention,
  4. Runs a feed-forward + attention pooling to get a graph embedding z_g.

Forward call:

    Z = hgt(H, src, dst, node2graph)

Inputs:
- `H`          :: N_total×D node embeddings (CPU or GPU)
- `src`, `dst` :: edge lists (Int/Int32, CPU or GPU)
- `node2graph` :: length N_total vector with graph ids in 1..B

Returns:
- `Z` :: B×D matrix of graph embeddings on the same device as `H`.
"""
struct HGT
    K       :: Int
    D       :: Int
    heads   :: Int
    queries :: Vector{AbstractMatrix}   # length = heads, each K×D
    gcn_k   :: Vector{GCNLayer}
    gcn_v   :: Vector{GCNLayer}
    fc1     :: Dense                    # The internal MLP
    pool    :: AttnPool
end

Flux.@layer HGT trainable = (queries, gcn_k, gcn_v, fc1, pool,)

function HGT(D::Int; K::Int = 16, heads::Int = 1)
    queries = [0.02f0 .* randn(Float32, K, D) for _ in 1:heads]
    gcn_k   = [GCNLayer(D, D; σ = identity) for _ in 1:heads]
    gcn_v   = [GCNLayer(D, D; σ = identity) for _ in 1:heads]
    fc1     = Dense(heads * D, D, NNlib.relu)
    pool    = AttnPool(D)
    return HGT(K, D, heads, queries, gcn_k, gcn_v, fc1, pool)
end




"""
Index helpers on CPU, no grad.
Compute [start,end] node index ranges per graph, assuming node2graph values
are in 1..B and non-decreasing. Always operates on CPU copies.
"""
function _graph_ranges(node2graph::AbstractVector{<:Integer}, B::Int)
    node2 = collect(node2graph)             # ensure CPU
    N = length(node2)
    N == 0 && throw(ArgumentError("HGT: empty node2graph"))

    starts = fill(0, B)
    ends   = fill(0, B)

    current = Int(node2[1])
    starts[current] = 1
    @inbounds for i in 2:N
        g = Int(node2[i])
        if g != current
            ends[current] = i - 1
            current = g
            starts[current] = i
        end
    end
    ends[current] = N

    any(==(0), starts) &&
        throw(ArgumentError("HGT: node2graph missing some graph id(s) in 1..$B"))

    return starts, ends
end

# Extract edges fully inside [a,b] and shift indices by -(a-1). CPU only.
function _subedges(
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
    a::Int,
    b::Int,
)
    src_cpu = collect(src)
    dst_cpu = collect(dst)

    off = a - 1
    ssub = Int32[]
    dsub = Int32[]
    @inbounds for e in eachindex(src_cpu)
        i = src_cpu[e]
        j = dst_cpu[e]
        if a <= i <= b && a <= j <= b
            push!(ssub, Int32(i - off))
            push!(dsub, Int32(j - off))
        end
    end
    return ssub, dsub
end

# Pure index bookkeeping: AD should not differentiate through these.
Zygote.@nograd _graph_ranges
Zygote.@nograd _subedges





"""
Forward pass
"""

function (hgt::HGT)(
    H::AbstractMatrix,                      # N_total×D
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
    node2graph::AbstractVector{<:Integer},
)
    Ntotal, D = size(H)
    D == hgt.D ||
        throw(ArgumentError("HGT: H has D = $D, but HGT was built with D = $(hgt.D)"))

    # Ensure graph ids live on CPU for indexing
    node2 = collect(node2graph)
    B = Int(maximum(node2))            # number of graphs
    starts, ends = _graph_ranges(node2, B)

    # build all graph embeddings via list-comprehension
    z_list = [begin
        # slice graph g
        a = starts[g]; b = ends[g]

        # Node slice for graph g: plain indexing so GPU gets CuArray (not SubArray)
        Hg = H[a:b, :]                      # N_g×D (CPU or GPU)
        sg, dg = _subedges(src, dst, a, b)  # edge lists on CPU

        # per head attention
        Cs = [begin
            Ki = hgt.gcn_k[h](Hg, sg, dg)      # N_g×D
            Vi = hgt.gcn_v[h](Hg, sg, dg)      # N_g×D

            Qi = hgt.queries[h]                # K×D, stored on CPU
            # Move queries to device of Ki if necessary
            Qi_dev = (Ki isa CUDA.AbstractGPUArray) ? cu(Qi) : Qi

            T = eltype(Hg)
            scores = (Qi_dev * permutedims(Ki)) ./ sqrt(T(hgt.D))  # K×N_g
            A      = NNlib.softmax(scores; dims = 2)               # K×N_g
            A * Vi                                                # K×D
        end for h in 1:hgt.heads]

        # pool to graph embedding
        Ccat  = hcat(Cs...)                        # K×(heads*D)
        U     = _dense_rows_attn(hgt.fc1, Ccat)    # K×D
        hgt.pool(U)                                # returns z_g ∈ ℝ^D
    end for g in 1:B]

    # Stack z_g rows into B×D without in-place writes
    Z = reduce(vcat, (permutedims(z) for z in z_list))   # B×D
    return Z
end

