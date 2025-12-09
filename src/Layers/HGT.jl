using LinearAlgebra
using Flux
using NNlib

export HGT

# ---------------------- Attention Pooling (Eq. 5) ----------------------

"""
Attention pooling (HEAL Eq. (5)):

z = softmax( Qᵖ · (U · Kᵖ)ᵀ / √D ) · (U · Vᵖ),

where U ∈ ℝᴷˣᴰ are super-node embeddings and z ∈ ℝᴰ is a graph-level vector.
"""
struct AttnPool
    q::Vector{Float32}   # length D (learnable query Qᵖ)
    KP::Dense            # D → D
    VP::Dense            # D → D
end

Flux.@layer AttnPool trainable=(q, KP, VP,)

function AttnPool(D::Int)
    q  = 0.02f0 .* randn(Float32, D)
    KP = Dense(D, D; bias=false)
    VP = Dense(D, D; bias=false)
    return AttnPool(q, KP, VP)
end

# # Julia's Dense layers do things in a transposed manner, so we fix it here.
_dense_rows(d::Dense, U::AbstractMatrix{<:Real}) =
    permutedims(d(permutedims(Float32.(U)))) # (WUᵀ)ᵀ = UWᵀ

function (p::AttnPool)(U::AbstractMatrix{<:Real})
    K, D = size(U)
    length(p.q) == D || throw(ArgumentError("AttnPool: q has length $(length(p.q)), but U has D = $D"))

    Kp = _dense_rows(p.KP, U)             # K×D, (U · KPᵀ)
    Vp = _dense_rows(p.VP, U)             # K×D  (U · VPᵀ)

    # Qᵖ is 1×D
    qrow   = reshape(Float32.(p.q), 1, D)          # 1×D
    scores = (qrow * permutedims(Kp)) ./ sqrt(Float32(D))  # 1×K
    w      = NNlib.softmax(scores; dims=2)         # 1×K
    z      = w * Vp                                # 1×D

    return vec(z)  # D
end


# ---------------------- HGT block (w/ pooling) ----------------------

"""
Hierarchical Graph Transformer with attention pooling (HEAL Eqs. (2)–(5)).

Inputs are a *batched* microbatch:
- H          : Ntotal×D node embeddings
- src, dst   : edge lists (1-based batch node indices, typically Int32)
- node2graph : length Ntotal vector in 1..B giving graph id per node

Assumes concatenated graphs: nodes of each graph are in one contiguous block.
"""
struct HGT
    K::Int                          # number of super-nodes per graph
    D::Int                          # feature dimension
    heads::Int                      # number of attention heads
    queries::Vector{Matrix{Float32}}  # each K×D, one per head
    gcn_k::Vector{GCNLayer}         # per-head key encoder
    gcn_v::Vector{GCNLayer}         # per-head value encoder
    fc1::Dense                      # (heads*D) → D (FC¹)
    pool::AttnPool                  # K×D → D (attention pooling)
end

Flux.@layer HGT trainable=(queries, gcn_k, gcn_v, fc1, pool,)

function HGT(D::Int; K::Int = 16, heads::Int = 1)
    queries = [0.02f0 .* randn(Float32, K, D) for _ in 1:heads]
    gcn_k   = [GCNLayer(D, D; σ = identity) for _ in 1:heads]
    gcn_v   = [GCNLayer(D, D; σ = identity) for _ in 1:heads]
    fc1     = Dense(heads * D, D, NNlib.relu)
    pool    = AttnPool(D)
    return HGT(K, D, heads, queries, gcn_k, gcn_v, fc1, pool)
end

# Compute (start,end) index ranges per graph assuming node2graph is nondecreasing 1..B.
function _graph_ranges(node2graph::AbstractVector{<:Integer}, B::Int)
    N = length(node2graph)
    N == 0 && throw(ArgumentError("HGT: empty node2graph"))

    starts = fill(0, B)
    ends   = fill(0, B)

    current = Int(node2graph[1])
    starts[current] = 1
    @inbounds for i in 2:N
        g = Int(node2graph[i])
        if g != current
            ends[current] = i - 1
            current = g
            starts[current] = i
        end
    end
    ends[current] = N

    any(==(0), starts) && throw(ArgumentError("HGT: node2graph missing some graph id(s) in 1..$B"))
    return starts, ends
end

# Extract edges inside [a,b], then shift indices by -(a-1). Keep edges as Int32.
function _subedges(src, dst, a::Int, b::Int)
    s = Int32[]
    d = Int32[]
    off = Int32(a - 1)
    @inbounds for i in eachindex(src)
        si = Int(src[i]); di = Int(dst[i])
        if a <= si <= b && a <= di <= b
            push!(s, Int32(si) - off)
            push!(d, Int32(di) - off)
        end
    end
    return s, d
end

# reuse _dense_rows from above for FC¹
# _dense_rows(d::Dense, U::AbstractMatrix{<:Real}) defined above

function (hgt::HGT)(
    H::AbstractMatrix{<:Real},
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
    node2graph::AbstractVector{<:Integer},
)
    Ntotal, D = size(H)
    D == hgt.D || throw(ArgumentError("HGT: H has D = $D, but HGT was built with D = $(hgt.D)"))

    B = Int(maximum(node2graph))          # number of graphs in microbatch
    starts, ends = _graph_ranges(node2graph, B)

    Z = zeros(Float32, B, D)              # B×D graph-level representations

    for g in 1:B
        a, b = starts[g], ends[g]
        Hg = Float32.(@view H[a:b, :])    # Ng×D for graph g
        sg, dg = _subedges(src, dst, a, b)

        # For each head: Γₕ = softmax(Qₕ Kₕᵀ / √D) Vₕ  (K×D)
        Cs = Vector{Matrix{Float32}}(undef, hgt.heads)
        for h in 1:hgt.heads
            Ki = hgt.gcn_k[h](Hg, sg, dg)         # Ng×D
            Vi = hgt.gcn_v[h](Hg, sg, dg)         # Ng×D
            Qi = hgt.queries[h]                   # K×D

            scores = (Qi * permutedims(Ki)) ./ sqrt(Float32(D))  # K×Ng
            A      = NNlib.softmax(scores; dims = 2)             # K×Ng (row-wise)
            Ci     = A * Vi                                      # K×D
            Cs[h]  = Ci
        end

        # U = FC¹([Γ₁, …, Γ_H])  (Eq. 4)
        Ccat = hcat(Cs...)                      # K×(heads*D)
        U    = _dense_rows(hgt.fc1, Ccat)       # K×D
        # z = attention pooling over super-nodes (Eq. 5)
        z    = hgt.pool(U)                      # D
        Z[g, :] .= z
    end

    return Z  # B×D
end

