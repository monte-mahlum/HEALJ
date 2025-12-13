using LinearAlgebra
using Random
using Flux
using NNlib
using SparseArrays
using Zygote
using CUDA

export GCNLayer, GCNEncoder, gcn_forward, perturb_nodes


"""
    GCNLayer(in_dim, out_dim; σ = relu, bias = false)

One Kipf–Welling style GCN layer.

Given node features `H :: N×Fin` and an (undirected) graph with edges
`src, dst` (1-based indices), it computes

    H′ = σ( Â * H * W )

where Â is the fixed normalized adjacency matrix and W are the layer
weights from a `Dense` layer.
"""
struct GCNLayer
    lin :: Dense
    σ   :: Function
end

GCNLayer(in_dim::Integer, out_dim::Integer;
         σ::Function = NNlib.relu,
         bias::Bool = false) =
    GCNLayer(Dense(in_dim, out_dim; bias = bias), σ)

# Apply Dense row-wise: H :: N×Fin  ->  N×Fout
_dense_rows(d::Dense, H::AbstractMatrix) =
    permutedims(d(permutedims(H)))


"""
    build_norm_adj(src, dst, N) -> Â::Matrix{Float32}

Build symmetric normalized adjacency

    Â = D^{-1/2} (A + I) D^{-1/2}

for a graph with `N` nodes and edge list `(src, dst)` (1-based).

- `src`, `dst` can be any integer vectors (CPU or GPU); they are copied
  to CPU internally.
- Returns a dense `Matrix{Float32}`.
"""
function build_norm_adj(
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
    N::Integer,
)
    # Ensure CPU 32-bit indices
    src_cpu = Int32.(collect(src))
    dst_cpu = Int32.(collect(dst))

    E = length(src_cpu)
    nentries = 2E + N

    I = Vector{Int32}(undef, nentries)
    J = Vector{Int32}(undef, nentries)
    V = Vector{Float32}(undef, nentries)

    k = 1
    @inbounds for e in 1:E
        i = src_cpu[e]
        j = dst_cpu[e]
        I[k] = i; J[k] = j; V[k] = 1.0f0; k += 1
        I[k] = j; J[k] = i; V[k] = 1.0f0; k += 1
    end

    # Self-loops
    @inbounds for n in 1:Int32(N)
        I[k] = n; J[k] = n; V[k] = 1.0f0; k += 1
    end

    A = sparse(I, J, V, N, N)

    # Degree vector
    d = vec(sum(A, dims = 2))            

    invsqrtd = similar(d, Float32)
    @inbounds for i in eachindex(d)
        v = d[i]
        invsqrtd[i] = v > 0 ? inv(sqrt(Float32(v))) : 0.0f0
    end

    Dinv = spdiagm(0 => invsqrtd)
    Ahat = Dinv * A * Dinv                    

    # Dense CPU matrix
    return Array{Float32}(Ahat)
end

# Graph structure is fixed, so cannot backprop through it
Zygote.@nograd build_norm_adj

"""
    gcn_forward(H, src, dst) -> Hprop

Apply fixed normalized adjacency to node features `H`.

- `H`   :: N×F (CPU or GPU)
- `src`, `dst` :: edge lists for the graph

The result lives on the **same device** as `H` (if CUDA is available).
"""
function gcn_forward(
    H::AbstractMatrix,
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
)
    N = size(H, 1)
    Ahat_cpu = build_norm_adj(src, dst, N)

    if CUDA.has_cuda()
        Hgpu     = cu(H)
        Ahat_gpu = cu(Ahat_cpu)
        return Ahat_gpu * Hgpu
    else
        # Pure CPU path
        return Ahat_cpu * H
    end
end


# One GCN layer: H -> σ(Â H W)
function (layer::GCNLayer)(
    H::AbstractMatrix,
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
)
    Z = gcn_forward(H, src, dst)           # N×Fin
    return layer.σ(_dense_rows(layer.lin, Z))
end

"""
    GCNEncoder(dims; σ = relu, bias = false)

Stack of GCN layers with widths given by `dims`.

Example:
    enc = GCNEncoder([F, D, D])
    Henc = enc(H0, src, dst)
"""
struct GCNEncoder
    layers :: Vector{GCNLayer}
end

function GCNEncoder(
    dims::AbstractVector{<:Integer};
    σ::Function = NNlib.relu,
    bias::Bool = false,
)
    @assert length(dims) ≥ 2 "dims must have at least input and output size"
    layers = GCNLayer[]
    for i in 1:(length(dims) - 1)
        push!(layers, GCNLayer(dims[i], dims[i+1]; σ = σ, bias = bias))
    end
    return GCNEncoder(layers)
end

GCNEncoder(dims::Tuple{Vararg{Int}}; σ::Function = NNlib.relu, bias::Bool = false) =
    GCNEncoder(collect(dims); σ = σ, bias = bias)

function (enc::GCNEncoder)(
    H::AbstractMatrix,
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
)
    Z = H
    for layer in enc.layers
        Z = layer(Z, src, dst)
    end
    return Z
end

"""
    perturb_nodes(H; ε = 0.1f0, rng = Random.default_rng())

Add small adversarial-style perturbations along the sign direction
of the features:

    H̃ = H + ε * (V / ‖V‖₂) .* sign(H)

This function is intended to be used on **CPU** feature matrices `H`
coming from the data loader, before moving batches to GPU.
Works on any AbstractMatrix without mutation.
"""
function perturb_nodes(
    H::AbstractMatrix;
    ε::Float32 = 0.1f0,
    rng::AbstractRNG = Random.default_rng(),
)
    T = eltype(H)
    X = H
    V = randn(rng, T, size(X))                     
    norms = sqrt.(sum(V .* V; dims = 2)) .+ T(1e-8)
    Vnorm = V ./ norms
    return X .+ ε .* (Vnorm .* sign.(X))
end
