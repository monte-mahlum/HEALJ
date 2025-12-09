using LinearAlgebra
using Random
using Flux
using NNlib

export GCNLayer, GCNEncoder, gcn_forward, perturb_nodes

struct GCNLayer
    W::Dense
    σ::Function
end

Flux.@layer GCNLayer trainable=(W,)

GCNLayer(in_dim::Integer, out_dim::Integer; σ = NNlib.relu, bias::Bool = false) =
    GCNLayer(Dense(in_dim, out_dim; bias=bias), σ)

# H: N×F_in → Dense expects F_in×N
_linear(d::Dense, H::AbstractMatrix) =
    permutedims(d(permutedims(Float32.(H))))

function gcn_forward(
    Z::AbstractMatrix{<:Real},
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
)
    N, F = size(Z)
    E = length(src)

    # Add self loops:
    # ---------
    all_src = Vector{Int32}(undef, E + N)
    all_dst = Vector{Int32}(undef, E + N)

    @inbounds for e in 1:E
        all_src[e] = Int32(src[e])
        all_dst[e] = Int32(dst[e])
    end
    @inbounds for i in 1:N
        all_src[E + i] = Int32(i)
        all_dst[E + i] = Int32(i)
    end
    # ---------

    # 
    deg = zeros(Float32, N)
    @inbounds for d in all_dst
        deg[d] += 1f0
    end
    invsqrt = 1f0 ./ sqrt.(deg .+ 1f-8)

    out = zeros(Float32, N, F)
    @inbounds for e in eachindex(all_src)
        s = Int(all_src[e])
        d = Int(all_dst[e])
        c = invsqrt[d] * invsqrt[s]
        out[d, :] .+= c .* Float32.(Z[s, :])
    end
    return out
end

function (l::GCNLayer)(
    H::AbstractMatrix{<:Real},
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
)
    Z = _linear(l.W, H)          # N×F_out
    P = gcn_forward(Z, src, dst)
    return l.σ.(P)
end

struct GCNEncoder
    layers::Vector{GCNLayer}
end

Flux.@layer GCNEncoder trainable=(layers,)

GCNEncoder(dims::AbstractVector{<:Integer}; σ = NNlib.relu) =
    GCNEncoder([GCNLayer(dims[i], dims[i+1]; σ=σ) for i in 1:length(dims)-1])

function (enc::GCNEncoder)(
    H::AbstractMatrix{<:Real},
    src::AbstractVector{<:Integer},
    dst::AbstractVector{<:Integer},
)
    X = H
    for l in enc.layers
        X = l(X, src, dst)
    end
    return X
end

# For contrastive learning
function perturb_nodes(
    H::AbstractMatrix{<:Real};
    ε::Float32 = 0.1f0,
    rng::AbstractRNG = Random.default_rng(),
)
    X = Float32.(H)
    V = randn(rng, Float32, size(X))
    norms = sqrt.(sum(V .* V; dims=2)) .+ 1f-8
    V ./= norms
    return X .+ ε .* (V .* sign.(X))
end
