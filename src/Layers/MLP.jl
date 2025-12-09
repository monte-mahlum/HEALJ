using LinearAlgebra
using Flux
using NNlib

export MLPHead

"""
MLPHead: simple MLP classifier head on top of HGT graph embeddings.

Input:
  Z :: B×D    (batch of B graph embeddings, dim D)

Output:
  Ŷ :: B×C    (batch of logits for C labels)
"""
struct MLPHead
    layers::Vector{Dense}   # each is Din -> Dout
end

Flux.@layer MLPHead trainable = (layers,)


function MLPHead(
    in_dim::Int,
    hidden_dims::AbstractVector{<:Int},
    out_dim::Int;
    activation = NNlib.relu,
)
    dims = Int[in_dim; hidden_dims...; out_dim]
    L = length(dims) - 1

    dense_layers = Vector{Dense}(undef, L)
    for i in 1:L
        Din = dims[i]
        Dout = dims[i+1]
        σ = (i == L) ? identity : activation   # last layer: no activation (logits)
        dense_layers[i] = Dense(Din, Dout, σ)
    end

    return MLPHead(dense_layers)
end

# Julia's Dense layers do things in a transposed manner, so we fix it here.
_dense_rows(d::Dense, X::AbstractMatrix{Float32}) =
    permutedims(d(permutedims(X)))

"""
Forward pass:

Z :: B×D (Float32) → Ŷ :: B×C (Float32)

Rows are independent, so we apply each Dense row-wise using _dense_rows from HGT.
"""
function (m::MLPHead)(Z::AbstractMatrix{Float32})
    X = Z
    for l in m.layers
        X = _dense_rows(l, X)
    end
    return X   # B×C logits
end
