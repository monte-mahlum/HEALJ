using LinearAlgebra
using Flux
using NNlib

export MLPHead

"""
    MLPHead(layer_dims)

Simple MLP classifier head on top of graph embeddings.

- Input:  Z :: B×D_in   (graph embeddings)
- Output: Ŷ :: B×D_out (logits)

`layer_dims` is a vector of layer widths, including input and output.
For example:

    MLPHead([D, 512, C])

creates a 2-layer MLP with hidden size 512 and output size C.
"""
struct MLPHead
    layers :: Vector{Dense}
end

_dense_rows_mlp(d::Dense, X::AbstractMatrix) =
    permutedims(d(permutedims(X)))

# Constructor
function MLPHead(layer_dims::AbstractVector{<:Integer})
    @assert length(layer_dims) ≥ 2 "layer_dims must include input and output size"

    dense_layers = Dense[]
    for i in 1:(length(layer_dims) - 1)
        in_dim  = layer_dims[i]
        out_dim = layer_dims[i+1]
        σ = i == length(layer_dims) - 1 ? identity : NNlib.relu
        push!(dense_layers, Dense(in_dim, out_dim, σ))
    end
    return MLPHead(dense_layers)
end

MLPHead(dims::Tuple{Vararg{Int}}) = MLPHead(collect(dims))

# Better Constructor
function MLPHead(in_dim::Integer, hidden::AbstractVector{<:Integer}, out_dim::Integer)
    return MLPHead(vcat(in_dim, hidden, out_dim))
end

# Forward pass
function (m::MLPHead)(Z::AbstractMatrix)
    X = Z
    for l in m.layers
        X = _dense_rows_mlp(l, X)
    end
    return X
end
