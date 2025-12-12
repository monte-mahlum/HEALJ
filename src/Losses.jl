using LinearAlgebra
using Flux
using NNlib
using Statistics

export supervised_bce_loss, contrastive_loss, total_loss


"""
    supervised_bce_loss(logits, y)

Binary cross-entropy with logits for multi-label targets.

- `logits` :: B×C  (pre-sigmoid outputs from `MLPHead`)
- `y`      :: B×C  (targets in [0,1])

Works on CPU and GPU arrays.
"""
function supervised_bce_loss(
    logits::AbstractMatrix{<:Real},
    y::AbstractMatrix{<:Real},
)
    size(logits) == size(y) ||
        throw(ArgumentError("logits and y must have same size, got $(size(logits)) vs $(size(y))"))

    return Flux.Losses.logitbinarycrossentropy(
        logits,
        y;
        agg = mean,
    )
end


"""
    contrastive_loss(Z1, Z2; τ = 0.5f0)

InfoNCE / NT-Xent style contrastive loss between two views of the same
batch of graph embeddings.

- `Z1`, `Z2` :: B×D
- `τ`        :: temperature
"""
function contrastive_loss(
    Z1::AbstractMatrix{<:Real},
    Z2::AbstractMatrix{<:Real};
    τ::Real = 0.5f0,
)
    B, D = size(Z1)
    size(Z2, 1) == B || throw(ArgumentError("Z1 and Z2 batch sizes differ"))
    size(Z2, 2) == D || throw(ArgumentError("Z1 and Z2 feature dims differ"))

    # Normalize rows
    function _row_norm(X)
        T = eltype(X)
        norms = sqrt.(sum(abs2, X; dims = 2)) .+ T(1e-8)
        return X ./ norms
    end

    Z1n = _row_norm(Z1)
    Z2n = _row_norm(Z2)

    T = eltype(Z1n)
    temp = T(τ)

    # Similarity matrix between views
    S12 = (Z1n * permutedims(Z2n)) ./ temp   # B×B
    S21 = (Z2n * permutedims(Z1n)) ./ temp   # B×B

    # InfoNCE: positives on diagonal
    function _nce(S)
        # subtract max for numerical stability along each row
        S_stable = S .- maximum(S; dims = 2)
        expS = exp.(S_stable)
        Z = sum(expS; dims = 2)
        # log softmax diagonal
        log_probs = S_stable .- log.(Z)
        return -mean(diag(log_probs))
    end

    L12 = _nce(S12)
    L21 = _nce(S21)

    return (L12 + L21) / 2
end


"""
    total_loss(logits, y, Z1, Z2; λ = 0.1f0, τ = 0.5f0)

Combined loss

    L = L_sup + λ * L_con,

where `L_sup` is supervised BCE and `L_con` is contrastive InfoNCE.

Returns:

    (L_total, (Lsup = L_sup, Lcon = L_con))
"""
function total_loss(
    logits,
    y,
    Z1,
    Z2;
    λ::Real = 0.1f0,
    τ::Real = 0.5f0,
)
    Lsup = supervised_bce_loss(logits, y)
    Lcon = contrastive_loss(Z1, Z2; τ = Float32(τ))
    return Lsup + Float32(λ) * Lcon, (Lsup = Lsup, Lcon = Lcon)
end
