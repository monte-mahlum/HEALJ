#!/usr/bin/env julia
using Pkg
Pkg.activate(@__DIR__)  # HEALJ root
try
    using HEALJ
catch
    # not strictly required if the .jls only contains standard types,
    # but harmless to try
end

using Serialization
using Statistics
using Printf

sigmoid(x) = 1 / (1 + exp(-x))

# ---------- utilities ----------
# NamedTuple field lookup
function pickfield(nt::NamedTuple, candidates::Vector{Symbol})
    ks = Set(keys(nt))
    for c in candidates
        if c in ks
            return c
        end
    end
    return nothing
end

# Heuristic: treat as logits if values not in [0,1] (common case)
function looks_like_logits(v)
    m = minimum(v); M = maximum(v)
    return (m < -1e-6) || (M > 1 + 1e-6)
end

"""
Compute protein-centric Fmax from score matrix S (N×C) and truth matrix Y (N×C).
Protein-centric aggregation: average per-protein precision over proteins with >=1 predicted label;
average per-protein recall over proteins with >=1 true label; F1 from aggregated P,R.
"""
function protein_centric_fmax(S::AbstractMatrix, Y::AbstractMatrix; step=0.01)
    N, C = size(S)
    @assert size(Y) == (N, C)

    Yb = Y .!= 0

    bestF = -Inf
    bestt = 0.0
    bestP = 0.0
    bestR = 0.0

    for t in 0.0:step:1.0
        Psum = 0.0; Pcnt = 0
        Rsum = 0.0; Rcnt = 0

        @inbounds for i in 1:N
            ŷ = @view(S[i, :]) .>= t
            npred = count(ŷ)
            ntrue = count(@view(Yb[i, :]))

            if npred > 0
                tp = count(ŷ .& @view(Yb[i, :]))
                Psum += tp / npred
                Pcnt += 1
            end
            if ntrue > 0
                tp = count(ŷ .& @view(Yb[i, :]))
                Rsum += tp / ntrue
                Rcnt += 1
            end
        end

        P = (Pcnt == 0) ? 0.0 : Psum / Pcnt
        R = (Rcnt == 0) ? 0.0 : Rsum / Rcnt
        F = (P + R == 0) ? 0.0 : 2P*R/(P + R)

        if F > bestF
            bestF, bestt, bestP, bestR = F, t, P, R
        end
    end

    return bestF, bestt, bestP, bestR
end

"""
Compute protein-centric P,R,F1 from per-protein confusion counts TP/FP/FN.
(Your CSV-style columns: positive_positive, positive_negative, negative_positive)
"""
function protein_centric_PRF_from_counts(TP, FP, FN)
    N = length(TP)
    @assert length(FP) == N && length(FN) == N

    pred_pos = TP .+ FP
    true_pos = TP .+ FN

    prec_i = TP ./ pred_pos
    rec_i  = TP ./ true_pos

    # CAFA-style: precision averaged over proteins where you predicted something;
    # recall averaged over proteins where there is at least one true label.
    prec_i = prec_i[pred_pos .> 0]
    rec_i  = rec_i[true_pos .> 0]

    P = isempty(prec_i) ? 0.0 : mean(prec_i)
    R = isempty(rec_i)  ? 0.0 : mean(rec_i)
    F = (P + R == 0) ? 0.0 : 2P*R/(P + R)
    return P, R, F
end

# ---------- main ----------
length(ARGS) >= 1 || error("Usage: julia --project=. GOcount.jl path/to/raw_results.jls")

jls_path = ARGS[1]
obj = Serialization.deserialize(jls_path)

println("Loaded: ", jls_path)
println("Top-level type: ", typeof(obj))

(obj isa AbstractVector) || error("Expected a Vector/Array at top-level; got $(typeof(obj))")
length(obj) > 0 || error("Empty results vector.")

# Expect Vector{NamedTuple}
nt = obj[1]
(nt isa NamedTuple) || error("Expected elements to be NamedTuple; got element type $(typeof(nt))")

println("\nFirst element keys:")
println(collect(keys(nt)))

# --------- Mode B: confusion-count only? ---------
kTP = pickfield(nt, [:positive_positive, :tp, :TP])
kFP = pickfield(nt, [:positive_negative, :fp, :FP])
kFN = pickfield(nt, [:negative_positive, :fn, :FN])

if kTP !== nothing && kFP !== nothing && kFN !== nothing
    println("\nDetected confusion-count fields ($(kTP), $(kFP), $(kFN)).")
    println("-> Computing protein-centric P/R/F1 at this single operating point (NOT F_max).")

    TP = Float64[ obj[i][kTP] for i in eachindex(obj) ]
    FP = Float64[ obj[i][kFP] for i in eachindex(obj) ]
    FN = Float64[ obj[i][kFN] for i in eachindex(obj) ]

    P, R, F = protein_centric_PRF_from_counts(TP, FP, FN)
    @printf("\nProtein-centric (single threshold): P=%.6f, R=%.6f, F1=%.6f\n", P, R, F)
    println("\nNote: F_max requires per-label scores (or confusion counts across many thresholds).")
    exit()
end

# --------- Mode A: per-label scores/logits + labels ---------
kY = pickfield(nt, [:y, :ytrue, :y_true, :labels, :targets, :truth])
kS = pickfield(nt, [:scores, :score, :probs, :prob, :yhat, :ŷ, :pred, :preds])
kZ = pickfield(nt, [:logits, :logit, :raw_logits, :z])

(kY === nothing) && error("Could not find y/labels field in NamedTuple keys above.")
(kS === nothing && kZ === nothing) && error("Could not find scores/probs or logits field in NamedTuple keys above.")

println("\nDetected label field: ", kY)
if kS !== nothing
    println("Detected score/prob field: ", kS)
else
    println("Detected logit field: ", kZ, " (will apply sigmoid)")
end

# Build matrices by stacking vectors
N = length(obj)

# determine C from first row
y1 = obj[1][kY]
(y1 isa AbstractVector) || error("Expected $(kY) to be a vector; got $(typeof(y1))")
C = length(y1)

S = Matrix{Float64}(undef, N, C)
Y = Matrix{Int8}(undef, N, C)

for i in 1:N
    yi = obj[i][kY]
    length(yi) == C || error("Row $i: length($(kY))=$(length(yi)) != C=$C")

    # truth
    @inbounds for j in 1:C
        Y[i, j] = (yi[j] != 0) ? Int8(1) : Int8(0)
    end

    # scores
    if kS !== nothing
        si = obj[i][kS]
        length(si) == C || error("Row $i: length($(kS))=$(length(si)) != C=$C")
        if looks_like_logits(si)
            @inbounds for j in 1:C
                S[i, j] = sigmoid(Float64(si[j]))
            end
        else
            @inbounds for j in 1:C
                S[i, j] = Float64(si[j])
            end
        end
    else
        zi = obj[i][kZ]
        length(zi) == C || error("Row $i: length($(kZ))=$(length(zi)) != C=$C")
        @inbounds for j in 1:C
            S[i, j] = sigmoid(Float64(zi[j]))
        end
    end
end

@printf("\nStacked matrices: scores %d×%d, labels %d×%d\n", size(S)..., size(Y)...)

Fmax, tstar, Pstar, Rstar = protein_centric_fmax(S, Y; step=0.01)
@printf("\nOVERALL protein-centric: F_max = %.6f at t* = %.2f (P=%.6f, R=%.6f)\n",
        Fmax, tstar, Pstar, Rstar)

println("\nMF/BP/CC note: to compute aspect-specific F_max, this file must also include a length-C namespace/aspect vector,")
println("or you must provide a separate mapping label_index -> {MF,BP,CC}.")
