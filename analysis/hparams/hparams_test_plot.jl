using CSV
using DataFrames
using Statistics
using Plots
using Measures: mm


# Config: paths relative to this script's directory
const BASE_DIR = @__DIR__
const CSV_PATH = joinpath(BASE_DIR, "hparams_test_data.csv")
const OUT_DIR  = BASE_DIR


# Helpers for subscript indices
function sub_digit(c::Char)
    c == '0' && return '₀'
    c == '1' && return '₁'
    c == '2' && return '₂'
    c == '3' && return '₃'
    c == '4' && return '₄'
    c == '5' && return '₅'
    c == '6' && return '₆'
    c == '7' && return '₇'
    c == '8' && return '₈'
    c == '9' && return '₉'
    return c
end

function subscript_int(n::Integer)
    s = string(n)
    chars = Char[sub_digit(c) for c in s]
    return String(chars)
end


# Load data
@info "Reading CSV from $CSV_PATH"
df = CSV.read(CSV_PATH, DataFrame)

required = ["lambda","tau","eps","epoch","train_sup","val_sup"]
missing  = setdiff(required, names(df))
@assert isempty(missing) "Missing expected columns in CSV: $(missing)"


# Group by (tau, lambda, eps) and compute metrics
gdf = groupby(df, [:tau, :lambda, :eps])

struct HParamMetrics
    tau::Float64
    lambda::Float64
    eps::Float64
    risk::Float64    # min val_sup over epochs
    gap::Float64     # val_sup - train_sup at best epoch
end

metrics = HParamMetrics[]

for g in gdf
    τ  = Float64(first(g.tau))
    λ  = Float64(first(g.lambda))
    ε  = Float64(first(g.eps))

    idx = argmin(g.val_sup)              # best epoch (min val_sup)
    best_val_sup   = g.val_sup[idx]
    best_train_sup = g.train_sup[idx]

    risk = best_val_sup
    gap  = best_val_sup - best_train_sup

    push!(metrics, HParamMetrics(τ, λ, ε, risk, gap))
end



# Define ordering of h = (tau, lambda, eps)
# Order: (τ₁,λ₁,ε₁), (τ₂,λ₁,ε₁), (τ₃,λ₁,ε₁),
#        (τ₁,λ₂,ε₁), ..., (τ₃,λ₃,ε₃)
# eps outer, lambda middle, tau inner.

taus    = sort(unique(df.tau))        # τ₁, τ₂, τ₃
lambdas = sort(unique(df.lambda))     # λ₁, λ₂, λ₃
epses   = sort(unique(df.eps))        # ε₁, ε₂, ε₃

ordered_keys = [(τ, λ, ε) for ε in epses, λ in lambdas, τ in taus] |> vec

"""
print_hparam_dictionary(taus, lambdas, epses, ordered_keys)

Prints a mapping from index-style labels (τᵢ, λⱼ, εₖ)
to their numeric values, and also lists the ordered triples.
"""
function print_hparam_dictionary(
    taus::AbstractVector,
    lambdas::AbstractVector,
    epses::AbstractVector,
    ordered_keys::AbstractVector{<:Tuple}
)
    println("=== Index mapping for τ, λ, ε ===")
    for (i, τ) in enumerate(taus)
        τsub = subscript_int(i)
        println("τ$τsub = $τ")
    end
    println()

    for (j, λ) in enumerate(lambdas)
        λsub = subscript_int(j)
        println("λ$λsub = $λ")
    end
    println()

    for (k, ε) in enumerate(epses)
        εsub = subscript_int(k)
        println("ε$εsub = $ε")
    end
    println()

    println("=== Ordered hyperparameter triples h = (τᵢ, λⱼ, εₖ) ===")
    for (idx, (τ, λ, ε)) in enumerate(ordered_keys)
        i = findfirst(==(τ), taus)
        j = findfirst(==(λ), lambdas)
        k = findfirst(==(ε), epses)

        τsub = subscript_int(i)
        λsub = subscript_int(j)
        εsub = subscript_int(k)

        println(
            "h[$idx] = (τ$τsub, λ$λsub, ε$εsub) = ",
            "($τ, $λ, $ε)",
        )
    end
    println()
end

print_hparam_dictionary(taus, lambdas, epses, ordered_keys)

# Prepare data for plotting
metric_map = Dict{Tuple{Float64,Float64,Float64},HParamMetrics}()
for m in metrics
    metric_map[(m.tau, m.lambda, m.eps)] = m
end

labels      = String[]
risk_sq     = Float64[]   # (generalization risk)^2
gap_vals    = Float64[]   # generalization gap

for (τ, λ, ε) in ordered_keys
    key = (Float64(τ), Float64(λ), Float64(ε))
    @assert haskey(metric_map, key) "Missing metrics for (tau,lambda,eps) = $key in CSV."

    m = metric_map[key]

    i = findfirst(==(τ), taus)
    j = findfirst(==(λ), lambdas)
    k = findfirst(==(ε), epses)

    τsub = subscript_int(i)
    λsub = subscript_int(j)
    εsub = subscript_int(k)

    push!(labels, "τ$τsub, λ$λsub, ε$εsub")

    push!(risk_sq,  m.risk^2) 
    push!(gap_vals, m.gap)      
end

x = 1:length(labels)


# Scale y-values for nicer scientific notation on axes
#   - risk² in units of 10⁻³
#   - gap   in units of 10⁻²
risk_scaled = risk_sq .* 1e3 
gap_scaled  = gap_vals .* 1e2

# Plot style defaults (bigger fonts)

default(
    legend       = false,
    grid         = :y,
    framestyle   = :box,
    tickfont     = font(14),
    guidefont    = font(16),
    titlefont    = font(22),
)

margins = (left = 10mm, right = 5mm, top = 7mm, bottom = 25mm)


# Plot 1: (Generalization risk)^2 (scaled by 10⁻³) as stems + points

plt_risk = plot(
    x,
    risk_scaled;
    seriestype        = :sticks,
    marker            = :circle,
    markersize        = 7,
    markerstrokewidth = 1.0,
    linealpha         = 0.8,
    xticks            = (x, labels),
    xrotation         = 60,
    size              = (1400, 700),
    dpi               = 200,
    title             = "Generalization risk by (τᵢ, λⱼ, εₖ)",
    left_margin       = margins.left,
    right_margin      = margins.right,
    top_margin        = margins.top,
    bottom_margin     = margins.bottom,
)

xlabel!(plt_risk, "Hyperparameter triple  h = (τᵢ, λⱼ, εₖ)")
ylabel!(plt_risk, "Generalization risk²  (×10⁻³)")

risk_path = joinpath(OUT_DIR, "hparams_generalization_risk_scatter.png")
@info "Saving risk plot to $risk_path"
savefig(plt_risk, risk_path)


# Plot 2: Generalization gap (scaled by 10⁻²) as stems + points
plt_gap = plot(
    x,
    gap_scaled;
    seriestype        = :sticks,
    marker            = :circle,
    markersize        = 7,
    markerstrokewidth = 1.0,
    linealpha         = 0.8,
    xticks            = (x, labels),
    xrotation         = 60,
    size              = (1400, 700),
    dpi               = 200,
    title             = "Generalization gap by (τᵢ, λⱼ, εₖ)",
    left_margin       = margins.left,
    right_margin      = margins.right,
    top_margin        = margins.top,
    bottom_margin     = margins.bottom,
)

xlabel!(plt_gap, "Hyperparameter triple  h = (τᵢ, λⱼ, εₖ)")
ylabel!(plt_gap, "Generalization gap  (×10⁻²)")

gap_path = joinpath(OUT_DIR, "hparams_generalization_gap_scatter.png")
@info "Saving gap plot to $gap_path"
savefig(plt_gap, gap_path)

@info "Done."
