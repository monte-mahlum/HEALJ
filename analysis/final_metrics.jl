
using Printf

""" To Run:
 julia --project=. .\analysis\final_metrics.jl `
.\artifacts\test_results_20251211_153949\raw_results_20251211_153949_analysis.csv

Output will be something like:
Metric 1 (positive_positive / n_y1): mean = 0.216295, sample var = 0.054980 (N=3123)
Metric 2 (positive_negative / n_y1): mean = 0.091276, sample var = 0.101557 (N=3123)
Metric 3 (negative_positive / n_y0): mean = 0.029440, sample var = 0.001275 (N=3123)
Metric 4 (negative_negative / n_y0): mean = 0.998586, sample var = 0.000011 (N=3123)
Metric 5 (abs(n_y1 - n_yhat1)):   mean = 72.489273, sample var = 7102.476743 (N=3123)
"""

# Compute mean and sample variance for a vector of Float64
function mean_and_sample_var(xs::Vector{Float64})
    n = length(xs)
    if n == 0
        return (NaN, NaN)
    elseif n == 1
        return (xs[1], NaN)   # variance undefined with 1 sample
    end

    μ = sum(xs) / n
    s2 = sum((x - μ)^2 for x in xs) / (n - 1)
    return (μ, s2)
end

function final_metrics(csv_path::String)
    # metric storage
    m1 = Float64[]  # positive_positive / n_y1
    m2 = Float64[]  # positive_negative / n_y1
    m3 = Float64[]  # negative_positive / n_y0
    m4 = Float64[]  # negative_negative / n_y0
    m5 = Float64[]  # abs(n_y1 - n_yhat1)

    open(csv_path, "r") do io
        # skip header
        header = readline(io)

        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue

            parts = split(line, ',')
            if length(parts) != 9
                @warn "Unexpected number of columns, skipping line" line length_parts=length(parts)
                continue
            end

            # columns should be:
            # 1: protein_id (string)
            # 2: n_y0
            # 3: n_y1
            # 4: n_yhat0
            # 5: n_yhat1
            # 6: positive_positive
            # 7: positive_negative
            # 8: negative_positive
            # 9: negative_negative

            # protein_id = parts[1]  # not needed for final metrics
            n_y0              = parse(Int, parts[2])
            n_y1              = parse(Int, parts[3])
            n_yhat0           = parse(Int, parts[4])
            n_yhat1           = parse(Int, parts[5])
            positive_positive = parse(Int, parts[6])
            positive_negative = parse(Int, parts[7])
            negative_positive = parse(Int, parts[8])
            negative_negative = parse(Int, parts[9])

            # Metrics with denominators n_y1 / n_y0:
            if n_y1 > 0
                push!(m1, positive_positive / n_y1)  # TP rate among positives
                push!(m2, positive_negative / n_y1)  # FP among positives (as you defined)
            end

            if n_y0 > 0
                push!(m3, negative_positive / n_y0)  # FN rate among negatives
                push!(m4, negative_negative / n_y0)  # TN rate among negatives
            end

            # Fifth metric: absolute difference in number of predicted vs true positives
            push!(m5, abs(n_y1 - n_yhat1))
        end
    end

    # Compute mean + sample variance for each metric
    m1_mean, m1_var = mean_and_sample_var(m1)
    m2_mean, m2_var = mean_and_sample_var(m2)
    m3_mean, m3_var = mean_and_sample_var(m3)
    m4_mean, m4_var = mean_and_sample_var(m4)
    m5_mean, m5_var = mean_and_sample_var(m5)

    println("Final metrics from: $csv_path")
    println()

    @printf("Metric 1 (positive_positive / n_y1): mean = %.6f, sample var = %.6f (N=%d)\n",
            m1_mean, m1_var, length(m1))
    @printf("Metric 2 (positive_negative / n_y1): mean = %.6f, sample var = %.6f (N=%d)\n",
            m2_mean, m2_var, length(m2))
    @printf("Metric 3 (negative_positive / n_y0): mean = %.6f, sample var = %.6f (N=%d)\n",
            m3_mean, m3_var, length(m3))
    @printf("Metric 4 (negative_negative / n_y0): mean = %.6f, sample var = %.6f (N=%d)\n",
            m4_mean, m4_var, length(m4))
    @printf("Metric 5 (abs(n_y1 - n_yhat1)):   mean = %.6f, sample var = %.6f (N=%d)\n",
            m5_mean, m5_var, length(m5))
end

# Allow running as a script
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia --project=. analysis/final_metrics.jl path/to/raw_results_..._analysis.csv")
        exit(1)
    end
    final_metrics(ARGS[1])
end
