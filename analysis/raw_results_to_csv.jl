using HEALJ
using Serialization
using Printf
using Flux: sigmoid
using Dates

"""
analyze_results(binary_path; out_csv=nothing)

Read the raw test results `.jls` at `binary_path` and write a CSV with
per-protein confusion counts.

Returns the path to the CSV.

to run:

 julia --project=. scripts/analyze_test_results.jl path/to/raw_results_xxx.jls [out.csv]
"""
function analyze_results(binary_path::String; out_csv::Union{Nothing,String}=nothing)
    @info "Deserializing results from $binary_path"
    results = Serialization.deserialize(binary_path)

    if out_csv === nothing
        base   = splitext(basename(binary_path))[1]
        out_dir = dirname(binary_path)
        out_csv = joinpath(out_dir, base * "_analysis.csv")
    end

    @info "Writing analysis CSV to $out_csv"

    open(out_csv, "w") do io
        # Header
        println(io,
            "protein_id," *
            "n_y0,n_y1," *
            "n_yhat0,n_yhat1," *
            "positive_positive," *
            "positive_negative," *
            "negative_positive," *
            "negative_negative"
        )

        for r in results
            pid    = r.graph_pid
            logits = r.logits
            labels = r.labels

            # Ensure they are plain vectors
            y_vec = vec(labels)
            z_vec = vec(logits)

            if length(y_vec) != length(z_vec)
                @warn "Length mismatch between labels and logits; skipping" pid length_y=length(y_vec) length_logits=length(z_vec)
                continue
            end

            # Ground truth booleans (assuming labels in {0,1})
            y_bool = y_vec .> 0.5f0   # equivalent to .== 1f0 if labels are 0/1

            # Counts of 0s and 1s in y
            n_y1 = count(y_bool)
            n_y0 = length(y_bool) - n_y1

            # Predictions: sigmoid + threshold at 0.5
            p_vec    = sigmoid.(z_vec)
            yhat_bool = p_vec .>= 0.5f0

            n_yhat1 = count(yhat_bool)
            n_yhat0 = length(yhat_bool) - n_yhat1

            # Confusion matrix entries
            # positive_positive: true 1, predicted 1 (TP)
            positive_positive = count((y_bool .== 1) .& (yhat_bool .== 1))

            # positive_negative: true 0, predicted 1 (Type I error, FP)
            positive_negative = count((y_bool .== 0) .& (yhat_bool .== 1))

            # negative_positive: true 1, predicted 0 (Type II error, FN)
            negative_positive = count((y_bool .== 1) .& (yhat_bool .== 0))

            # negative_negative: true 0, predicted 0 (TN)
            negative_negative = count((y_bool .== 0) .& (yhat_bool .== 0))

            @printf(io,
                "%s,%d,%d,%d,%d,%d,%d,%d,%d\n",
                pid,
                n_y0, n_y1,
                n_yhat0, n_yhat1,
                positive_positive,
                positive_negative,
                negative_positive,
                negative_negative,
            )
        end
    end

    return out_csv
end

# Allow running as a script
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia --project=. scripts/analyze_test_results.jl path/to/raw_results_xxx.jls [out.csv]")
        exit(1)
    end

    binary_path = ARGS[1]
    out_csv     = length(ARGS) >= 2 ? ARGS[2] : nothing

    out_path = analyze_results(binary_path; out_csv=out_csv)
    println("Wrote analysis CSV to $out_path")
end
