using HEALJ
using HEALJ.DataLoader: build_dataset, make_batch_plan, make_microbatch
using HDF5
using Random
using Dates
using Serialization
using Statistics
using Printf

const NODE_BUDGET = 8000
const SEED        = 123

# pid helper
function get_graph_pid(ref::HEALJ.DataLoader.GraphRef)
    h5open(ref.path, "r") do f
        g     = f[ref.key]
        atts  = attributes(g)

        if haskey(atts, "id")
            return String(read(atts["id"]))
        else
            return ""
        end
    end
end


# Main
function main_test(; model_path::AbstractString = "",
                     node_budget::Int          = NODE_BUDGET,
                     seed::Int                 = SEED,
                     test_dir::AbstractString  = "preprocess/data/processed/shards/test")

    project_root  = joinpath(@__DIR__, "..")
    artifacts_dir = joinpath(project_root, "artifacts")
    isdir(artifacts_dir) || mkpath(artifacts_dir)

    # 1. Locate model file
    if isempty(model_path)
        model_path = joinpath(artifacts_dir, "latest_model.jls")
    end
    @assert isfile(model_path) "Model file not found at $model_path. Run train.jl first."

    println("=== HEALJ test run ===")
    println("[test] Using model: $model_path")
    println("[test] node_budget = $node_budget, seed = $seed")
    println("[test] test_dir    = $test_dir")

    # 2. Load model
    println("[test] Deserializing model ...")
    model = Serialization.deserialize(model_path)

    # 3. Build test dataset
    println("[test] Building test Dataset via build_dataset(test_dir) ...")
    test_ds = build_dataset(test_dir)

    # 4. Build batch plan
    rng = Random.MersenneTwister(seed)
    println("[test] Creating test batch plan ...")
    plan = make_batch_plan(test_ds;
                           node_budget = node_budget,
                           shuffle     = false,
                           rng         = rng)

    # Container for per-graph results
    results = NamedTuple[]

    for batch_idxs in plan.batches
        # CPU microbatch with real data
        mb_cpu = make_microbatch(batch_idxs, test_ds;
                                 use_native = true, use_esm = true, strict = true)

        # Move features + labels to GPU if available
        mb = HEALJ.to_device_mb(mb_cpu)

        # Forward pass
        logits, _ = HEALJ.model_forward(model, mb)

        # Work with CPU copies for bookkeeping
        logits_cpu = Array(logits)         # B×C
        y_batch    = mb_cpu.y              # B×C on CPU
        refs       = mb_cpu.graph_refs
        B, C = size(y_batch)

        for i in 1:B
            ref      = refs[i]
            graph_pid = get_graph_pid(ref)

            logit_i = vec(view(logits_cpu, i, :))  # C
            y_i     = vec(view(y_batch,    i, :))  # C

            push!(results, (
                graph_pid     = String(graph_pid),
                graph_ref    = ref,
                logits       = Float32.(logit_i),
                labels       = Float32.(y_i),
            ))
        end
    end

    # 6. Save results
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    out_dir   = joinpath(artifacts_dir, "test_results_$timestamp")
    mkpath(out_dir)

    binary_path = joinpath(out_dir, "raw_results_$timestamp.jls")

    Serialization.serialize(binary_path, results)
    println("[test] Saved detailed test results to: $binary_path")

    return (results_path = binary_path,
            out_dir      = out_dir)
end

# Run if executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main_test()
end
