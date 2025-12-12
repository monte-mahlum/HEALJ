using HEALJ
using HEALJ.DataLoader: make_datasets
using Flux
using Random
using Dates
using Serialization
using Printf

# ---------------- Hyperparameters ----------------

const EPOCHS      = 25          # this took ~12 hours on an RTX 5060
const NODE_BUDGET = 8000        # depends on your RAM / GPU VRAM


const LAMBDA  = 0.10f0
const TAU     = 0.10f0
const EPSILON = 0.001f0
const LR      = 1f-3
const SEED    = 42

# ---------------- Main train entrypoint ----------------

function main_train(; epochs::Int      = EPOCHS,
                     node_budget::Int  = NODE_BUDGET,
                     λ::Float32        = LAMBDA,
                     τ::Float32        = TAU,
                     ε::Float32        = EPSILON,
                     lr::Float32       = LR,
                     seed::Int         = SEED)

    println("=== HEALJ training run ===")
    println("epochs       = $epochs")
    println("node_budget  = $node_budget")
    println("λ (lambda)   = $λ")
    println("τ (tau)      = $τ")
    println("ε (epsilon)  = $ε")
    println("learning rate= $lr")
    println("seed         = $seed")

    rng = Random.MersenneTwister(seed)

    # 1. Build datasets (train/val/test)
    println("\n[train] Loading datasets via DataLoader.make_datasets() ...")
    dsets = make_datasets()

    # 2. Build model using train metadata
    println("[train] Building HEALModel from dsets.train metadata ...")
    model = HEALModel(dsets.train; D = 256, K = 16, heads = 1, head_hidden = Int[512])

    # 3. Optimizer
    opt = Flux.Adam(lr)

    # 4. Train (this moves model to GPU if available inside train!)
    println("[train] Starting training loop ...")
    model, history = train!(model, opt, dsets;
                            epochs      = epochs,
                            node_budget = node_budget,
                            λ           = λ,
                            τ           = τ,
                            ε           = ε,
                            rng         = rng)

    println("[train] Training complete, saving artifacts ...")

    # File structure:
    #
    # project_root/
    #   artifacts/
    #     run_yyyymmdd_HHMMSS/
    #       model_yyyymmdd_HHMMSS.jls
    #       history_yyyymmdd_HHMMSS.jls
    #       history_yyyymmdd_HHMMSS.csv
    #     latest_model.jls
    #     latest_history.jls

    project_root  = joinpath(@__DIR__, "..")
    artifacts_dir = joinpath(project_root, "artifacts")
    isdir(artifacts_dir) || mkpath(artifacts_dir)

    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    run_dir   = joinpath(artifacts_dir, "run_$timestamp")
    mkpath(run_dir)

    model_path   = joinpath(run_dir, "model_$timestamp.jls")
    history_path = joinpath(run_dir, "history_$timestamp.jls")
    history_csv  = joinpath(run_dir, "history_$timestamp.csv")

    # 5. Serialize model + history
    Serialization.serialize(model_path,   model)
    Serialization.serialize(history_path, history)

    # 6. Also write a CSV version of history
    open(history_csv, "w") do io
        println(io, "epoch,train_total,train_sup,train_con,val_sup")
        for h in history
            @printf(io, "%d,%.6f,%.6f,%.6f,%.6f\n",
                    h.epoch,
                    h.train_total,
                    h.train_sup,
                    h.train_con,
                    h.val_sup)
        end
    end

    # 7. Maintain "latest" symlinks for convenience with multiple models
    Serialization.serialize(joinpath(artifacts_dir, "latest_model.jls"),   model)
    Serialization.serialize(joinpath(artifacts_dir, "latest_history.jls"), history)

    println("[train] Saved model to:   $model_path")
    println("[train] Saved history to: $history_path")
    println("[train] Saved history CSV to: $history_csv")

    return (model_path   = model_path,
            history_path = history_path,
            history_csv  = history_csv,
            run_dir      = run_dir,
            timestamp    = timestamp)
end

# Run if executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main_train()
end
