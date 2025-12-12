#!/usr/bin/env julia

using HEALJ
using HEALJ.DataLoader: make_datasets, Dataset
using Flux
using Random
using Dates

# ---------------- Hyperparameter grid ----------------

lambda_values = Float32[0.01, 0.1, 1.0]
tau_values    = Float32[0.1, 0.5, 1.0]
eps_values    = Float32[1.0, 0.1, 0.001]   # ε values you requested

epochs      = 10
node_budget = 4000          # adjust if you want

# ---------------- Datasets & model builder ----------------

println("Loading datasets...")
dsets = make_datasets()

# Helper to build a fresh model for each run
build_model(ds::Dataset) = HEALModel(ds; D = 256, K = 16, heads = 1, head_hidden = Int[512])

# ---------------- Logging setup ----------------

project_root = joinpath(@__DIR__, "..")
analysis_dir     = joinpath(project_root, "analysis")
isdir(analysis_dir) || mkpath(analysis_dir)

timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
logfile   = joinpath(analysis_dir, "grid_lambda_tau_eps_$(timestamp).csv")

open(logfile, "w") do io
    # CSV header
    println(io, "lambda,tau,eps,epoch,train_total,train_sup,train_con,val_sup")

    # ------------- Grid search -------------
    for λ in lambda_values
        for τ in tau_values
            for ε in eps_values
                println("Starting run: λ = $(λ), τ = $(τ), ε = $(ε)")

                # fresh RNG and model each run
                rng   = Random.MersenneTwister(42)
                model = build_model(dsets.train)
                opt   = Flux.Adam(1f-3)

                # NOTE: your modified Train.jl should accept `ε` keyword
                _, history = train!(model, opt, dsets;
                                    epochs = epochs,
                                    node_budget = node_budget,
                                    λ = λ,
                                    τ = τ,
                                    ε = ε,
                                    rng = rng)

                # write one row per epoch for this (λ, τ, ε)
                for h in history
                    println(io,
                        "$(λ),$(τ),$(ε)," *
                        "$(h.epoch),$(h.train_total),$(h.train_sup),$(h.train_con),$(h.val_sup)",
                    )
                end
            end
        end
    end
end

println("Saved all histories to $logfile")
