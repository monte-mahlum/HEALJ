using Random
using Flux
using Statistics
using .DataLoader: Datasets, Dataset, make_batch_plan, make_microbatch

using CUDA    
using Functors: fmap

# GPU if available, otherwise leave as-is.
to_device(x) = CUDA.functional() ? cu(x) : x

function to_device_mb(mb)
    return (
        H0         = to_device(mb.H0),   # features to GPU if available
        src        = mb.src,             # indices stay on CPU
        dst        = mb.dst,
        y          = to_device(mb.y),    # labels can go to GPU
        node2graph = mb.node2graph,      # indices stay on CPU
        graph_refs = mb.graph_refs,      # metadata on CPU
    )
end





"""
The model: encoder -> HGT -> MLP head
"""
struct HEALModel
    enc  :: GCNEncoder
    hgt  :: HGT
    head :: MLPHead
end

Flux.@layer HEALModel trainable = (enc, hgt, head,)

"""
    HEALModel(F, D, K, C; heads=1, head_hidden=[512])

Builds a HEAL model with:
- input feature dim `F`
- graph embedding dim `D`
- `K` super-nodes per graph in HGT
- `C` output labels
"""
function HEALModel(F::Int, D::Int, K::Int, C::Int;
                   heads::Int = 1,
                   head_hidden = Int[512])
    enc  = GCNEncoder([F, D, D])
    hgt  = HGT(D; K=K, heads=heads)
    head = MLPHead(D, head_hidden, C)
    return HEALModel(enc, hgt, head)
end

"""
    HEALModel(ds::Dataset; D=256, K=16, heads=1, head_hidden=[512])

Convenience constructor using metadata from a `Dataset`:
- `F` = `ds.feature_dim`
- `C` = `ds.num_labels`
"""
function HEALModel(ds::Dataset;
                   D::Int = 256,
                   K::Int = 16,
                   heads::Int = 1,
                   head_hidden = Int[512])
    return HEALModel(ds.feature_dim, D, K, ds.num_labels;
                     heads = heads, head_hidden = head_hidden)
end

# Move the whole HEALModel to device (recursively through its sublayers).
function to_device(m::HEALModel)
    return fmap(to_device, m)
end




# Forward pass
"""
    graph_embeddings(m, mb)

Compute graph-level embeddings (B×D) for a microbatch-like object `mb`
with fields:
- `H0`          :: N×F
- `src`, `dst`  :: edge lists
- `node2graph`  :: length N vector in 1..B
"""
function graph_embeddings(m::HEALModel, mb)
    Henc = m.enc(mb.H0, mb.src, mb.dst)
    return m.hgt(Henc, mb.src, mb.dst, mb.node2graph)  # B×D
end

"""
    model_forward(m, mb) -> (logits, Z)

- `Z`      :: B×D graph embeddings
- `logits` :: B×C classifier logits
"""
function model_forward(m::HEALModel, mb)
    Z = graph_embeddings(m, mb)          # B×D
    logits = m.head(Float32.(Z))         # B×C
    return logits, Z
end



# One training step
"""
    train_step!(m, opt_state, mb; λ=0.1f0, τ=0.5f0, rng=Random.default_rng())

One optimization step on a microbatch `mb`.

- Contrastive term uses two perturbed node-feature views (same graph structure).
- Supervised term uses the clean batch.

`opt_state` is the optimizer state created via `Flux.setup(opt, m)`.

Returns the total loss (scalar).
"""
function train_step!(m::HEALModel, opt_state, mb;
                     λ::Real = 0.1f0,
                     τ::Real = 0.5f0,
                     ε::Float32 = 0.1f0,
                     rng = Random.default_rng())

    # Two perturbed views for contrastive term
    H1 = perturb_nodes(mb.H0; ε = ε, rng = rng)
    H2 = perturb_nodes(mb.H0; ε = ε, rng = rng)

    mb1_cpu = (
        H0         = H1,
        src        = mb.src,
        dst        = mb.dst,
        y          = mb.y,
        node2graph = mb.node2graph,
        graph_refs = mb.graph_refs,
    )
    mb2_cpu = (
        H0         = H2,
        src        = mb.src,
        dst        = mb.dst,
        y          = mb.y,
        node2graph = mb.node2graph,
        graph_refs = mb.graph_refs,
    )

    # 2. Move all three microbatches to the same device as the model
    mb_clean = to_device_mb(mb)
    mb1      = to_device_mb(mb1_cpu)
    mb2      = to_device_mb(mb2_cpu)

    stats_ref = Ref((Lsup = 0f0, Lcon = 0f0))

    loss, back = Flux.withgradient(m) do m2
        Z1 = graph_embeddings(m2, mb1)
        Z2 = graph_embeddings(m2, mb2)
        logits, _ = model_forward(m2, mb_clean)     # supervised on clean batch
        L, stats = total_loss(logits, mb_clean.y, Z1, Z2; λ = λ, τ = τ)
        stats_ref[] = stats
        return L
    end

    grad = back[1]
    Flux.update!(opt_state, m, grad)

    stats = stats_ref[]
    return (Ltotal = loss,
            Lsup   = stats.Lsup,
            Lcon   = stats.Lcon)
end




# Epoch-level
"""
    train_epoch!(m, opt_state, ds; node_budget=4000, λ=0.1f0, τ=0.5f0, rng=Random.default_rng())

Run one training epoch over dataset `ds` (a `Dataset`, e.g. `dsets.train`).

Returns mean training loss over all batches.
"""
function train_epoch!(m::HEALModel,
                      opt_state,
                      ds::Dataset;
                      node_budget::Int = 4000,
                      λ::Real = 0.1f0,
                      τ::Real = 0.5f0,
                      ε::Float32 = 0.1f0,
                      rng = Random.default_rng())

    plan = make_batch_plan(ds; node_budget = node_budget,
                               shuffle     = true,
                               rng         = rng)

    totals = Float32[]
    sups   = Float32[]
    cons   = Float32[]

    for batch_idxs in plan.batches
        mb = make_microbatch(batch_idxs, ds;
                             use_native = true, use_esm = true, strict = true)
        step = train_step!(m, opt_state, mb; λ = λ, τ = τ, ε = ε, rng = rng)
        push!(totals, Float32(step.Ltotal))
        push!(sups,   Float32(step.Lsup))
        push!(cons,   Float32(step.Lcon))
    end

    return (Ltotal = isempty(totals) ? NaN : mean(totals),
            Lsup   = isempty(sups)   ? NaN : mean(sups),
            Lcon   = isempty(cons)   ? NaN : mean(cons))
end


"""
    validate_epoch(m, ds; node_budget=4000)

Simple validation loop that computes the *supervised* BCE loss on `ds`
(no contrastive term, no gradients).

Returns mean validation loss over all batches.
"""
function validate_epoch(m::HEALModel,
                        ds::Dataset;
                        node_budget::Int = 4000)
    plan = make_batch_plan(ds; node_budget = node_budget,
                               shuffle     = false)

    losses = Float32[]

    for batch_idxs in plan.batches
        mb_cpu = make_microbatch(batch_idxs, ds;
                             use_native = true, use_esm = true, strict = true)
        mb = to_device_mb(mb_cpu)

        logits, _ = model_forward(m, mb)
        Lsup = supervised_bce_loss(logits, mb.y)
        push!(losses, Float32(Lsup))
    end

    return isempty(losses) ? NaN : mean(losses)
end




# Full train. Note, as written now this will not save the model after each epoch.
"""
    train!(m, opt, dsets;
           epochs=10, node_budget=4000,
           λ=0.1f0, τ=0.5f0, rng=Random.default_rng())

High-level training loop over `dsets::Datasets` (with `.train` and `.val`).

- `opt` is a Flux optimiser, e.g. `Adam(1f-3)`.
- Internally we build an optimizer *state* with `Flux.setup(opt, m)` and
  reuse it across all epochs.

Returns `(m, history)` where `history` is a vector of named tuples
`(epoch, train_loss, val_loss)`.
"""
function train!(m::HEALModel,
                opt,
                dsets::Datasets;
                epochs::Int = 10,
                node_budget::Int = 4000,
                λ::Real = 0.1f0,
                τ::Real = 0.5f0,
                ε::Float32 = 0.1f0,
                rng = Random.default_rng())

     m = to_device(m)

    opt_state = Flux.setup(opt, m)

    hist = Vector{NamedTuple}(undef, epochs)

    for epoch in 1:epochs
        train_stats = train_epoch!(m, opt_state, dsets.train;
                                   node_budget = node_budget,
                                   λ = λ, τ = τ, ε = ε, rng = rng)

        val_sup = validate_epoch(m, dsets.val;
                                 node_budget = node_budget)

        hist[epoch] = (
            epoch      = epoch,
            train_total = train_stats.Ltotal,
            train_sup   = train_stats.Lsup,
            train_con   = train_stats.Lcon,
            val_sup     = val_sup,
        )

        @info "Epoch $epoch" train_total = train_stats.Ltotal
                             train_sup   = train_stats.Lsup
                             train_con   = train_stats.Lcon
                             val_sup     = val_sup
    end

    return m, hist
end

