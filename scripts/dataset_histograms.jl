# scripts/dataset_histograms.jl
# Usage:
#   julia --project=. scripts/dataset_histograms.jl
# Optional:
#   julia --project=. scripts/dataset_histograms.jl <train_dir> <val_dir> <test_dir>

using Printf
using Statistics
using HDF5

# Load your project code (DataLoader)
include(joinpath(@__DIR__, "..", "src", "HEALJ.jl"))
using .HEALJ
const DL = HEALJ.DataLoader

# Plotting
try
    @eval using Plots
catch e
    println("ERROR: Plots.jl not installed in this Julia project.")
    println("Fix: julia --project=. -e 'using Pkg; Pkg.add(\"Plots\")'")
    rethrow(e)
end

# --- shape helpers (no heavy reads) ---

# Infer node count N from a 2D shape where you know feature dim F.
# Works for both N×F and F×N.
function infer_N_from_shape(s::Tuple{Int,Int}, F::Int; name::String="tensor")
    d1, d2 = s
    if d1 == F && d2 != F
        return d2
    elseif d2 == F && d1 != F
        return d1
    else
        # fallback: if neither dimension matches F, we can't be sure;
        # choose max as a defensive fallback (but this should be rare)
        return max(d1, d2)
    end
end

# Edge count E from edge_index shape, accepting 2×E or E×2.
function infer_E_from_edge_index_shape(s::Tuple{Int,Int})
    d1, d2 = s
    if d1 == 2 && d2 != 2
        return d2
    elseif d2 == 2 && d1 != 2
        return d1
    else
        return max(d1, d2)
    end
end

# Get N and E for one graph group without reading big datasets
function graph_counts(path::String, key::String)
    h5open(path, "r") do f
        g = f[key]

        # edges
        E = haskey(g, "edge_index") ? infer_E_from_edge_index_shape(size(g["edge_index"])) : 0

        # nodes: prefer native_x (20 dims) as it's unambiguous, else x (1280 dims)
        N = 0
        if haskey(g, "native_x")
            N = infer_N_from_shape(size(g["native_x"]), 20; name="native_x")
        elseif haskey(g, "x")
            N = infer_N_from_shape(size(g["x"]), 1280; name="x")
        end

        return N, E
    end
end

function collect_counts(ds::DL.Dataset)
    nodes = Int[]
    edges = Int[]
    for r in ds.refs
        N, E = graph_counts(r.path, r.key)
        (N > 0 && E > 0) || continue
        push!(nodes, N)
        push!(edges, E)
    end
    return nodes, edges
end

function save_pretty_hist(vals::Vector{Int}; title::String, xlabel::String, outfile::String, bins::Int=60)
    v = Float64.(vals)
    μ = mean(v)

    p = histogram(
        v;
        bins=bins,
        normalize=:pdf,
        xlabel=xlabel,
        ylabel="Probability density",
        title=title,
        legend=false,
        framestyle=:box,
        grid=true,
        dpi=200,
        size=(950,520),
    )

    vline!([μ]; label=false)
    yl = ylims(p)
    annotate!(μ, yl[2]*0.92, text("mean = $(round(μ, digits=1))", 11))

    savefig(p, outfile)
end

# -------------------------
# Main
# -------------------------
train_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath("preprocess","data","processed","shards","train")
val_dir   = length(ARGS) >= 2 ? ARGS[2] : joinpath("preprocess","data","processed","shards","validate")
test_dir  = length(ARGS) >= 3 ? ARGS[3] : joinpath("preprocess","data","processed","shards","test")

println("Building datasets from:")
println("  train:    $train_dir")
println("  validate: $val_dir")
println("  test:     $test_dir")

train_ds    = DL.build_dataset(train_dir)
validate_ds = DL.build_dataset(val_dir)
test_ds     = DL.build_dataset(test_dir)

println("\nGraph counts by refs:")
@printf("  train:    %d\n", length(train_ds.refs))
@printf("  validate: %d\n", length(validate_ds.refs))
@printf("  test:     %d\n\n", length(test_ds.refs))

n_tr, e_tr = collect_counts(train_ds)
n_va, e_va = collect_counts(validate_ds)
n_te, e_te = collect_counts(test_ds)

nodes_all = vcat(n_tr, n_va, n_te)
edges_all = vcat(e_tr, e_va, e_te)

@printf("Usable graphs for nodes hist: %d\n", length(nodes_all))
@printf("Usable graphs for edges hist: %d\n", length(edges_all))

mkpath("artifacts")
save_pretty_hist(nodes_all;
    title="Nodes per graph (train+validate+test)",
    xlabel="Nodes (residues)",
    outfile=joinpath("artifacts","hist_nodes_per_graph_pdf.png"),
    bins=70
)
save_pretty_hist(edges_all;
    title="Edges per graph (train+validate+test)",
    xlabel="Edges",
    outfile=joinpath("artifacts","hist_edges_per_graph_pdf.png"),
    bins=70
)

println("\nWrote:")
println("  artifacts/hist_nodes_per_graph_pdf.png")
println("  artifacts/hist_edges_per_graph_pdf.png")
