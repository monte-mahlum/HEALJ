using HEALJ
using HEALJ.DataLoader: make_datasets, load_graph
using Statistics

# Computes the average number of nodes and edges per graph in each dataset split

function avg_stats_for_split(name::String, dset)
    ngraphs = length(dset.graph_refs)
    println("Computing stats for $name split with $ngraphs graphs...")

    node_counts = Vector{Int}(undef, ngraphs)
    edge_counts = Vector{Int}(undef, ngraphs)

    for (i, ref) in enumerate(dset.graph_refs)
        H0, src, dst, y = load_graph(ref, dset)

        # Assume H0 is N×F, src/dst are edge index vectors
        node_counts[i] = size(H0, 1)
        edge_counts[i] = length(src)
    end

    println("=== $name split ===")
    println("  # graphs: ", ngraphs)
    println("  avg nodes per graph: ", mean(node_counts))
    println("  avg edges per graph: ", mean(edge_counts))
    println()
end

if abspath(PROGRAM_FILE) == @__FILE__
    dsets = make_datasets()

    avg_stats_for_split("train", dsets.train)
    avg_stats_for_split("val",   dsets.val)
    avg_stats_for_split("test",  dsets.test)
end
