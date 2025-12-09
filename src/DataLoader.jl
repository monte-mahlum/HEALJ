module DataLoader

using HDF5
using Random

# ---------------- Types ----------------

"""
The first two act as METADATA. The MicroBatch type holds more than just metadata.

---GraphRef---
DESCRIPTION
Each object of the GraphRef type points to ONE graph group (numeric key, e.g., "0") inside an HDF5 shard.
`nodes` is cached to make batching fast (without opening the shard). 

USAGE 
All objects of this type will be defined only within a Dataset

---Dataset---
DESCRIPTION
refs stores a collection of pointers to GraphRef objects whose graphs should all 
have the same number of labels and feature dimensions.

USAGE
There will be one object of this type for each of the train/validate/test datasets.

---Datasets---
This one is obvious enough.

---MicroBatch---
Each object of this type will hold the relevant data to obtain a batch of graph groups.
The partition into disjoint graph groups can be be recovered from the node_to_graph attribute. 
Instead of a sparce adjacency matrix, we store edge data in src and 
dst (destination) vectors. 

Suppose mb::MicroBatch. To obtain the GraphRef object corresponding to the ith graph in mb, we call 
mb.graph_refs[i]

---BatchPlan---
This is a lightweight plan for how to batch a dataset. Not much more explanation needed.

"""

struct GraphRef
    path::String
    key::String
    nodes::Int32    # Int32 to save on memory (Int means Int64 for my machine)
end

struct Dataset
    graph_refs::Vector{GraphRef}
    num_labels::Int            # For us, this will be 2748; more natural to leave as Int32, infinitesimal cost anyway.
    feature_dim::Int           # For us, this will be 1300 = 20 (native) + 1280 (ESM1b)
end

struct Datasets
    train::Dataset
    val::Dataset
    test::Dataset
end

struct MicroBatch   
    H0::Matrix{Float32}               # (Ntotal × F)  nodes × features
    src::Vector{Int32}                # (Etotal,) 
    dst::Vector{Int32}                # (Etotal,) 
    y::Matrix{Float32}                # (B × K) graphs × labels
    node2graph::Vector{Int32}         # (Ntotal,) values in 1..B
    graph_refs::Vector{GraphRef}      # (B,)
end

struct BatchPlan
    ds::Dataset
    batches::Vector{Vector{Int32}}   # indices into ds.graph_refs
end








# ---------------- random helpers ----------------

# Ensures a key is nonempty and numeric to load only the graph groups (key 0,1,...) from HDF5
numeric_key(k::AbstractString) = !isempty(k) && all(isdigit, k)


function list_h5_files(dir::String)
    files = String[]
    for (root, _, fnames) in walkdir(dir)
        for f in fnames
            endswith(lowercase(f), ".h5") || continue
            push!(files, joinpath(root, f))
        end
    end
    sort!(files)
    return files
end

"""
Julia reads the matricies from HDF5 in a transposed manner. This function
converts a 2D array to nodes×features (N×F) given expected feature dim F.
Handles both N×F and F×N inputs.

WARNING: In model implementation, must never allow N=F (=1300) where N is total 
number of nodes in a batch.
"""
function ensure_NxF(A::AbstractArray, F::Int; name::String="tensor")
    ndims(A) == 2 || throw(ArgumentError("$name must be 2D, got size=$(size(A))"))
    s = size(A)
    if s[2] == F
        return Array{Float32}(A)
    elseif s[1] == F
        return Array{Float32}(permutedims(A))
    else
        throw(ArgumentError("$name has size=$(s), cannot coerce to N×$F"))
    end
end

"""
Convert edge_index to E×2 Int32, then return (src, dst) as 1-based Int32 vectors.
HEAL preprocessing in Python typically stores 0-based node indices, so we +1 here.
Accepts 2×E or E×2.
"""
function edge_to_srcdst(edge_raw::AbstractArray; name::String="edge_index")
    ndims(edge_raw) == 2 || throw(ArgumentError("$name must be 2D, got size=$(size(edge_raw))"))
    s = size(edge_raw)
    ei = if s[2] == 2
        Array{Int32}(edge_raw)
    elseif s[1] == 2
        Array{Int32}(permutedims(edge_raw))
    else
        throw(ArgumentError("$name must be E×2 or 2×E, got size=$(s)"))
    end
    # 0-based -> 1-based
    ei .+= Int32(1)
    return view(ei, :, 1), view(ei, :, 2)  # src view, dst view (each length E)
end

"""
Infer node count N from a dataset shape that could be N×F or F×N.)
"""
function infer_nodes_from_shape(s::Tuple{Int,Int}, F::Int)
    d1, d2 = s
    if (d1 == F) && !(d2 == F)
        return d2
    elseif (d2 == F) && !(d1 == F)
        return d1
    else
        return max(d1,d2) # fallback
    end
end







# ---------------- Dataset Builder ----------------

"""
Build Dataset by scanning all shards under `dir`.
- Ignores non-numeric keys (e.g. go_terms).
- Caches node counts for fast batching.

feature_dim defaults to 1300 = 20 (native) + 1280 (ESM1b)
"""
function build_dataset(dir::String; feature_dim::Int=1300)
    h5files = list_h5_files(dir)
    isempty(h5files) && throw(ArgumentError("No .h5 files found under $dir"))

    # Determine num_labels from first shard
    num_labels = 0
    h5open(h5files[1], "r") do f
        a = attributes(f)
        if haskey(a, "num_labels")
            nl = read(a["num_labels"])
            num_labels = nl isa Integer ? Int(nl) : Int(round(nl))
        elseif haskey(f, "go_terms")
            num_labels = length(f["go_terms"])
        else
            num_labels = 0
        end
    end
    num_labels > 0 || throw(ArgumentError("Could not determine num_labels from shards in $dir"))

    graph_refs = GraphRef[]
    for p in h5files
        h5open(p, "r") do f
            for k in keys(f)
                ks = String(k)
                numeric_key(ks) || continue
                graph_group = f[ks]

                # Prefer x for node count; fallback to native_x
                N = 0
                if haskey(graph_group, "x")
                    N = infer_nodes_from_shape(size(graph_group["x"]), 1280)
                elseif haskey(g, "native_x")
                    N = infer_nodes_from_shape(size(graph_group["native_x"]), 20)
                else
                    continue
                end
                N > 0 || continue
                push!(graph_refs, GraphRef(p, ks, Int32(N)))
            end
        end
    end

    isempty(graph_refs) && throw(ArgumentError("Found 0 graph groups under $dir (did you point to the right folder?)"))
    return Dataset(graph_refs, num_labels, feature_dim)
end







# ---------------- Datasets Builder ----------------------

"""
Build train/val/test Dataset objects from shard directories.
"""
function make_datasets(;
    train_dir::AbstractString = "preprocess/data/processed/shards/train",
    val_dir::AbstractString   = "preprocess/data/processed/shards/validate",
    test_dir::AbstractString  = "preprocess/data/processed/shards/test",
    feature_dim::Int = 1300,
)
    train = build_dataset(String(train_dir); feature_dim=feature_dim)
    val   = build_dataset(String(val_dir);   feature_dim=feature_dim)
    test  = build_dataset(String(test_dir);  feature_dim=feature_dim)
    return Datasets(train, val, test)
end









# ---------------- MicroBatch Helper (graph loading) ----------------

"""
Load a single graph and return:
- H0: N×F (nodes×features)
- (src, dst): vectors Int32 (1-based)
- y: Vector Float32 length L

Set strict=false to skip malformed graphs (returns `nothing`).
"""
function load_graph(ref::GraphRef, ds::Dataset; use_native::Bool=true, use_esm::Bool=true, strict::Bool=true)
    required = ["edge_index", "y"]
    H0 = nothing
    src = nothing
    dst = nothing
    y = nothing

    try
        h5open(ref.path, "r") do f
            haskey(f, ref.key) || throw(ArgumentError("Missing group key $(ref.key) in $(ref.path)"))
            g = f[ref.key]

            for r in required
                haskey(g, r) || throw(ArgumentError("Graph $(ref.key) missing dataset '$r' in $(ref.path)"))
            end

            # Edge list
            e_raw = read(g["edge_index"])
            s_view, d_view = edge_to_srcdst(e_raw)
            src = Vector{Int32}(s_view)
            dst = Vector{Int32}(d_view)

            # Labels
            y_raw = read(g["y"])
            y = Vector{Float32}(y_raw)
            length(y) == ds.num_labels || throw(ArgumentError("y length $(length(y)) != num_labels $(ds.num_labels) for path $(ref.path), key $(ref.key)"))

            # Features
            feats = Matrix{Float32}(undef, 0, 0)

            if use_native
                haskey(g, "native_x") || throw(ArgumentError("Graph group at path $(ref.path), key $(ref.key) is missing native_x"))
                native = ensure_NxF(read(g["native_x"]), 20; name="native_x")
                feats = native
            end

            if use_esm
                haskey(g, "x") || throw(ArgumentError("Graph group at path $(ref.path), key $(ref.key) is missing x (ESM)"))
                x = ensure_NxF(read(g["x"]), 1280; name="x (ESM)")
                feats = (size(feats, 1) == 0) ? x : hcat(feats, x)
            end

            H0 = feats  # N×F

            # Validate node counts agree
            N = size(H0, 1)
            (length(src) == length(dst)) || throw(ArgumentError("src/dst length mismatch for path $(ref.path), key $(ref.key)"))
            if !isempty(src)
                maxidx = max(maximum(src), maximum(dst))
                minidx = min(minimum(src), minimum(dst))
                (minidx >= 1) || throw(ArgumentError("edge_index has indices < 1 after +1 (unexpected) for path $(ref.path), key $(ref.key)"))
                (maxidx <= N) || throw(ArgumentError("edge_index references node $maxidx but N=$N for path $(ref.path), key $(ref.key)"))
            end
        end
    catch e
        if strict
            rethrow(e)
        else
            return nothing
        end
    end

    return (H0::Matrix{Float32}, src::Vector{Int32}, dst::Vector{Int32}, y::Vector{Float32})
end











# ---------------- Microbatch builder (disjoint union) ----------------

"""
Build one microbatch by disjoint union batching:
- graph_refs: Vector{GraphRef}
- H0_batch: Ntotal×F
- src/dst offset per graph
- y_batch: B×K
- node2graph: Ntotal

strict=false will skip bad graphs and still return a batch if at least 1 graph survives.
"""
function make_microbatch(batch_idxs::Vector{Int32}, ds::Dataset; use_native::Bool=true, use_esm::Bool=true, strict::Bool=true)
    loaded = Tuple{Matrix{Float32}, Vector{Int32}, Vector{Int32}, Vector{Float32}, GraphRef}[]
    
    refs = [ds.graph_refs[i] for i in batch_idxs]
    for r in refs
        g = load_graph(r, ds; use_native=use_native, use_esm=use_esm, strict=strict)
        g === nothing && continue
        push!(loaded, (g..., r))
    end
    isempty(loaded) && throw(ArgumentError("No valid graphs in requested batch"))

    B = length(loaded)
    F = size(loaded[1][1], 2)

    node_counts = [size(t[1], 1) for t in loaded]
    edge_counts = [length(t[2]) for t in loaded]
    Ntotal = sum(node_counts)
    Etotal = sum(edge_counts)

    H0_batch = Matrix{Float32}(undef, Ntotal, F)
    src_batch = Vector{Int32}(undef, Etotal)
    dst_batch = Vector{Int32}(undef, Etotal)
    y_batch = Matrix{Float32}(undef, B, ds.num_labels)
    node2graph = Vector{Int32}(undef, Ntotal)
    graph_refs = Vector{GraphRef}(undef, B)

    node_off = 0
    edge_off = 0
    for i in 1:B
        H0, src, dst, y, graph_ref = loaded[i]
        size(H0, 2) == F || throw(ArgumentError("Feature dim mismatch inside batch; expected F=$F got $(size(H0,2))"))
        N = size(H0, 1)
        E = length(src)

        H0_batch[node_off+1:node_off+N, :] .= H0
        fill!(view(node2graph, node_off+1:node_off+N), Int32(i))
        y_batch[i, :] .= y
        graph_refs[i] = graph_ref

        # Offset edges by node_off
        @inbounds for e in 1:E
            src_batch[edge_off+e] = src[e] + Int32(node_off)
            dst_batch[edge_off+e] = dst[e] + Int32(node_off)
        end

        node_off += N
        edge_off += E
    end

    return MicroBatch(H0_batch, src_batch, dst_batch, y_batch, node2graph, graph_refs)
end










# ---------------- Batch Planner ----------------

"""
Create a lightweight batch plan: a vector of index vectors.
Each inner Vector{Int} is the graph indices in the dataset for one microbatch.
"""
function make_batch_plan(
    ds::Dataset;
    node_budget::Int = 4000,
    shuffle::Bool = true,
    rng = Random.default_rng(),
)
    idxs = collect(eachindex(ds.graph_refs))
    shuffle && Random.shuffle!(rng, idxs)

    batches = Vector{Vector{Int32}}()
    i = 1
    while i <= length(idxs)
        batch_idxs = Int32[]
        nodes = 0

        while i <= length(idxs)
            idx = idxs[i]
            ref = ds.graph_refs[idx]
            N = ref.nodes

            if !isempty(batch_idxs) && (nodes + N > node_budget)
                break
            end

            push!(batch_idxs, Int32(idx))
            nodes += N
            i += 1
        end

        isempty(batch_idxs) || push!(batches, batch_idxs)
    end

    return BatchPlan(ds, batches)
end


end # module

