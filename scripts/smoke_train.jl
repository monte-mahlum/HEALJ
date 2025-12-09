using Random
using Flux
using HEALJ

# -------------------------------------------------------
# Smoke test for GCN -> HGT -> MLPHead
# -------------------------------------------------------

Random.seed!(1234)

# Hyperparameters
Ntotal    = 60          # total nodes in microbatch
D_in      = 32          # input feature dim (e.g. from ESM / projection)
D_gcn     = 32          # GCN output dim (keep equal for simplicity)
K         = 4           # supernodes per graph
heads     = 1           # number of HGT heads
B         = 3           # number of graphs in batch
num_labels = 10         # pretend we have 10 labels (for test)

# 1) Fake node features H0 :: Ntotal×D_in
H0 = randn(Float32, Ntotal, D_in)

# 2) Fake edges: connect each node to the next within the whole batch
src = Int32.(collect(1:Ntotal-1))
dst = Int32.(collect(2:Ntotal))

# 3) node2graph: split nodes contiguously into B graphs
#    e.g. for Ntotal = 60, B = 3 → 20 nodes per graph
nodes_per_graph = Ntotal ÷ B
node2graph = similar(src, Int32, Ntotal)
for g in 1:B
    a = (g-1)*nodes_per_graph + 1
    b = g*nodes_per_graph
    node2graph[a:b] .= Int32(g)
end

println("Ntotal    = $Ntotal")
println("D_in      = $D_in")
println("B (graphs)= $B")
println()

# ---------------- GCN ----------------
println("---- GCN ----")
gcn = GCNEncoder([D_in, D_gcn])   # just one layer D_in -> D_gcn
H1  = gcn(H0, src, dst)           # Ntotal×D_gcn

println("H0 size: ", size(H0))
println("H1 size (after GCN): ", size(H1))
println()

# ---------------- HGT ----------------
println("---- HGT ----")
hgt = HGT(D_gcn; K=K, heads=heads)
Z   = hgt(H1, src, dst, node2graph)   # B×D_gcn

println("Z size (HGT output, B×D): ", size(Z))
println()

# ---------------- MLPHead ----------------
println("---- MLPHead ----")
head = MLPHead(D_gcn, [D_gcn], num_labels)   # simple 1-hidden-layer MLP
Yhat = head(Z)                               # B×num_labels logits

println("Yhat size (B×num_labels): ", size(Yhat))
println()

println("GCN-HGT-MLP smoke test completed.")
