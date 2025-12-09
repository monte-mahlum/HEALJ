using HEALJ

dsets = DataLoader.make_datasets()
println("LENGTH TRAINING SET: $(length(dsets.train.graph_refs))")
println("LENGTH VAL SET: $(length(dsets.val.graph_refs))")
println("LENGTH TEST SET: $(length(dsets.test.graph_refs))")

train_plan = DataLoader.make_batch_plan(dsets.train)
val_plan = DataLoader.make_batch_plan(dsets.val)
test_plan = DataLoader.make_batch_plan(dsets.test)

println("NUM TRAINING BATCHES: $(length(train_plan.batches))")
println("NUM VAL BATCHES: $(length(val_plan.batches))")
println("NUM TEST BATCHES: $(length(test_plan.batches))")
println()
println("--------------------------------")
println()
for i in 1:length(train_plan.batches)
    println("Size training batch $(i): $(length(train_plan.batches[i]))")
end
println()
println("--------------------------------")
println()

using Test

let plan = train_plan, ds = dsets.train, budget = 4000
    # 1. Every graph appears exactly once
    all_idxs = vcat(plan.batches...)
    @test sort(all_idxs) == collect(1:length(ds.graph_refs))

    # 2. Node budget is respected
    for (bi, batch_idxs) in enumerate(plan.batches)
        total_nodes = sum(ds.graph_refs[i].nodes for i in batch_idxs)
        @test total_nodes <= budget ||
              (bi == length(plan.batches))  # maybe last batch slightly over if you allow that
    end

    # 3. Make a few batches and check shapes
    for batch_idxs in plan.batches[1:min(end, 3)]
        mb = DataLoader.make_microbatch(Int32.(batch_idxs), ds)
        @test size(mb.H0, 1) == length(mb.node2graph)
        @test size(mb.y, 1) == length(mb.graph_refs)
    end
end


for (i, idxs) in enumerate(train_plan.batches)
    total_nodes = sum(dsets.train.graph_refs[j].nodes for j in idxs)
    println("Batch $i: |graphs|=$(length(idxs)), total_nodes=$total_nodes")
end


