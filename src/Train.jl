# SKETCH

# optional helper
function to_device(mb::MicroBatch)
    return MicroBatch(
        cu(mb.H0),
        cu(mb.src),
        cu(mb.dst),
        cu(mb.y),
        cu(mb.node2graph),
        mb.graph_refs,  # stay on CPU
    )
end

plan = make_batch_plan(train_ds; node_budget=4000)

for batch_idxs in plan.batches
    mb_cpu = make_microbatch(batch_idxs, train_ds; use_native=true, use_esm=true)
    mb_gpu = to_device(mb_cpu)

    train_step!(model, mb_gpu)
end




"""
SO SIMPLE TO RUN MODEL
Need functions model_forward and update_val_metrics
"""
val_plan  = make_batch_plan(val_ds;  node_budget=4000, shuffle=false)
test_plan = make_batch_plan(test_ds; node_budget=4000, shuffle=false)

for batch_idxs in val_plan.batches
    mb_cpu = make_microbatch(batch_idxs, val_ds)
    mb_gpu = to_device(mb_cpu)

    y_pred = model_forward(mb_gpu)
    update_val_metrics!(metrics, y_pred, mb_gpu.y)
end
