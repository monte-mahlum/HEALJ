using Dates

include("train.jl")
include("test.jl")

"""
To run:
    julia --project=. scripts/train_and_test.jl
"""

function main()
    println("=== HEALJ: train_and_test driver ===")
    t_start = Dates.now()
    println("[driver] Starting training at $t_start")

    train_info = main_train()  # uses defaults from train.jl
    println("[driver] Training finished at $(Dates.now()).")

    # Run test on the freshly-trained model
    test_info = main_test(model_path = train_info.model_path)

    t_end = Dates.now()
    println("[driver] All done at $t_end")
    println("[driver] Total wall-clock time: $(t_end - t_start)")
    println("[driver] Model / history directory: $(train_info.run_dir)")
    println("[driver] Test outputs directory:   $(test_info.out_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
