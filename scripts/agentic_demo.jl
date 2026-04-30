using CSV, DataFrames, Random, Plots

repo_root = normpath(joinpath(dirname(@__FILE__), ".."))
include(joinpath(repo_root, "neural_ode_framework.jl"))
using .ZoneOptNeuralODEFramework

function main()
    long_df, index_df = ZoneOptNeuralODEFramework.load_sequence_tables()
    examples = ZoneOptNeuralODEFramework.build_sequence_examples(long_df, index_df)
    train_set, val_set, test_set = ZoneOptNeuralODEFramework.split_sequences(examples; seed=42)

    println("Sequences loaded: total=$(length(examples)), test=$(length(test_set))")

    model = ZoneOptNeuralODEFramework.build_model(length(first(examples).context), train_set; hidden_dim=64, depth=3)
    # restore selected checkpoint
    ck = joinpath(repo_root, "models", "epoch_210_best.jls")
    if isfile(joinpath(repo_root, "models", "SELECTED_CHECKPOINT.md"))
        txt = read(joinpath(repo_root, "models", "SELECTED_CHECKPOINT.md"), String)
        for line in split(txt, '\n')
            if startswith(strip(line), "- Path:")
                p = strip(split(line, ":", limit=2)[2])
                ck = joinpath(repo_root, p)
            end
        end
    end
    if isfile(ck)
        println("Restoring checkpoint: ", ck)
        ZoneOptNeuralODEFramework.restore_model_checkpoint!(model, ck)
    else
        println("Checkpoint not found: ", ck)
    end

    rng = MersenneTwister(123)
    sample_idxs = rand(rng, 1:length(test_set), min(3, length(test_set)))
    out_dir = joinpath(repo_root, "presentation", "figures")
    mkpath(out_dir)

    results = DataFrame(seq_id=Int[], best_strategy=String[], confidence=Float64[])

    for idx in sample_idxs
        seq = test_set[idx]
        rec = ZoneOptNeuralODEFramework.recommend_action(model, seq)
        println("Seq $(seq.seq_id): recommended=$(rec.best_label) (confidence=$(round(rec.confidence,digits=3)))")
        push!(results, (seq_id=seq.seq_id, best_strategy=rec.best_label, confidence=rec.confidence))

        # Plot actual vs predicted vs strategy trajectories
        pred = ZoneOptNeuralODEFramework.predict_sequence(model, seq)
        plt = plot(seq.u[1, :], seq.u[2, :], label="actual", lw=2)
        plot!(plt, pred[1, :], pred[2, :], label="model_pred", lw=2, ls=:dash)

        # plot each candidate strategy trajectory
        for (sname, traj) in rec.trajectories
            plot!(plt, traj[1, :], traj[2, :], label=string(sname), lw=1, ls=:dot)
        end

        title!(plt, "Seq $(seq.seq_id): recommendation=$(rec.best_label)")
        xlabel!(plt, "x (ft)")
        ylabel!(plt, "y (ft)")
        savefig(plt, joinpath(out_dir, "agentic_demo_seq_$(seq.seq_id).png"))
    end

    CSV.write(joinpath(out_dir, "agentic_demo_results.csv"), results)
    println("Agentic demo complete. Results and plots in: ", out_dir)
end

main()
