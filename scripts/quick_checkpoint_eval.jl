using CSV, DataFrames, Serialization, Statistics
include(joinpath("..", "neural_ode_framework.jl"))
using .ZoneOptNeuralODEFramework

function main(checkpoint_path::AbstractString = joinpath("models", "epoch_210_best.jls"))
    long_df, index_df = ZoneOptNeuralODEFramework.load_sequence_tables()
    examples = ZoneOptNeuralODEFramework.build_sequence_examples(long_df, index_df)
    train_set, val_set, test_set = ZoneOptNeuralODEFramework.split_sequences(examples; seed=42)

    if isempty(test_set)
        println("No test sequences available.")
        return
    end

    model = ZoneOptNeuralODEFramework.build_model(length(first(examples).context), train_set; hidden_dim=64, depth=3)
    if isfile(checkpoint_path)
        ZoneOptNeuralODEFramework.restore_model_checkpoint!(model, checkpoint_path)
        println("Restored checkpoint: ", checkpoint_path)
    else
        println("Checkpoint not found: ", checkpoint_path)
    end

    # Compute test RMSE (mean per-sequence)
    seq_errors = Float64[]
    for seq in test_set
        pred = ZoneOptNeuralODEFramework.predict_sequence(model, seq)
        n = min(size(pred, 2), size(seq.u, 2))
        if n > 0
            push!(seq_errors, sqrt(mean((pred[1:3, 1:n] .- seq.u[1:3, 1:n]).^2)))
        end
    end
    test_rmse = isempty(seq_errors) ? NaN : mean(seq_errors)
    println("Test RMSE (mean per-sequence): ", test_rmse)

    # RMSE by entry type
    grouped = Dict{String, Vector{Float64}}()
    for (i, seq) in enumerate(test_set)
        pred = ZoneOptNeuralODEFramework.predict_sequence(model, seq)
        n = min(size(pred, 2), size(seq.u, 2))
        if n == 0
            continue
        end
        val = sqrt(mean((pred[1:3, 1:n] .- seq.u[1:3, 1:n]).^2))
        push!(get!(grouped, seq.entry_type, Float64[]), val)
    end
    rows = DataFrame(entry_type = String[], n = Int[], rmse = Float64[])
    for key in sort(collect(keys(grouped)))
        vals = grouped[key]
        push!(rows, (entry_type = key, n = length(vals), rmse = mean(vals)))
    end
    println("RMSE by entry type:")
    show(rows)

    # Decision validity using model recommendations vs. actual successful sequences
    eligible = filter(seq -> seq.shot_within_horizon, test_set)
    decision_validity = isempty(eligible) ? NaN : mean([ (ZoneOptNeuralODEFramework.recommend_strategy(model, seq).best_label == seq.entry_type) for seq in eligible ])
    # Baseline: most common entry type
    type_counts = Dict{String, Int}()
    for seq in test_set
        type_counts[seq.entry_type] = get(type_counts, seq.entry_type, 0) + 1
    end
    most_common = isempty(type_counts) ? "" : findmax(collect(values(type_counts)))[2]
    # compute baseline correctly: find the key with max value
    most_common_key = isempty(type_counts) ? "" : first(sort(collect(keys(type_counts)); by = k -> type_counts[k], rev=true))
    eligible2 = filter(seq -> seq.shot_within_horizon, test_set)
    baseline = isempty(eligible2) ? NaN : mean([ seq.entry_type == most_common_key for seq in eligible2 ])
    println("Decision validity: ", decision_validity)
    println("Baseline validity (most-common): ", baseline)

    mkpath("evaluation")
    CSV.write(joinpath("evaluation", "quick_checkpoint_summary.csv"), DataFrame(metric = ["test_rmse", "decision_validity", "baseline_validity"], value = [test_rmse, decision_validity, baseline]))
    CSV.write(joinpath("evaluation", "quick_checkpoint_rmse_by_type.csv"), rows)

    # Also save a small sample of recommended strategies
    rec_df = ZoneOptNeuralODEFramework.strategy_recommendation_stats(model, test_set; max_samples=50, seed=42)
    CSV.write(joinpath("evaluation", "quick_checkpoint_recommendations_sample.csv"), rec_df)

    println("Wrote evaluation/quick_checkpoint_*.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    cp = length(ARGS) >= 1 ? ARGS[1] : joinpath("models", "epoch_210_best.jls")
    main(cp)
end
