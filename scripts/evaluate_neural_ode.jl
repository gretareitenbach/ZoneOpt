using CSV
using DataFrames
using Statistics
using Random
using Dates

include("../src/neural_ode_framework.jl")
using .ZoneOptNeuralODEFramework

Base.@kwdef mutable struct EvalConfig
    long_path::String = joinpath("processed", "neural_ode_sequences_long.csv")
    index_path::String = joinpath("processed", "neural_ode_sequence_index.csv")
    out_dir::String = "evaluation"
    epochs::Int = 20
    batch_size::Int = 16
    learning_rate::Float64 = 1e-3
    seed::Int = 42
    hidden_dim::Int = 64
    depth::Int = 3
    train_fraction::Float64 = 0.7
    val_fraction::Float64 = 0.15
    strategy_samples::Int = 150
end

function parse_args(args)
    cfg = EvalConfig()

    i = 1
    while i <= length(args)
        arg = args[i]
        if !startswith(arg, "--")
            error("Unexpected argument: $arg")
        end
        key = replace(arg, "--" => "")
        
        # Handle key-value options
        if i == length(args)
            error("Missing value for option: $arg")
        end
        value = args[i + 1]

        if key == "long-path"
            cfg.long_path = value
        elseif key == "index-path"
            cfg.index_path = value
        elseif key == "out-dir"
            cfg.out_dir = value
        elseif key == "epochs"
            cfg.epochs = parse(Int, value)
        elseif key == "batch-size"
            cfg.batch_size = parse(Int, value)
        elseif key == "learning-rate"
            cfg.learning_rate = parse(Float64, value)
        elseif key == "seed"
            cfg.seed = parse(Int, value)
        elseif key == "hidden-dim"
            cfg.hidden_dim = parse(Int, value)
        elseif key == "depth"
            cfg.depth = parse(Int, value)
        elseif key == "train-fraction"
            cfg.train_fraction = parse(Float64, value)
        elseif key == "val-fraction"
            cfg.val_fraction = parse(Float64, value)
        elseif key == "strategy-samples"
            cfg.strategy_samples = parse(Int, value)
        else
            error("Unknown option: --$key")
        end

        i += 2
    end

    if cfg.train_fraction <= 0 || cfg.train_fraction >= 1
        error("train-fraction must be in (0, 1)")
    end
    if cfg.val_fraction <= 0 || cfg.val_fraction >= 1
        error("val-fraction must be in (0, 1)")
    end
    if cfg.train_fraction + cfg.val_fraction >= 1
        error("train-fraction + val-fraction must be < 1")
    end

    return cfg
end

function trajectory_rmse_for_sequence(model, seq)
    pred = ZoneOptNeuralODEFramework.predict_sequence(model, seq)
    n = min(size(pred, 2), size(seq.u, 2))
    n == 0 && return NaN
    return sqrt(mean((pred[1:3, 1:n] .- seq.u[1:3, 1:n]).^2))
end

function rmse_by_entry_type(model, dataset)
    grouped = Dict{String,Vector{Float64}}()
    for seq in dataset
        key = seq.entry_type
        if !haskey(grouped, key)
            grouped[key] = Float64[]
        end
        push!(grouped[key], trajectory_rmse_for_sequence(model, seq))
    end

    rows = DataFrame(entry_type = String[], n = Int[], rmse = Float64[])
    for key in sort(collect(keys(grouped)))
        vals = grouped[key]
        push!(rows, (entry_type = key, n = length(vals), rmse = mean(vals)))
    end
    return rows
end

function baseline_decision_validity(dataset::Vector)
    """Compute baseline decision validity: always recommend the most common entry type."""
    if isempty(dataset)
        return NaN
    end
    type_counts = Dict{String, Int}()
    for seq in dataset
        type_counts[seq.entry_type] = get(type_counts, seq.entry_type, 0) + 1
    end
    most_common_type = argmax(type_counts)
    
    eligible = filter(seq -> seq.shot_within_horizon, dataset)
    if isempty(eligible)
        return NaN
    end
    matches = count(seq -> seq.entry_type == most_common_type for seq in eligible)
    return matches / length(eligible)
end

function strategy_recommendation_stats(model, dataset; max_samples::Int = 150, seed::Int = 42)
    if isempty(dataset)
        return DataFrame(
            seq_id = Int[],
            actual_entry_type = String[],
            recommended_entry_type = String[],
            score_carry = Float64[],
            score_dump = Float64[],
            score_pass = Float64[],
            matched_actual = Bool[]
        )
    end

    rng = MersenneTwister(seed)
    n = min(max_samples, length(dataset))
    sampled = shuffle(rng, dataset)[1:n]

    df = DataFrame(
        seq_id = Int[],
        actual_entry_type = String[],
        recommended_entry_type = String[],
        score_carry = Float64[],
        score_dump = Float64[],
        score_pass = Float64[],
        matched_actual = Bool[]
    )

    for seq in sampled
        rec = recommend_strategy(model, seq)
        label = rec.best_label
        scores = rec.scores
        push!(df, (
            seq_id = seq.seq_id,
            actual_entry_type = seq.entry_type,
            recommended_entry_type = label,
            score_carry = get(scores, :carry, NaN),
            score_dump = get(scores, :dump, NaN),
            score_pass = get(scores, :pass, NaN),
            matched_actual = (label == seq.entry_type)
        ))
    end

    return df
end

function history_table(history)
    n = length(history.train_loss)
    df = DataFrame(
        epoch = 1:n,
        train_loss = history.train_loss,
        train_rmse = history.train_rmse,
    )

    if length(history.train_strategy_loss) == n
        df.train_strategy_loss = history.train_strategy_loss
    else
        df.train_strategy_loss = fill(NaN, n)
    end

    if length(history.val_loss) == n
        df.val_loss = history.val_loss
    else
        df.val_loss = fill(NaN, n)
    end

    if length(history.val_rmse) == n
        df.val_rmse = history.val_rmse
    else
        df.val_rmse = fill(NaN, n)
    end

    if length(history.val_strategy_loss) == n
        df.val_strategy_loss = history.val_strategy_loss
    else
        df.val_strategy_loss = fill(NaN, n)
    end

    return df
end

function run_evaluation(cfg::EvalConfig)
    mkpath(cfg.out_dir)

    long_df, index_df = load_sequence_tables(cfg.long_path, cfg.index_path)
    examples = build_sequence_examples(long_df, index_df)
    isempty(examples) && error("No training examples built from processed CSV files.")

    train_set, val_set, test_set = split_sequences(
        examples;
        train_fraction = cfg.train_fraction,
        val_fraction = cfg.val_fraction,
        seed = cfg.seed
    )

    model = build_model(length(first(examples).context); hidden_dim = cfg.hidden_dim, depth = cfg.depth)

    train_cfg = TrainingConfig(
        epochs = cfg.epochs,
        batch_size = cfg.batch_size,
        learning_rate = cfg.learning_rate,
        seed = cfg.seed,
    )

    history = train!(model, train_set; val_set = val_set, config = train_cfg)


    train_loss = ZoneOptNeuralODEFramework.average_loss(model, train_set)
    val_loss = ZoneOptNeuralODEFramework.average_loss(model, val_set)
    test_loss = ZoneOptNeuralODEFramework.average_loss(model, test_set)

    train_rmse = ZoneOptNeuralODEFramework.trajectory_rmse(model, train_set)
    val_rmse = ZoneOptNeuralODEFramework.trajectory_rmse(model, val_set)
    test_rmse = ZoneOptNeuralODEFramework.trajectory_rmse(model, test_set)

    decision_validity = evaluate_decision_validity(model, test_set)
    baseline_validity = baseline_decision_validity(test_set)

    hist_df = history_table(history)
    CSV.write(joinpath(cfg.out_dir, "training_history.csv"), hist_df)

    entry_rmse_df = rmse_by_entry_type(model, test_set)
    CSV.write(joinpath(cfg.out_dir, "test_rmse_by_entry_type.csv"), entry_rmse_df)

    rec_df = strategy_recommendation_stats(model, test_set; max_samples = cfg.strategy_samples, seed = cfg.seed)
    CSV.write(joinpath(cfg.out_dir, "strategy_recommendations_sample.csv"), rec_df)

    summary = DataFrame(
        metric = String[],
        value = Float64[]
    )

    push!(summary, (metric = "n_total_sequences", value = float(length(examples))))
    push!(summary, (metric = "n_train", value = float(length(train_set))))
    push!(summary, (metric = "n_val", value = float(length(val_set))))
    push!(summary, (metric = "n_test", value = float(length(test_set))))

    push!(summary, (metric = "train_loss", value = train_loss))
    push!(summary, (metric = "val_loss", value = val_loss))
    push!(summary, (metric = "test_loss", value = test_loss))

    push!(summary, (metric = "train_rmse", value = train_rmse))
    push!(summary, (metric = "val_rmse", value = val_rmse))
    push!(summary, (metric = "test_rmse", value = test_rmse))

    push!(summary, (metric = "decision_validity", value = decision_validity))
    push!(summary, (metric = "baseline_decision_validity", value = baseline_validity))
    push!(summary, (metric = "validity_improvement_over_baseline", value = decision_validity - baseline_validity))
    push!(summary, (metric = "generated_at_unix", value = float(datetime2unix(now()))))

    CSV.write(joinpath(cfg.out_dir, "evaluation_summary.csv"), summary)

    println("Evaluation complete.")
    println("Sequences: total=$(length(examples)), train=$(length(train_set)), val=$(length(val_set)), test=$(length(test_set))")
    println("Loss: train=$(round(train_loss, digits=5)), val=$(round(val_loss, digits=5)), test=$(round(test_loss, digits=5))")
    println("RMSE: train=$(round(train_rmse, digits=5)), val=$(round(val_rmse, digits=5)), test=$(round(test_rmse, digits=5))")
    println("Decision validity: $(round(decision_validity, digits=5))")
    println("Saved files in: $(cfg.out_dir)")
end

function print_usage()
    println("Usage: julia evaluate_neural_ode.jl [options]")
    println("Options:")
    println("  --long-path <path>          (default: processed/neural_ode_sequences_long.csv)")
    println("  --index-path <path>         (default: processed/neural_ode_sequence_index.csv)")
    println("  --out-dir <path>            (default: evaluation)")
    println("  --epochs <int>              (default: 20)")
    println("  --batch-size <int>          (default: 16)")
    println("  --learning-rate <float>     (default: 0.001)")
    println("  --seed <int>                (default: 42)")
    println("  --hidden-dim <int>          (default: 64)")
    println("  --depth <int>               (default: 3)")
    println("  --train-fraction <float>    (default: 0.7)")
    println("  --val-fraction <float>      (default: 0.15)")
    println("  --strategy-samples <int>    (default: 150)")
end

function main()
    if any(a -> a in ("-h", "--help"), ARGS)
        print_usage()
        return
    end
    cfg = parse_args(ARGS)
    run_evaluation(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
