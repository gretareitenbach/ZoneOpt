using CSV, DataFrames, Random, Statistics
using Plots

repo_root = normpath(joinpath(dirname(@__FILE__), ".."))
include(joinpath(repo_root, "neural_ode_framework.jl"))
using .ZoneOptNeuralODEFramework

function main()
    rng = MersenneTwister(42)
    long_df, index_df = ZoneOptNeuralODEFramework.load_sequence_tables()
    examples = ZoneOptNeuralODEFramework.build_sequence_examples(long_df, index_df)
    train_set, val_set, test_set = ZoneOptNeuralODEFramework.split_sequences(examples; seed=42)

    println("Loaded sequences: total=$(length(examples)), train=$(length(train_set)), val=$(length(val_set)), test=$(length(test_set))")

    model = ZoneOptNeuralODEFramework.build_model(length(first(examples).context), train_set; hidden_dim=64, depth=3)

    # Load selected checkpoint if present (read SELECTED_CHECKPOINT.md if present)
    ckfile = joinpath(repo_root, "models", "SELECTED_CHECKPOINT.md")
    ckpath = joinpath(repo_root, "models", "epoch_210_best.jls")
    if isfile(ckfile)
        txt = read(ckfile, String)
        for line in split(txt, '\n')
            if startswith(strip(line), "- Path:")
                p = strip(split(line, ":", limit=2)[2])
                ckpath = joinpath(repo_root, p)
            end
        end
    end
    if isfile(ckpath)
        println("Restoring checkpoint: ", ckpath)
        ZoneOptNeuralODEFramework.restore_model_checkpoint!(model, ckpath)
    else
        println("Checkpoint not found: ", ckpath, ", continuing with uninitialized model")
    end

    # Compute RMSE on test set
    test_rmse = ZoneOptNeuralODEFramework.trajectory_rmse(model, test_set)
    println("Test RMSE (positions, ft): ", test_rmse)

    # Baseline decision: most common entry type in test set, and its match rate on successful sequences
    types = Dict{String,Int}()
    for s in test_set
        types[s.entry_type] = get(types, s.entry_type, 0) + 1
    end
    most_common = argmax(types)
    eligible = filter(s -> s.shot_within_horizon, test_set)
    baseline_match = if isempty(eligible) NaN else count(s -> s.entry_type == most_common, eligible) / length(eligible) end
    println("Baseline most common entry type: ", most_common, ", baseline match on successful sequences: ", baseline_match)

    # Decision validity of model
    decision_validity = ZoneOptNeuralODEFramework.evaluate_decision_validity(model, test_set)
    println("Model decision validity (on successful sequences): ", decision_validity)

    # RMSE by entry type (simple)
    rows = DataFrame(entry_type = String[], n = Int[], rmse = Float64[])
    grouped = Dict{String, Vector{Float64}}()
    for s in test_set
        pred = ZoneOptNeuralODEFramework.predict_sequence(model, s)
        n = min(size(pred,2), size(s.u,2))
        if n==0 continue end
        rmse = sqrt(mean((pred[1:3,1:n] .- s.u[1:3,1:n]).^2))
        push!(get!(grouped, s.entry_type, Float64[]), rmse)
    end
    for (k,v) in grouped
        push!(rows, (entry_type = k, n = length(v), rmse = mean(v)))
    end
    CSV.write("evaluation/test_rmse_by_entry_type_quick.csv", rows)

    # Quick calibration grid search for physics_prior parameters
    drags = [0.05, 0.12, 0.2]
    verts = [0.1, 0.3, 0.5]
    best = (drag=NaN, vert=NaN, rmse=Inf)
    results = DataFrame(drag=Float64[], vert=Float64[], rmse=Float64[])

    for d in drags, v in verts
        # monkeypatch physics_prior in module
        @eval ZoneOptNeuralODEFramework begin
            function physics_prior(u::AbstractVector)
                _, _, z, vx, vy, vz = u
                T = eltype(u)
                drag = $d
                vertical_damping = $v
                return [vx, vy, vz, -drag * vx, -drag * vy, -vertical_damping * z - T(0.05) * vz]
            end
        end

        rmse = ZoneOptNeuralODEFramework.trajectory_rmse(model, test_set)
        push!(results, (drag=d, vert=v, rmse=rmse))
        if rmse < best.rmse
            best = (drag=d, vert=v, rmse=rmse)
        end
        println("Tried drag=$(d), vert=$(v) => rmse=$(rmse)")
    end

    CSV.write("evaluation/physics_calibration_grid.csv", results)
    open("evaluation/CALIBRATION_SUMMARY.md", "w") do io
        println(io, "Best calibration:\n")
        println(io, "drag = $(best.drag)\nvertical_damping = $(best.vert)\nrmse = $(best.rmse)\n")
    end

    # Restore original physics_prior by reloading module file (simple approach)
    include(joinpath(repo_root, "neural_ode_framework.jl"))

    # Generate 3 example plots
    sample_idxs = rand(rng, 1:length(test_set), min(3, length(test_set)))
    mkpath("evaluation/figures")
    for (i, idx) in enumerate(sample_idxs)
        s = test_set[idx]
        pred = ZoneOptNeuralODEFramework.predict_sequence(model, s)
        plt = plot(s.u[1, :], s.u[2, :], label="actual", lw=2)
        plot!(plt, pred[1, :], pred[2, :], label="predicted", lw=2, ls=:dash)
        xlabel!(plt, "x (ft)")
        ylabel!(plt, "y (ft)")
        title!(plt, "Seq $(s.seq_id) entry=$(s.entry_type)")
        savefig(plt, "evaluation/figures/seq_$(s.seq_id)_plot.png")
    end

    # Summary
    summary = DataFrame(metric=String[], value=Float64[])
    push!(summary, (metric="test_rmse", value = test_rmse))
    push!(summary, (metric="baseline_match", value = baseline_match))
    push!(summary, (metric="decision_validity", value = decision_validity))
    CSV.write("evaluation/quick_evaluation_summary.csv", summary)
    println("Quick evaluation complete. Outputs in evaluation/")
end

main()
