using CSV
using DataFrames
using Random
using Statistics
using Plots

include("../src/neural_ode_framework.jl")
using .ZoneOptNeuralODEFramework

Base.@kwdef mutable struct VizConfig
    long_path::String = joinpath("processed", "neural_ode_sequences_long.csv")
    index_path::String = joinpath("processed", "neural_ode_sequence_index.csv")
    checkpoint_path::String = joinpath("models", "epoch_140.jls")
    out_dir::String = "visualizations"
    split::String = "test"
    seq_id::Union{Nothing, Int} = nothing
    max_sequences::Int = 6
    seed::Int = 42
    hidden_dim::Int = 64
    depth::Int = 3
    make_gif::Bool = false
    gif_fps::Int = 10
    max_frames::Int = 0
    policy_csv::Union{Nothing,String} = nothing
end

function parse_bool(value::AbstractString)
    v = lowercase(strip(value))
    if v in ("1", "true", "yes", "y", "on")
        return true
    elseif v in ("0", "false", "no", "n", "off")
        return false
    end
    error("Could not parse boolean value: $(value)")
end

function parse_args(args)
    cfg = VizConfig()

    i = 1
    while i <= length(args)
        arg = args[i]
        if !startswith(arg, "--")
            error("Unexpected argument: $(arg)")
        end

        key = replace(arg, "--" => "")
        if i == length(args)
            error("Missing value for option: $(arg)")
        end
        value = args[i + 1]

        if key == "long-path"
            cfg.long_path = value
        elseif key == "index-path"
            cfg.index_path = value
        elseif key == "checkpoint"
            cfg.checkpoint_path = value
        elseif key == "out-dir"
            cfg.out_dir = value
        elseif key == "split"
            cfg.split = lowercase(value)
        elseif key == "seq-id"
            cfg.seq_id = parse(Int, value)
        elseif key == "max-sequences"
            cfg.max_sequences = parse(Int, value)
        elseif key == "seed"
            cfg.seed = parse(Int, value)
        elseif key == "hidden-dim"
            cfg.hidden_dim = parse(Int, value)
        elseif key == "depth"
            cfg.depth = parse(Int, value)
        elseif key == "make-gif"
            cfg.make_gif = parse_bool(value)
        elseif key == "gif-fps"
            cfg.gif_fps = parse(Int, value)
        elseif key == "max-frames"
            cfg.max_frames = parse(Int, value)
        elseif key == "policy-csv"
            cfg.policy_csv = value
        else
            error("Unknown option: --$(key)")
        end

        i += 2
    end

    if !(cfg.split in ("train", "val", "test", "all"))
        error("--split must be one of: train, val, test, all")
    end
    cfg.max_sequences > 0 || error("--max-sequences must be positive")
    cfg.gif_fps > 0 || error("--gif-fps must be positive")
    cfg.max_frames >= 0 || error("--max-frames must be >= 0")

    return cfg
end

safe_float_or_nan(x) = ismissing(x) ? NaN : Float64(x)

function player_positions_from_context(seq::TrajectorySequence)
    puck_x = seq.u[1, 1]
    puck_y = seq.u[2, 1]

    entry_players = Vector{Tuple{Float64, Float64}}()
    defend_players = Vector{Tuple{Float64, Float64}}()

    for i in 1:5
        dx = seq.context[3 + 2 * (i - 1) + 1]
        dy = seq.context[3 + 2 * (i - 1) + 2]
        push!(entry_players, (puck_x + dx, puck_y + dy))
    end

    for i in 1:5
        base = 13 + 2 * (i - 1)
        dx = seq.context[base + 1]
        dy = seq.context[base + 2]
        push!(defend_players, (puck_x + dx, puck_y + dy))
    end

    return entry_players, defend_players
end

function sequence_rmse_xy(actual::AbstractMatrix, predicted::AbstractMatrix)
    n = min(size(actual, 2), size(predicted, 2))
    n == 0 && return NaN
    return sqrt(mean((predicted[1:2, 1:n] .- actual[1:2, 1:n]) .^ 2))
end

function draw_rink!()
    rink_x = [-100.0, 100.0, 100.0, -100.0, -100.0]
    rink_y = [-42.5, -42.5, 42.5, 42.5, -42.5]

    plot!(rink_x, rink_y; color = :black, linewidth = 2, label = "")
    vline!([0.0]; color = :gray40, linestyle = :dash, linewidth = 1.5, label = "")
    vline!([-25.0, 25.0]; color = :royalblue, linestyle = :dot, linewidth = 1.2, label = "")
end

function frame_player_positions(frame_row)
    entry_players = Vector{Tuple{Float64, Float64}}()
    defend_players = Vector{Tuple{Float64, Float64}}()

    for i in 1:5
        ex = safe_float_or_nan(frame_row[Symbol("entry_p$(i)_x")])
        ey = safe_float_or_nan(frame_row[Symbol("entry_p$(i)_y")])
        dx = safe_float_or_nan(frame_row[Symbol("defend_p$(i)_x")])
        dy = safe_float_or_nan(frame_row[Symbol("defend_p$(i)_y")])
        push!(entry_players, (ex, ey))
        push!(defend_players, (dx, dy))
    end

    return entry_players, defend_players
end

function finite_points(points::Vector{Tuple{Float64, Float64}})
    xs = Float64[]
    ys = Float64[]
    for (x, y) in points
        if isfinite(x) && isfinite(y)
            push!(xs, x)
            push!(ys, y)
        end
    end
    return xs, ys
end

function plot_sequence_comparison(seq::TrajectorySequence, pred::Matrix{Float64}, out_path::String)
    actual_xy = seq.u[1:2, :]
    pred_xy = pred[1:2, :]
    n = min(size(actual_xy, 2), size(pred_xy, 2))

    entry_players, defend_players = player_positions_from_context(seq)

    p = plot(
        size = (1200, 700),
        xlim = (-100, 100),
        ylim = (-42.5, 42.5),
        aspect_ratio = :equal,
        legend = :topright,
        xlabel = "Rink X (ft)",
        ylabel = "Rink Y (ft)",
        background_color = :white,
        grid = false,
    )

    draw_rink!()

    if n > 0
        plot!(actual_xy[1, 1:n], actual_xy[2, 1:n]; color = :dodgerblue3, linewidth = 3, label = "Actual puck")
        plot!(pred_xy[1, 1:n], pred_xy[2, 1:n]; color = :crimson, linewidth = 3, linestyle = :dash, label = "Predicted puck")

        scatter!([actual_xy[1, 1]], [actual_xy[2, 1]]; color = :dodgerblue3, markersize = 7, markerstrokewidth = 0, label = "Actual start")
        scatter!([pred_xy[1, 1]], [pred_xy[2, 1]]; color = :crimson, markersize = 7, markerstrokewidth = 0, label = "Pred start")

        scatter!([actual_xy[1, n]], [actual_xy[2, n]]; color = :dodgerblue3, markershape = :diamond, markersize = 7, markerstrokewidth = 0, label = "Actual end")
        scatter!([pred_xy[1, n]], [pred_xy[2, n]]; color = :crimson, markershape = :diamond, markersize = 7, markerstrokewidth = 0, label = "Pred end")
    end

    entry_x = first.(entry_players)
    entry_y = last.(entry_players)
    defend_x = first.(defend_players)
    defend_y = last.(defend_players)

    scatter!(entry_x, entry_y; color = :forestgreen, markersize = 6, markerstrokewidth = 0.5, markerstrokecolor = :black, label = "Entry skaters")
    scatter!(defend_x, defend_y; color = :darkorange2, markersize = 6, markerstrokewidth = 0.5, markerstrokecolor = :black, label = "Defending skaters")

    puck_x0 = seq.u[1, 1]
    puck_y0 = seq.u[2, 1]
    scatter!([puck_x0], [puck_y0]; color = :black, markershape = :star5, markersize = 8, label = "Entry puck")

    rmse_xy = sequence_rmse_xy(seq.u, pred)
    title!(
        "Seq $(seq.seq_id) | $(seq.entry_type) | shot=$(seq.shot_within_horizon) | RMSE_xy=$(round(rmse_xy; digits = 3))"
    )

    savefig(p, out_path)
end

function animate_sequence_comparison(seq::TrajectorySequence,
                                     pred::Matrix{Float64},
                                     seq_frames::DataFrame,
                                     out_path::String;
                                     fps::Int = 10,
                                     max_frames::Int = 0,
                                     policy_row = nothing)
    n_actual = size(seq.u, 2)
    n_pred = size(pred, 2)
    n_frames = nrow(seq_frames)
    n_frames == 0 && error("No frame rows available for seq_id=$(seq.seq_id)")

    if max_frames > 0
        n_frames = min(n_frames, max_frames)
    end

    anim = @animate for k in 1:n_frames
        k_actual = min(k, n_actual)
        k_pred = min(k, n_pred)

        p = plot(
            size = (1200, 700),
            xlim = (-100, 100),
            ylim = (-42.5, 42.5),
            aspect_ratio = :equal,
            legend = :topright,
            xlabel = "Rink X (ft)",
            ylabel = "Rink Y (ft)",
            background_color = :white,
            grid = false,
        )

        draw_rink!()

        plot!(seq.u[1, 1:k_actual], seq.u[2, 1:k_actual]; color = :dodgerblue3, linewidth = 3, label = "Actual puck")
        # Only show the actual puck trajectory (no predicted puck in GIF)
        plot!(seq.u[1, 1:k_actual], seq.u[2, 1:k_actual]; color = :dodgerblue3, linewidth = 3, label = "Actual puck")
        scatter!([seq.u[1, k_actual]], [seq.u[2, k_actual]]; color = :dodgerblue3, markersize = 7, markerstrokewidth = 0, label = "Actual puck (current)")

        frame_row = seq_frames[k, :]
        entry_players, defend_players = frame_player_positions(frame_row)
        ex, ey = finite_points(entry_players)
        dx, dy = finite_points(defend_players)

        scatter!(ex, ey; color = :forestgreen, markersize = 6, markerstrokewidth = 0.5, markerstrokecolor = :black, label = "Entry skaters")
        scatter!(dx, dy; color = :darkorange2, markersize = 6, markerstrokewidth = 0.5, markerstrokecolor = :black, label = "Defending skaters")

        # Overlay policy decision if available
            if policy_row !== nothing
            # policy CSV uses confidence in 0..1
            conf = try
                Float64(policy_row[:confidence])
            catch
                NaN
            end
            pct = isfinite(conf) ? Int(round(conf * 100)) : -1
            band = if !isfinite(conf)
                "unknown"
            elseif conf < 0.40
                "low"
            elseif conf <= 0.70
                "medium"
            else
                "high"
            end

            recommended = String(policy_row[:best_label])
            actual_label = seq.entry_type

            label_text = "Recommend: $(recommended) ($(uppercase(band)), $(pct)%)"
            actual_text = "Actual: $(actual_label)"

            # color map for confidence band
            band_color = band == "low" ? :red : band == "medium" ? :orange : band == "high" ? :green : :black
            annotate!(-95, 38, text(label_text, 14, band_color))
            annotate!(-95, 33, text(actual_text, 12, :gray30))
        else
            title!("Seq $(seq.seq_id) | frame $(k)/$(n_frames) | $(seq.entry_type) | shot=$(seq.shot_within_horizon)")
        end
    end

    gif(anim, out_path; fps = fps)
end

function choose_sequences(cfg::VizConfig, train_set, val_set, test_set, all_examples)
    pool = if cfg.split == "train"
        train_set
    elseif cfg.split == "val"
        val_set
    elseif cfg.split == "test"
        test_set
    else
        all_examples
    end

    if !isnothing(cfg.seq_id)
        chosen = filter(seq -> seq.seq_id == cfg.seq_id, pool)
        isempty(chosen) && error("Sequence $(cfg.seq_id) was not found in split=$(cfg.split)")
        return chosen
    end

    rng = MersenneTwister(cfg.seed)
    shuffled = shuffle(rng, pool)
    n = min(cfg.max_sequences, length(shuffled))
    n > 0 || error("No sequences available in split=$(cfg.split)")
    return shuffled[1:n]
end

function run_visualization(cfg::VizConfig)
    println("Loading processed data...")
    long_df, index_df = ZoneOptNeuralODEFramework.load_sequence_tables(cfg.long_path, cfg.index_path)
    examples = ZoneOptNeuralODEFramework.build_sequence_examples(long_df, index_df)
    isempty(examples) && error("No sequences found.")

    train_set, val_set, test_set = ZoneOptNeuralODEFramework.split_sequences(examples; seed = cfg.seed)
    long_by_seq = Dict{Int, DataFrame}()
    for sdf in groupby(long_df, :seq_id)
        seq_id = Int(first(sdf.seq_id))
        long_by_seq[seq_id] = sort(DataFrame(sdf), :t_rel)
    end

    println("Building model and restoring checkpoint: $(cfg.checkpoint_path)")
    model = ZoneOptNeuralODEFramework.build_model(length(first(examples).context), train_set; hidden_dim = cfg.hidden_dim, depth = cfg.depth)
    payload = ZoneOptNeuralODEFramework.restore_model_checkpoint!(model, cfg.checkpoint_path)

    selected = choose_sequences(cfg, train_set, val_set, test_set, examples)
    mkpath(cfg.out_dir)

    summary = DataFrame(
        seq_id = Int[],
        split = String[],
        entry_type = String[],
        shot_within_horizon = Bool[],
        rmse_xy = Float64[],
        plot_path = String[],
        gif_path = String[],
        checkpoint_epoch = Int[],
    )

    for seq in selected
        pred = ZoneOptNeuralODEFramework.predict_sequence(model, seq)
        rmse_xy = sequence_rmse_xy(seq.u, pred)

        out_path = joinpath(cfg.out_dir, "seq_$(lpad(string(seq.seq_id), 4, '0'))_comparison.png")
        plot_sequence_comparison(seq, pred, out_path)

        gif_path = ""
        if cfg.make_gif
            gif_path = joinpath(cfg.out_dir, "seq_$(lpad(string(seq.seq_id), 4, '0'))_comparison.gif")
            haskey(long_by_seq, seq.seq_id) || error("Missing long-form frames for seq_id=$(seq.seq_id)")

            # Find policy row for this sequence when policy CSV provided (only supports test split mapping)
            policy_row = nothing
            if cfg.policy_csv !== nothing
                try
                    policy_df = CSV.read(cfg.policy_csv, DataFrame)
                    if :seq_id in names(policy_df)
                        rows = filter(row -> row[:seq_id] == seq.seq_id, policy_df)
                        if nrow(rows) >= 1
                            policy_row = rows[1, :]
                        end
                    else
                        # fallback to seq_index mapping for backwards compatibility
                        pos_in_test = findfirst(s -> s.seq_id == seq.seq_id, test_set)
                        if pos_in_test !== nothing
                            rows = filter(row -> row[:seq_index] == pos_in_test, policy_df)
                            if nrow(rows) >= 1
                                policy_row = rows[1, :]
                            end
                        end
                    end
                catch e
                    @warn "Failed to load or map policy CSV: $e"
                end
            end

            animate_sequence_comparison(
                seq,
                pred,
                long_by_seq[seq.seq_id],
                gif_path;
                fps = cfg.gif_fps,
                max_frames = cfg.max_frames,
                policy_row = policy_row,
            )
            println("Saved: $(gif_path)")
        end

        seq_split = if any(s -> s.seq_id == seq.seq_id, train_set)
            "train"
        elseif any(s -> s.seq_id == seq.seq_id, val_set)
            "val"
        elseif any(s -> s.seq_id == seq.seq_id, test_set)
            "test"
        else
            "unknown"
        end

        push!(summary, (
            seq_id = seq.seq_id,
            split = seq_split,
            entry_type = seq.entry_type,
            shot_within_horizon = seq.shot_within_horizon,
            rmse_xy = rmse_xy,
            plot_path = out_path,
            gif_path = gif_path,
            checkpoint_epoch = Int(payload.epoch),
        ))

        println("Saved: $(out_path)")
    end

    summary_path = joinpath(cfg.out_dir, "summary.csv")
    CSV.write(summary_path, summary)
    println("Saved summary: $(summary_path)")
end

function main(args)
    cfg = parse_args(args)
    run_visualization(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
