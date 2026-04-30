module ZoneOptNeuralODEFramework

using CSV
using DataFrames
using Random
using Statistics
using LinearAlgebra

using Flux
using OrdinaryDiffEq
using DiffEqFlux
using SciMLSensitivity
using ProgressMeter
using Dates
using Serialization

export TrajectorySequence,
       TrainingConfig,
       load_sequence_tables,
       build_sequence_examples,
       split_sequences,
       build_model,
       predict_sequence,
       trajectory_loss,
       train!,
       evaluate_model,
       recommend_strategy,
       evaluate_decision_validity,
       run_training_pipeline

const ENTRY_TYPES = ("Carried", "Dumped", "Played")

"""
Context vector structure (length 23):
  [1:3]     Entry type one-hot (Carried, Dumped, Played)
  [4:13]    Entry team 5 players: Δx, Δy for each (puck-relative)
  [14:23]   Defending team 5 players: Δx, Δy for each (puck-relative)
All positions are relative to the puck at the zone entry moment.
"""
Base.@kwdef struct TrajectorySequence
    seq_id::Int
    game_id::String
    entry_type::String
    entry_team::String
    defend_team::String
    shot_within_horizon::Bool
    t::Vector{Float64}
    u::Matrix{Float64}
    context::Vector{Float64}
end

Base.@kwdef mutable struct TrainingConfig
    epochs::Int = 50
    batch_size::Int = 16
    learning_rate::Float64 = 1e-3
    lr_decay_factor::Float64 = 0.5
    lr_patience::Int = 8
    min_learning_rate::Float64 = 1e-5
    early_stopping_patience::Int = 20
    position_weight::Float64 = 1.0
    velocity_weight::Float64 = 0.25
    strategy_weight::Float64 = 0.2
    strategy_positive_weight::Float64 = 1.0
    strategy_negative_weight::Float64 = 1.0
    use_strategy_objective::Bool = true
    seed::Int = 42
    checkpoint_dir::String = joinpath(pwd(), "models")
    checkpoint_every::Int = 1
    save_best_checkpoint::Bool = true
    save_last_checkpoint::Bool = true
    solver = Tsit5()
end

Base.@kwdef mutable struct TrainingHistory
    train_loss::Vector{Float64} = Float64[]
    val_loss::Vector{Float64} = Float64[]
    train_rmse::Vector{Float64} = Float64[]
    val_rmse::Vector{Float64} = Float64[]
    train_strategy_loss::Vector{Float64} = Float64[]
    val_strategy_loss::Vector{Float64} = Float64[]
end

struct ZoneDynamics{M}
    residual::M
    input_shift::Vector{Float64}
    input_scale::Vector{Float64}
end

Flux.@functor ZoneDynamics (residual,)

logistic(x) = inv(1 + exp(-x))
timestamp() = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")

function entry_type_symbol(entry_type::AbstractString)
    if entry_type == "Carried"
        return :carry
    elseif entry_type == "Dumped"
        return :dump
    elseif entry_type == "Played"
        return :pass
    end
    return :unknown
end

function strategy_label(strategy::Symbol)
    if strategy == :carry
        return "Carried"
    elseif strategy == :dump
        return "Dumped"
    elseif strategy == :pass
        return "Played"
    elseif strategy == :shoot
        return "Shot"
    end
    return "Unknown"
end

function load_sequence_tables(long_path::AbstractString = joinpath("processed", "neural_ode_sequences_long.csv"),
                             index_path::AbstractString = joinpath("processed", "neural_ode_sequence_index.csv"))
    long_df = CSV.read(long_path, DataFrame; missingstring=["", "missing", "n/a"])
    index_df = CSV.read(index_path, DataFrame; missingstring=["", "missing", "n/a"])
    return long_df, index_df
end

function interpolate_series(values)
    n = length(values)
    out = Vector{Float64}(undef, n)
    valid = findall(!ismissing, values)

    if isempty(valid)
        fill!(out, 0.0)
        return out
    end

    first_valid = valid[1]
    first_value = Float64(values[first_valid])
    for i in 1:first_valid
        out[i] = first_value
    end

    for idx in 1:length(valid) - 1
        left = valid[idx]
        right = valid[idx + 1]
        left_value = Float64(values[left])
        right_value = Float64(values[right])
        out[left] = left_value
        gap = right - left
        if gap > 1
            for step in 1:gap - 1
                α = step / gap
                out[left + step] = (1 - α) * left_value + α * right_value
            end
        end
    end

    last_valid = valid[end]
    last_value = Float64(values[last_valid])
    for i in last_valid:n
        out[i] = last_value
    end

    return out
end

function sequence_time_grid(seq_df)
    props = propertynames(seq_df)
    if (:frame_index in props) && (:fps_estimate in props)
        fps_values = Float64[]
        for value in seq_df.fps_estimate
            if !ismissing(value)
                push!(fps_values, Float64(value))
            end
        end
        if !isempty(fps_values)
            fps = median(fps_values)
            if isfinite(fps) && fps > 0
                frame_index = Float64.(seq_df.frame_index)
                return (frame_index .- first(frame_index)) ./ fps
            end
        end
    end

    t = Float64.(seq_df.t_rel)
    if length(t) <= 1
        return t
    end

    for i in 2:length(t)
        if !(t[i] > t[i - 1])
            t[i] = t[i - 1] + 1e-3
        end
    end
    return t
end

function finite_difference(values::Vector{Float64}, t::Vector{Float64})
    n = length(values)
    diff_values = zeros(Float64, n)
    if n == 1
        return diff_values
    end

    for i in 2:n
        dt = t[i] - t[i - 1]
        if dt <= 1e-3
            diff_values[i] = diff_values[i - 1]
        else
            diff_values[i] = (values[i] - values[i - 1]) / dt
        end
    end
    diff_values[1] = diff_values[2]
    return diff_values
end

function extract_state_matrix(seq_df::SubDataFrame)
    t = sequence_time_grid(seq_df)
    x = interpolate_series(seq_df.puck_x)
    y = interpolate_series(seq_df.puck_y)
    z = interpolate_series(seq_df.puck_z)
    vx = finite_difference(x, t)
    vy = finite_difference(y, t)
    vz = finite_difference(z, t)
    u = permutedims(hcat(x, y, z, vx, vy, vz))
    return t, u
end

function safe_float(x)
    return ismissing(x) ? 0.0 : Float64(x)
end

function build_context_vector(first_row)
    puck_x = safe_float(first_row.puck_x)
    puck_y = safe_float(first_row.puck_y)
    context = Float64[]

    entry_type = String(first_row.entry_type)
    append!(context, entry_type == "Carried" ? [1.0, 0.0, 0.0] : entry_type == "Dumped" ? [0.0, 1.0, 0.0] : [0.0, 0.0, 1.0])

    for i in 1:5
        push!(context, safe_float(first_row[Symbol("entry_p$(i)_x")]) - puck_x)
        push!(context, safe_float(first_row[Symbol("entry_p$(i)_y")]) - puck_y)
    end

    for i in 1:5
        push!(context, safe_float(first_row[Symbol("defend_p$(i)_x")]) - puck_x)
        push!(context, safe_float(first_row[Symbol("defend_p$(i)_y")]) - puck_y)
    end

    return context
end

function build_sequence_example(seq_df::SubDataFrame, index_row)
    ordered = sort(seq_df, :t_rel)
    ordered = DataFrame(ordered)
    t, u = extract_state_matrix(view(ordered, :, :))
    first_row = ordered[1, :]
    context = build_context_vector(first_row)

    return TrajectorySequence(
        seq_id = Int(index_row.seq_id),
        game_id = String(index_row.game_id),
        entry_type = String(index_row.entry_type),
        entry_team = String(index_row.entry_team),
        defend_team = String(index_row.defend_team),
        shot_within_horizon = Bool(index_row.shot_within_horizon),
        t = t,
        u = u,
        context = context,
    )
end

function build_sequence_examples(long_df::DataFrame, index_df::DataFrame)
    index_lookup = Dict(Int(index_df.seq_id[i]) => index_df[i, :] for i in 1:nrow(index_df))
    examples = TrajectorySequence[]

    for sdf in groupby(long_df, :seq_id)
        seq_id = Int(first(sdf.seq_id))
        haskey(index_lookup, seq_id) || continue
        push!(examples, build_sequence_example(sdf, index_lookup[seq_id]))
    end

    return examples
end

function split_sequences(examples::Vector{TrajectorySequence}; train_fraction=0.7, val_fraction=0.15, seed::Int=42)
    rng = MersenneTwister(seed)
    shuffled = shuffle(rng, examples)
    n = length(shuffled)
    n == 0 && return TrajectorySequence[], TrajectorySequence[], TrajectorySequence[]
    n == 1 && return shuffled, TrajectorySequence[], TrajectorySequence[]

    n_train = max(1, round(Int, train_fraction * n))
    n_val = max(1, round(Int, val_fraction * n))
    n_train = min(n_train, max(1, n - 2))
    n_val = min(n_val, max(0, n - n_train - 1))
    train_set = shuffled[1:n_train]
    val_set = shuffled[n_train + 1:n_train + n_val]
    test_set = shuffled[n_train + n_val + 1:end]
    return train_set, val_set, test_set
end

function physics_prior(u::AbstractVector)
    _, _, z, vx, vy, vz = u
    T = eltype(u)
    drag = T(0.12)
    vertical_damping = T(0.30)
    return [vx, vy, vz, -drag * vx, -drag * vy, -vertical_damping * z - T(0.05) * vz]
end

function build_model(context_dim::Int; hidden_dim::Int = 64, depth::Int = 3)
    depth = max(2, depth)
    input_dim = 6 + context_dim + 1
    layers = Flux.Chain(
        Flux.Dense(input_dim, hidden_dim, tanh),
        [Flux.Dense(hidden_dim, hidden_dim, tanh) for _ in 1:max(0, depth - 2)]...,
        Flux.Dense(hidden_dim, 3)
    )
    input_shift = zeros(Float64, input_dim)
    input_scale = ones(Float64, input_dim)
    return ZoneDynamics(layers, input_shift, input_scale)
end

function build_model(context_dim::Int,
                     train_set::Vector{TrajectorySequence};
                     hidden_dim::Int = 64,
                     depth::Int = 3)
    model = build_model(context_dim; hidden_dim = hidden_dim, depth = depth)
    apply_input_normalization!(model, train_set)
    return model
end

function (model::ZoneDynamics)(u, context, t)
    raw_input = vcat(u, context, [t])
    normalized_input = (raw_input .- model.input_shift) ./ model.input_scale
    residual = model.residual(normalized_input)
    prior = physics_prior(u)
    return vcat(prior[1:3], prior[4:6] .+ residual)
end

function apply_input_normalization!(model::ZoneDynamics, train_set::Vector{TrajectorySequence})
    isempty(train_set) && return model

    input_dim = length(model.input_shift)
    sum_vec = zeros(Float64, input_dim)
    sum_sq_vec = zeros(Float64, input_dim)
    n = 0

    for seq in train_set
        context = seq.context
        for j in eachindex(seq.t)
            sample = vcat(seq.u[:, j], context, [seq.t[j]])
            sum_vec .+= sample
            sum_sq_vec .+= sample .^ 2
            n += 1
        end
    end

    n == 0 && return model

    mean_vec = sum_vec ./ n
    var_vec = max.(sum_sq_vec ./ n .- mean_vec .^ 2, 1e-8)
    std_vec = sqrt.(var_vec)

    model.input_shift .= mean_vec
    model.input_scale .= std_vec
    return model
end

function predict_sequence(model::ZoneDynamics, seq::TrajectorySequence; solver = Tsit5(), abstol = 1e-6, reltol = 1e-6)
    u0 = seq.u[:, 1]
    tspan = (seq.t[1], seq.t[end])
    flat_params, reconstruct = Flux.destructure(model)
    rhs = (du, u, p, t) -> begin
        du .= reconstruct(p)(u, seq.context, t)
    end
    prob = ODEProblem(rhs, u0, tspan, flat_params)
    sol = solve(prob, solver; saveat = seq.t, abstol = abstol, reltol = reltol, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
    return Array(sol)
end

function trajectory_loss(model::ZoneDynamics, seq::TrajectorySequence; position_weight::Float64 = 1.0, velocity_weight::Float64 = 0.25)
    pred = predict_sequence(model, seq)
    n = min(size(pred, 2), size(seq.u, 2))
    n == 0 && return Inf
    pos_loss = mean(abs2, pred[1:3, 1:n] .- seq.u[1:3, 1:n])
    vel_loss = mean(abs2, pred[4:6, 1:n] .- seq.u[4:6, 1:n])
    return position_weight * pos_loss + velocity_weight * vel_loss
end

function strategy_quality_loss(model::ZoneDynamics,
                               seq::TrajectorySequence;
                               positive_weight::Float64 = 1.0,
                               negative_weight::Float64 = 1.0)
    pred = predict_sequence(model, seq)
    quality = clamp(strategy_reward(pred, seq), 1e-6, 1 - 1e-6)

    if seq.shot_within_horizon
        return -positive_weight * log(quality)
    end
    return -negative_weight * log(1 - quality)
end

function strategy_class_weights(dataset::Vector{TrajectorySequence})
    n_total = length(dataset)
    n_total == 0 && return 1.0, 1.0

    n_pos = count(seq -> seq.shot_within_horizon, dataset)
    n_neg = n_total - n_pos

    if n_pos == 0 || n_neg == 0
        return 1.0, 1.0
    end

    pos_weight = n_total / (2 * n_pos)
    neg_weight = n_total / (2 * n_neg)
    return pos_weight, neg_weight
end

function sequence_total_loss(model::ZoneDynamics,
                             seq::TrajectorySequence,
                             config::TrainingConfig,
                             positive_weight::Float64,
                             negative_weight::Float64)
    traj = trajectory_loss(model, seq;
        position_weight = config.position_weight,
        velocity_weight = config.velocity_weight)

    if !config.use_strategy_objective || config.strategy_weight <= 0
        return traj, traj, 0.0
    end

    strat = strategy_quality_loss(model, seq;
        positive_weight = positive_weight,
        negative_weight = negative_weight)
    total = traj + config.strategy_weight * strat
    return total, traj, strat
end

function average_total_loss(model::ZoneDynamics,
                            dataset::Vector{TrajectorySequence},
                            config::TrainingConfig,
                            positive_weight::Float64,
                            negative_weight::Float64)
    isempty(dataset) && return NaN, NaN, NaN

    total_sum = 0.0
    traj_sum = 0.0
    strat_sum = 0.0
    for seq in dataset
        total, traj, strat = sequence_total_loss(model, seq, config, positive_weight, negative_weight)
        total_sum += total
        traj_sum += traj
        strat_sum += strat
    end

    n = length(dataset)
    return total_sum / n, traj_sum / n, strat_sum / n
end

function batch_indices(n::Int, batch_size::Int)
    batches = Vector{UnitRange{Int}}()
    start_idx = 1
    while start_idx <= n
        stop_idx = min(n, start_idx + batch_size - 1)
        push!(batches, start_idx:stop_idx)
        start_idx = stop_idx + 1
    end
    return batches
end

function average_loss(model::ZoneDynamics, dataset::Vector{TrajectorySequence}; position_weight::Float64 = 1.0, velocity_weight::Float64 = 0.25)
    isempty(dataset) && return NaN
    total = 0.0
    for seq in dataset
        total += trajectory_loss(model, seq; position_weight = position_weight, velocity_weight = velocity_weight)
    end
    return total / length(dataset)
end

function trajectory_rmse(model::ZoneDynamics, dataset::Vector{TrajectorySequence})
    isempty(dataset) && return NaN
    errors = Float64[]
    for seq in dataset
        pred = predict_sequence(model, seq)
        n = min(size(pred, 2), size(seq.u, 2))
        n == 0 && continue
        push!(errors, sqrt(mean((pred[1:3, 1:n] .- seq.u[1:3, 1:n]).^2)))
    end
    isempty(errors) && return NaN
    return mean(errors)
end

function load_model_checkpoint(checkpoint_path::AbstractString)
    isfile(checkpoint_path) || error("Checkpoint not found: $(checkpoint_path)")
    return Serialization.deserialize(checkpoint_path)
end

function restore_model_checkpoint!(model::ZoneDynamics, checkpoint_path::AbstractString)
    payload = load_model_checkpoint(checkpoint_path)
    hasproperty(payload, :model_state) || error("Checkpoint $(checkpoint_path) does not contain model_state")
    Flux.loadmodel!(model, payload.model_state)
    return payload
end

function save_model_checkpoint(model::ZoneDynamics,
                               history::TrainingHistory,
                               epoch::Int,
                               checkpoint_dir::AbstractString;
                               tag::AbstractString = "",
                               learning_rate::Float64 = NaN,
                               best_val_loss::Float64 = NaN,
                               best_epoch::Int = 0,
                               epochs_without_improvement::Int = 0)
    mkpath(checkpoint_dir)
    padded_epoch = lpad(string(epoch), 3, '0')
    filename = isempty(tag) ? "epoch_$(padded_epoch).jls" : "epoch_$(padded_epoch)_$(tag).jls"
    path = joinpath(checkpoint_dir, filename)

    payload = (
        epoch = epoch,
        saved_at = timestamp(),
        model_state = Flux.state(model),
        learning_rate = learning_rate,
        best_val_loss = best_val_loss,
        best_epoch = best_epoch,
        epochs_without_improvement = epochs_without_improvement,
        train_loss = history.train_loss[end],
        train_rmse = history.train_rmse[end],
        val_loss = isempty(history.val_loss) ? NaN : history.val_loss[end],
        val_rmse = isempty(history.val_rmse) ? NaN : history.val_rmse[end],
    )

    Serialization.serialize(path, payload)
    return path
end

function train!(model::ZoneDynamics, train_set::Vector{TrajectorySequence};
                val_set::Vector{TrajectorySequence} = TrajectorySequence[],
                config::TrainingConfig = TrainingConfig(),
                start_epoch::Int = 1,
                initial_best_val_loss::Float64 = Inf,
                initial_best_epoch::Int = 0,
                initial_epochs_without_improvement::Int = 0,
                initial_learning_rate::Union{Nothing, Float64} = nothing)
    Random.seed!(config.seed)
    history = TrainingHistory()
    current_lr = something(initial_learning_rate, config.learning_rate)
    opt = Flux.setup(Flux.Adam(current_lr), model)

    start_epoch > config.epochs && error("start_epoch ($(start_epoch)) exceeds target epochs ($(config.epochs))")
    progress = Progress(max(0, config.epochs - start_epoch + 1); desc = "Training epochs", showspeed = true)
    best_val_loss = initial_best_val_loss
    best_epoch = initial_best_epoch
    epochs_without_improvement = initial_epochs_without_improvement
    last_epoch = start_epoch - 1

    auto_pos_weight, auto_neg_weight = strategy_class_weights(train_set)
    strategy_pos_weight = auto_pos_weight * config.strategy_positive_weight
    strategy_neg_weight = auto_neg_weight * config.strategy_negative_weight

    for epoch in start_epoch:config.epochs
        last_epoch = epoch
        shuffled = shuffle(MersenneTwister(config.seed + epoch), train_set)
        epoch_total_loss = 0.0
        epoch_traj_loss = 0.0
        epoch_strategy_loss = 0.0

        for batch_range in batch_indices(length(shuffled), config.batch_size)
            batch = shuffled[batch_range]
            loss_fn(m) = begin
                batch_loss = 0.0
                for seq in batch
                    total_loss, _, _ = sequence_total_loss(m, seq, config, strategy_pos_weight, strategy_neg_weight)
                    batch_loss += total_loss
                end
                batch_loss / length(batch)
            end
            gs = gradient(loss_fn, model)[1]
            Flux.update!(opt, model, gs)

            batch_total, batch_traj, batch_strategy = average_total_loss(model, collect(batch), config, strategy_pos_weight, strategy_neg_weight)
            epoch_total_loss += batch_total
            epoch_traj_loss += batch_traj
            epoch_strategy_loss += batch_strategy
        end

        n_batches = max(1, length(batch_indices(length(shuffled), config.batch_size)))
        train_loss = epoch_total_loss / n_batches
        train_traj_loss = epoch_traj_loss / n_batches
        train_strategy_loss = epoch_strategy_loss / n_batches

        push!(history.train_loss, train_loss)
        push!(history.train_strategy_loss, train_strategy_loss)
        push!(history.train_rmse, trajectory_rmse(model, train_set))

        if !isempty(val_set)
            val_total_loss, _, val_strategy_loss = average_total_loss(model, val_set, config, strategy_pos_weight, strategy_neg_weight)
            push!(history.val_loss, val_total_loss)
            push!(history.val_strategy_loss, val_strategy_loss)
            push!(history.val_rmse, trajectory_rmse(model, val_set))
        end

        # Emit one stable line per epoch for cluster logs / tail -f monitoring.
        if !isempty(val_set)
            println("[$(timestamp())] Epoch $(epoch)/$(config.epochs) | train_loss=$(round(train_loss; digits=6)) | train_traj=$(round(train_traj_loss; digits=6)) | train_strategy=$(round(train_strategy_loss; digits=6)) | train_rmse=$(round(history.train_rmse[end]; digits=6)) | val_loss=$(round(history.val_loss[end]; digits=6)) | val_strategy=$(round(history.val_strategy_loss[end]; digits=6)) | val_rmse=$(round(history.val_rmse[end]; digits=6))")
        else
            println("[$(timestamp())] Epoch $(epoch)/$(config.epochs) | train_loss=$(round(train_loss; digits=6)) | train_traj=$(round(train_traj_loss; digits=6)) | train_strategy=$(round(train_strategy_loss; digits=6)) | train_rmse=$(round(history.train_rmse[end]; digits=6))")
        end
        flush(stdout)

        if !isempty(val_set)
            ProgressMeter.next!(progress; showvalues = [
                (:epoch, epoch),
                (:train_loss, round(train_loss; digits = 5)),
                (:val_loss, round(history.val_loss[end]; digits = 5)),
            ])
        else
            ProgressMeter.next!(progress; showvalues = [
                (:epoch, epoch),
                (:train_loss, round(train_loss; digits = 5)),
            ])
        end

        if config.save_best_checkpoint && !isempty(val_set)
            current_val_loss = history.val_loss[end]
            if current_val_loss < best_val_loss
                best_val_loss = current_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_path = save_model_checkpoint(model, history, epoch, config.checkpoint_dir;
                    tag = "best",
                    learning_rate = current_lr,
                    best_val_loss = best_val_loss,
                    best_epoch = best_epoch,
                    epochs_without_improvement = epochs_without_improvement)
                println("[$(timestamp())] Saved best checkpoint: $(best_path)")
                flush(stdout)
            else
                epochs_without_improvement += 1

                if config.lr_patience > 0 && (epochs_without_improvement % config.lr_patience == 0) && current_lr > config.min_learning_rate
                    current_lr = max(config.min_learning_rate, current_lr * config.lr_decay_factor)
                    opt = Flux.setup(Flux.Adam(current_lr), model)
                    println("[$(timestamp())] Learning rate reduced to $(round(current_lr; digits=8)) after $(epochs_without_improvement) epochs without validation improvement")
                    flush(stdout)
                end

                if config.early_stopping_patience > 0 && epochs_without_improvement >= config.early_stopping_patience
                    println("[$(timestamp())] Early stopping at epoch $(epoch) (best epoch=$(best_epoch), best val_loss=$(round(best_val_loss; digits=6)))")
                    flush(stdout)
                    break
                end
            end
        end

        if config.checkpoint_every > 0 && (epoch % config.checkpoint_every == 0)
            checkpoint_path = save_model_checkpoint(model, history, epoch, config.checkpoint_dir;
                learning_rate = current_lr,
                best_val_loss = best_val_loss,
                best_epoch = best_epoch,
                epochs_without_improvement = epochs_without_improvement)
            println("[$(timestamp())] Saved checkpoint: $(checkpoint_path)")
            flush(stdout)
        end
    end

    if config.save_last_checkpoint
        last_path = save_model_checkpoint(model, history, last_epoch, config.checkpoint_dir;
            tag = "last",
            learning_rate = current_lr,
            best_val_loss = best_val_loss,
            best_epoch = best_epoch,
            epochs_without_improvement = epochs_without_improvement)
        println("[$(timestamp())] Saved final checkpoint: $(last_path)")
        flush(stdout)
    end

    return history
end

function strategy_initial_state(seq::TrajectorySequence, strategy::Symbol)
    u0 = copy(seq.u[:, 1])
    attack_sign = u0[1] >= 0 ? 1.0 : -1.0
    initial_speed = norm(u0[4:5])
    initial_speed = initial_speed == 0 ? 1.0 : initial_speed

    if strategy == :carry
        u0[4] += 0.15 * attack_sign * initial_speed
        u0[5] *= 0.85
        u0[6] *= 0.95
    elseif strategy == :dump
        u0[4] += 0.35 * attack_sign * initial_speed
        u0[5] *= 0.55
        u0[6] -= 0.20
    elseif strategy == :pass
        # Context stores (entry_type_onehot[3], entry_p1_dx, entry_p1_dy, ..., entry_p5_dx, entry_p5_dy, defend_p1_dx, ...)
        # Extract puck-relative teammate positions and convert to absolute
        puck_x = u0[1]
        puck_y = u0[2]
        teammate_rel_x = seq.context[4:2:12]
        teammate_rel_y = seq.context[5:2:13]
        teammate_abs_x = puck_x .+ teammate_rel_x
        teammate_abs_y = puck_y .+ teammate_rel_y
        
        if attack_sign > 0
            target_idx = argmax(teammate_abs_x)
        else
            target_idx = argmin(teammate_abs_x)
        end
        target = [teammate_abs_x[target_idx], teammate_abs_y[target_idx]]
        direction = target .- u0[1:2]
        norm_direction = norm(direction) == 0 ? [attack_sign, 0.0] : direction / norm(direction)
        u0[4] = 1.10 * norm_direction[1] * initial_speed
        u0[5] = 1.10 * norm_direction[2] * initial_speed
    end

    if strategy == :shoot
        # Simple shoot heuristic: stronger forward velocity, less lateral, slight lift
        u0[4] += 1.50 * attack_sign * initial_speed
        u0[5] *= 0.60
        u0[6] += 0.30
    end

    return u0
end

function simulate_strategy(model::ZoneDynamics, seq::TrajectorySequence, strategy::Symbol; solver = Tsit5())
    u0 = strategy_initial_state(seq, strategy)
    tspan = (seq.t[1], seq.t[end])
    flat_params, reconstruct = Flux.destructure(model)
    rhs = (du, u, p, t) -> begin
        du .= reconstruct(p)(u, seq.context, t)
    end
    prob = ODEProblem(rhs, u0, tspan, flat_params)
    sol = solve(prob, solver; saveat = seq.t, abstol = 1e-6, reltol = 1e-6, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
    return Array(sol)
end

function strategy_reward(pred::AbstractMatrix, seq::TrajectorySequence)
    """Score strategy on: offensive zone depth, proximity to net, and puck control.
    
    Thresholds calibrated from Stathletes tracking data:
    - X coordinate: -100 to 100 feet (full rink representation, centerline at 0)
    - Y coordinate: ~±40 feet (rink width)
    - Z coordinate: ~0 feet (puck stays on ice, minimal height variation)
    
    Reward breakdown:
    - zone_score: peaks at x≈50 (deep in offensive zone)
    - danger_score: peaks at y≈0 (near net), danger area ~±10 feet
    - control_score: peaks at ~1 ft/s (controlled entry speed)
    """
    final_x = pred[1, end]
    final_y = abs(pred[2, end])
    final_speed = norm(pred[4:5, end])
    attack_sign = seq.u[1, 1] >= 0 ? 1.0 : -1.0

    # Zone score: reward puck deep in offensive zone (x≈50 optimal)
    zone_score = logistic((attack_sign * final_x - 50.0) / 8.0)
    # Danger score: reward puck close to net (y≈0 optimal, danger area ±10 feet)
    danger_score = logistic((10.0 - final_y) / 4.0)
    # Control score: reward moderate speed (not too slow, not out of control)
    control_score = logistic((final_speed - 1.0) / 2.0)

    return zone_score * danger_score * control_score
end

function recommend_strategy(model::ZoneDynamics, seq::TrajectorySequence; strategies::Vector{Symbol} = [:carry, :dump, :pass, :shoot])
    scores = Dict{Symbol,Float64}()
    trajectories = Dict{Symbol,Matrix{Float64}}()

    for strategy in strategies
        pred = simulate_strategy(model, seq, strategy)
        trajectories[strategy] = pred
        scores[strategy] = strategy_reward(pred, seq)
    end

    best_strategy = first(sort(collect(keys(scores)); by = s -> scores[s], rev = true))
    return (
        best_strategy = best_strategy,
        best_label = strategy_label(best_strategy),
        scores = scores,
        trajectories = trajectories,
    )
end

function recommend_action(model::ZoneDynamics, seq::TrajectorySequence; strategies::Vector{Symbol} = [:carry, :dump, :pass, :shoot])
    # Use recommend_strategy to get raw scores and trajectories
    rec = recommend_strategy(model, seq; strategies = strategies)
    scores = rec.scores

    # Convert scores (0..1) to logits via logit(s) = log(s/(1-s)) for better separation
    eps = 1e-9
    logits = Dict{Symbol,Float64}()
    for (k, v) in scores
        s = clamp(v, eps, 1 - eps)
        logits[k] = log(s / (1 - s))
    end

    # Softmax over logits
    exps = Dict{Symbol,Float64}()
    sumexp = 0.0
    for (k, v) in logits
        ev = exp(v)
        exps[k] = ev
        sumexp += ev
    end
    probs = Dict{Symbol,Float64}()
    for (k, v) in exps
        probs[k] = v / sumexp
    end

    best = rec.best_strategy
    confidence = probs[best]

    return (
        best_strategy = best,
        best_label = strategy_label(best),
        confidence = confidence,
        probabilities = probs,
        raw_scores = scores,
        trajectories = rec.trajectories,
    )
end

function evaluate_decision_validity(model::ZoneDynamics, dataset::Vector{TrajectorySequence})
    """Evaluate decision validity: how often model recommends strategy matching successful real-world outcome."""
    # Only consider sequences where a shot actually occurred (ground truth success)
    eligible = filter(seq -> seq.shot_within_horizon, dataset)
    isempty(eligible) && return NaN

    correct = 0
    for seq in eligible
        recommendation = recommend_strategy(model, seq)
        # Correct if recommendation matches the actual entry type used in this successful sequence
        if strategy_label(recommendation.best_strategy) == seq.entry_type
            correct += 1
        end
    end

    return correct / length(eligible)
end

function run_training_pipeline(; long_path::AbstractString = joinpath("processed", "neural_ode_sequences_long.csv"),
                               index_path::AbstractString = joinpath("processed", "neural_ode_sequence_index.csv"),
                               hidden_dim::Int = 64,
                               depth::Int = 3,
                               config::TrainingConfig = TrainingConfig(),
                               resume_checkpoint::Union{Nothing, AbstractString} = nothing)
    long_df, index_df = load_sequence_tables(long_path, index_path)
    examples = build_sequence_examples(long_df, index_df)
    isempty(examples) && error("No sequences were loaded from the preprocessing outputs.")

    train_set, val_set, test_set = split_sequences(examples; seed = config.seed)
    model = build_model(length(first(examples).context), train_set; hidden_dim = hidden_dim, depth = depth)
    start_epoch = 1
    initial_best_val_loss = Inf
    initial_best_epoch = 0
    initial_epochs_without_improvement = 0
    initial_learning_rate = nothing

    if !isnothing(resume_checkpoint)
        payload = restore_model_checkpoint!(model, resume_checkpoint)
        start_epoch = Int(payload.epoch) + 1
        initial_best_val_loss = if hasproperty(payload, :best_val_loss) && isfinite(Float64(payload.best_val_loss))
            Float64(payload.best_val_loss)
        elseif hasproperty(payload, :val_loss) && isfinite(Float64(payload.val_loss))
            Float64(payload.val_loss)
        else
            Inf
        end
        initial_best_epoch = hasproperty(payload, :best_epoch) ? Int(payload.best_epoch) : Int(payload.epoch)
        initial_epochs_without_improvement = hasproperty(payload, :epochs_without_improvement) ? Int(payload.epochs_without_improvement) : 0
        initial_learning_rate = if hasproperty(payload, :learning_rate) && isfinite(Float64(payload.learning_rate))
            Float64(payload.learning_rate)
        else
            nothing
        end
        println("[$(timestamp())] Resuming from checkpoint $(resume_checkpoint) at epoch $(payload.epoch)")
        flush(stdout)
    end

    history = train!(model, train_set;
        val_set = val_set,
        config = config,
        start_epoch = start_epoch,
        initial_best_val_loss = initial_best_val_loss,
        initial_best_epoch = initial_best_epoch,
        initial_epochs_without_improvement = initial_epochs_without_improvement,
        initial_learning_rate = initial_learning_rate)
    test_rmse = trajectory_rmse(model, test_set)
    test_loss = average_loss(model, test_set;
        position_weight = config.position_weight,
        velocity_weight = config.velocity_weight)
    decision_validity = evaluate_decision_validity(model, test_set)

    return (
        model = model,
        history = history,
        test_rmse = test_rmse,
        test_loss = test_loss,
        decision_validity = decision_validity,
        splits = (train = length(train_set), val = length(val_set), test = length(test_set)),
    )
end

end # module