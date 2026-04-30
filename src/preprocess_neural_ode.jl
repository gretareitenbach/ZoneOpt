using CSV
using DataFrames
using Dates
using Statistics

const REGULATION_PERIOD_SECONDS = 20 * 60
const OT_PERIOD_SECONDS = 5 * 60

"""
IMPORTANT CONSTRAINT FOR NEURAL ODE FRAMEWORK:
The preprocessing output must produce 6-dimensional state vectors [x,y,z,vx,vy,vz]
and context vectors matching the expected format:
  [1:3]   Entry type one-hot
  [4:13]  Entry team positions (5 players × 2 coords)
  [14:23] Defense team positions (5 players × 2 coords)
All player positions must be puck-relative (not absolute) for proper physics interpretation.
"""

"""Parse M:SS game-clock strings into remaining seconds."""
function parse_clock_seconds(clock_raw)
    if clock_raw isa Time
        return hour(clock_raw) * 60 + minute(clock_raw)
    end

    s = strip(string(clock_raw))
    parts = split(s, ":")
    if length(parts) == 2
        minutes = parse(Int, parts[1])
        seconds = parse(Int, parts[2])
        return minutes * 60 + seconds
    elseif length(parts) == 3
        hours = parse(Int, parts[1])
        minutes = parse(Int, parts[2])
        return hours * 60 + minutes
    end
    error("Invalid clock format: $s")
end

"""Normalize period values to Int (1,2,3,...) or :OT."""
function normalize_period(period_raw)
    s = strip(string(period_raw))
    su = uppercase(s)
    if su in ("OT", "POT")
        return :OT
    end
    p = tryparse(Int, s)
    if p !== nothing
        return p
    end
    # Handle values like 1.0
    pf = tryparse(Float64, s)
    if pf !== nothing
        return Int(round(pf))
    end
    error("Unsupported period value: $s")
end

period_duration_seconds(period_code) = period_code == :OT ? OT_PERIOD_SECONDS : REGULATION_PERIOD_SECONDS

period_sort_key(period_code) = period_code == :OT ? 99 : Int(period_code)

function period_start_elapsed_seconds(period_code)
    if period_code == :OT
        return 3 * REGULATION_PERIOD_SECONDS
    end
    return (period_code - 1) * REGULATION_PERIOD_SECONDS
end

"""Convert (period, game clock remaining) to elapsed game seconds."""
function elapsed_seconds(period_code, game_clock_raw)
    remain = parse_clock_seconds(game_clock_raw)
    return period_start_elapsed_seconds(period_code) + (period_duration_seconds(period_code) - remain)
end

function parse_image_index(image_id_raw)
    s = string(image_id_raw)
    m = match(r"_(\d+)$", s)
    return m === nothing ? missing : parse(Int, m.captures[1])
end

function rename_if_present!(df::DataFrame, mapping::Dict{String,Symbol})
    rename_map = Dict{Symbol,Symbol}()
    existing_syms = Set(Symbol.(names(df)))
    for (old_name, new_name) in mapping
        old_sym = Symbol(old_name)
        if old_sym in existing_syms
            rename_map[old_sym] = new_name
        end
    end
    if !isempty(rename_map)
        rename!(df, rename_map)
    end
    return df
end

function parse_int_or_missing(x)
    if x === missing
        return missing
    end
    v = tryparse(Int, strip(string(x)))
    return v === nothing ? missing : v
end

function parse_args(args)
    opts = Dict(
        "data-dir" => "data",
        "out-dir" => "processed",
        "horizon" => "5.0",
        "min-frames" => "40",
        "min-puck-points" => "20"
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = replace(arg, "--" => "")
            if !(key in keys(opts))
                error("Unknown option: $arg")
            end
            if i == length(args)
                error("Missing value for option: $arg")
            end
            opts[key] = args[i + 1]
            i += 2
        else
            error("Unexpected argument: $arg")
        end
    end

    return (
        data_dir = opts["data-dir"],
        out_dir = opts["out-dir"],
        horizon = parse(Float64, opts["horizon"]),
        min_frames = parse(Int, opts["min-frames"]),
        min_puck_points = parse(Int, opts["min-puck-points"])
    )
end

"""Return one row per frame with inferred sub-second timing."""
function build_frame_table(track::DataFrame)
    valid_mask = .!ismissing.(track.period_code) .& .!ismissing.(track.game_clock)
    track = track[valid_mask, :]

    frame_cols = [:image_id, :period_code, :game_clock]
    frames = unique(track[:, frame_cols])

    frames.image_index = [parse_image_index(x) for x in frames.image_id]
    frames.image_index_fallback = collect(1:nrow(frames))

    frames.period_sort = [period_sort_key(x) for x in frames.period_code]
    sort!(frames, [:period_sort, :image_index, :image_index_fallback])

    per_second_counts = combine(groupby(frames, [:period_code, :game_clock]), nrow => :nframes)
    fps_estimate = median(per_second_counts.nframes)
    fps = max(1.0, float(fps_estimate))

    ranks = Vector{Int}(undef, nrow(frames))
    for sdf in groupby(frames, [:period_code, :game_clock])
        idxs = parentindices(sdf)[1]
        sort!(idxs)
        for (k, idx) in enumerate(idxs)
            ranks[idx] = k
        end
    end

    frames.frame_rank_in_second = ranks
    frames.t_abs = [elapsed_seconds(frames.period_code[i], frames.game_clock[i]) + (frames.frame_rank_in_second[i] - 1) / fps for i in 1:nrow(frames)]

    return frames, fps
end

"""Find up to five skaters active for a team at a given (period, clock)."""
function active_skaters(shifts::DataFrame, team_name::AbstractString, period_code, clock_seconds::Int; n::Int=5)
    ids = Int[]
    for row in eachrow(shifts)
        if row.team != team_name
            continue
        end
        if row.period_code != period_code
            continue
        end
        pid = row.player_id_int
        if pid === missing
            continue
        end

        st = row.start_clock_sec
        en = row.end_clock_sec
        if st >= clock_seconds > en
            push!(ids, pid)
        end
    end

    unique_ids = sort!(unique(ids))
    if length(unique_ids) >= n
        return unique_ids[1:n]
    end
    return vcat(unique_ids, fill(missing, n - length(unique_ids)))
end

function game_prefix_from_file(path::String, suffix::String)
    name = splitpath(path)[end]
    return replace(name, suffix => "")
end

function tracking_team_side(entry_team::AbstractString, home_team::AbstractString, away_team::AbstractString)
    if entry_team == home_team
        return "Home", "Away"
    elseif entry_team == away_team
        return "Away", "Home"
    else
        # Fallback for rare naming mismatch
        return "Home", "Away"
    end
end

function find_tracking_files(data_dir::String, prefix::String)
    all_files = readdir(data_dir; join=true)
    target_prefix = prefix * ".Tracking_"
    files = filter(f -> endswith(f, ".csv") && occursin(target_prefix, splitpath(f)[end]), all_files)
    return sort(files)
end

function read_events(path::String)
    df = CSV.read(path, DataFrame; missingstring=["", "n/a"])
    rename_if_present!(df, Dict(
        "Date" => :date,
        "Home_Team" => :home_team,
        "Away_Team" => :away_team,
        "Period" => :period,
        "Clock" => :clock,
        "Team" => :team,
        "Player_Id" => :player_id,
        "Event" => :event,
        "Detail_1" => :detail_1,
        "X_Coordinate" => :event_x,
        "Y_Coordinate" => :event_y
    ))

    df.period_code = [normalize_period(x) for x in df.period]
    df.t_abs = [elapsed_seconds(df.period_code[i], df.clock[i]) for i in 1:nrow(df)]
    return df
end

function read_shifts(path::String)
    df = CSV.read(path, DataFrame; missingstring=["", "n/a"])
    rename_if_present!(df, Dict(
        "Team" => :team,
        "Player_Id" => :player_id,
        "period" => :period,
        "start_clock" => :start_clock,
        "end_clock" => :end_clock
    ))

    df.period_code = [normalize_period(x) for x in df.period]
    df.player_id_int = [parse_int_or_missing(x) for x in df.player_id]
    df.start_clock_sec = [parse_clock_seconds(x) for x in df.start_clock]
    df.end_clock_sec = [parse_clock_seconds(x) for x in df.end_clock]
    return df
end

function read_tracking(paths::Vector{String})
    dfs = DataFrame[]
    for path in paths
        t = CSV.read(path, DataFrame; missingstring=["", "n/a"])
        rename_if_present!(t, Dict(
            "Image Id" => :image_id,
            "Period" => :period,
            "Game Clock" => :game_clock,
            "Player or Puck" => :entity,
            "Team" => :team_side,
            "Player Jersey Number" => :jersey,
            "Rink Location X (Feet)" => :x,
            "Rink Location Y (Feet)" => :y,
            "Rink Location Z (Feet)" => :z
        ))
        append!(dfs, [t])
    end

    track = vcat(dfs...)
    track.period_code = [normalize_period(x) for x in track.period]
    track.jersey_int = [parse_int_or_missing(x) for x in track.jersey]

    frames, fps = build_frame_table(track)
    track = leftjoin(track, frames[:, [:image_id, :t_abs]], on=:image_id)
    return track, frames, fps
end

function shot_outcome(events::DataFrame, team::AbstractString, t0::Real, horizon::Real)
    mask = map(1:nrow(events)) do i
        ev_name = events.event[i]
        ev_team = events.team[i]
        t = events.t_abs[i]
        !ismissing(ev_name) && !ismissing(ev_team) &&
        ev_name == "Shot" && ev_team == team && t > t0 && t <= t0 + horizon
    end
    return any(mask)
end

function row_lookup_by_image(track::DataFrame)
    image_map = Dict{String,Vector{Int}}()
    for i in 1:nrow(track)
        img = string(track.image_id[i])
        if haskey(image_map, img)
            push!(image_map[img], i)
        else
            image_map[img] = [i]
        end
    end
    return image_map
end

function add_velocity_columns!(seq_df::DataFrame)
    n = nrow(seq_df)
    vx = Vector{Union{Missing,Float64}}(missing, n)
    vy = Vector{Union{Missing,Float64}}(missing, n)
    vz = Vector{Union{Missing,Float64}}(missing, n)

    for i in 2:n
        x1 = seq_df.puck_x[i - 1]
        y1 = seq_df.puck_y[i - 1]
        z1 = seq_df.puck_z[i - 1]
        x2 = seq_df.puck_x[i]
        y2 = seq_df.puck_y[i]
        z2 = seq_df.puck_z[i]
        dt = seq_df.t_rel[i] - seq_df.t_rel[i - 1]

        if dt <= 1e-3 || any(ismissing, (x1, y1, z1, x2, y2, z2))
            continue
        end

        vx[i] = (x2 - x1) / dt
        vy[i] = (y2 - y1) / dt
        vz[i] = (z2 - z1) / dt
    end

    seq_df.puck_vx = vx
    seq_df.puck_vy = vy
    seq_df.puck_vz = vz
    return seq_df
end

function process_game!(
    all_rows::DataFrame,
    index_rows::DataFrame,
    events_path::String,
    shifts_path::String,
    tracking_paths::Vector{String},
    horizon::Float64,
    min_frames::Int,
    min_puck_points::Int,
    seq_counter::Base.RefValue{Int}
)
    events = read_events(events_path)
    shifts = read_shifts(shifts_path)
    track, frames, fps = read_tracking(tracking_paths)
    by_image = row_lookup_by_image(track)

    zone_mask = map(1:nrow(events)) do i
        ev_name = events.event[i]
        entry_type = events.detail_1[i]
        !ismissing(ev_name) && !ismissing(entry_type) &&
        ev_name == "Zone Entry" && entry_type in ("Carried", "Dumped", "Played")
    end
    zone_entries = events[zone_mask, :]

    if nrow(zone_entries) == 0
        return
    end

    home_team = string(events.home_team[1])
    away_team = string(events.away_team[1])
    game_id = game_prefix_from_file(events_path, ".Events.csv")

    sort!(frames, :t_abs)

    for ev in eachrow(zone_entries)
        t0 = ev.t_abs
        t1 = t0 + horizon
        frame_window = frames[(frames.t_abs .>= t0) .& (frames.t_abs .<= t1), :]

        if nrow(frame_window) < min_frames
            continue
        end

        clock_sec = parse_clock_seconds(ev.clock)
        entry_team = string(ev.team)
        defend_team = entry_team == home_team ? away_team : home_team
        entry_side, defend_side = tracking_team_side(entry_team, home_team, away_team)

        entry_skaters = active_skaters(shifts, entry_team, ev.period_code, clock_sec; n=5)
        defend_skaters = active_skaters(shifts, defend_team, ev.period_code, clock_sec; n=5)

        # If shift-based lookup misses players, keep the sequence and fill missing coordinates downstream.
        seq_counter[] += 1
        seq_id = seq_counter[]

        seq_df = DataFrame(
            seq_id = Int[],
            game_id = String[],
            entry_period = String[],
            entry_clock = String[],
            entry_team = String[],
            defend_team = String[],
            entry_type = String[],
            shot_within_horizon = Bool[],
            fps_estimate = Float64[],
            frame_index = Int[],
            t_rel = Float64[],
            frame_image_id = String[],
            frame_game_clock = String[],
            puck_x = Union{Missing,Float64}[],
            puck_y = Union{Missing,Float64}[],
            puck_z = Union{Missing,Float64}[]
        )

        for i in 1:5
            seq_df[!, Symbol("entry_p$(i)_id")] = Union{Missing,Int}[]
            seq_df[!, Symbol("entry_p$(i)_x")] = Union{Missing,Float64}[]
            seq_df[!, Symbol("entry_p$(i)_y")] = Union{Missing,Float64}[]
            seq_df[!, Symbol("defend_p$(i)_id")] = Union{Missing,Int}[]
            seq_df[!, Symbol("defend_p$(i)_x")] = Union{Missing,Float64}[]
            seq_df[!, Symbol("defend_p$(i)_y")] = Union{Missing,Float64}[]
        end

        shot_flag = shot_outcome(events, entry_team, t0, horizon)

        for (k, fr) in enumerate(eachrow(frame_window))
            image_id = string(fr.image_id)
            rows = get(by_image, image_id, Int[])

            puck_x = missing
            puck_y = missing
            puck_z = missing

            player_xy = Dict{Tuple{String,Int},Tuple{Union{Missing,Float64},Union{Missing,Float64}}}()

            for ridx in rows
                entity = track.entity[ridx]
                if entity == "Puck"
                    puck_x = track.x[ridx]
                    puck_y = track.y[ridx]
                    puck_z = track.z[ridx]
                elseif entity == "Player"
                    team_side = string(track.team_side[ridx])
                    jersey = track.jersey_int[ridx]
                    if jersey !== missing
                        player_xy[(team_side, jersey)] = (track.x[ridx], track.y[ridx])
                    end
                end
            end

            row_nt = (
                seq_id = seq_id,
                game_id = game_id,
                entry_period = string(ev.period),
                entry_clock = string(ev.clock),
                entry_team = entry_team,
                defend_team = defend_team,
                entry_type = string(ev.detail_1),
                shot_within_horizon = shot_flag,
                fps_estimate = fps,
                frame_index = k,
                t_rel = (k - 1) / fps,
                frame_image_id = image_id,
                frame_game_clock = string(fr.game_clock),
                puck_x = puck_x,
                puck_y = puck_y,
                puck_z = puck_z
            )
            push!(seq_df, row_nt; cols=:subset)

            for i in 1:5
                e_pid = entry_skaters[i]
                d_pid = defend_skaters[i]

                seq_df[k, Symbol("entry_p$(i)_id")] = e_pid
                seq_df[k, Symbol("defend_p$(i)_id")] = d_pid

                if e_pid !== missing && haskey(player_xy, (entry_side, e_pid))
                    ex, ey = player_xy[(entry_side, e_pid)]
                    seq_df[k, Symbol("entry_p$(i)_x")] = ex
                    seq_df[k, Symbol("entry_p$(i)_y")] = ey
                end

                if d_pid !== missing && haskey(player_xy, (defend_side, d_pid))
                    dx, dy = player_xy[(defend_side, d_pid)]
                    seq_df[k, Symbol("defend_p$(i)_x")] = dx
                    seq_df[k, Symbol("defend_p$(i)_y")] = dy
                end
            end
        end

        puck_points = count(.!ismissing.(seq_df.puck_x) .& .!ismissing.(seq_df.puck_y) .& .!ismissing.(seq_df.puck_z))
        if puck_points < min_puck_points
            seq_counter[] -= 1
            continue
        end

        add_velocity_columns!(seq_df)

        append!(all_rows, seq_df; cols=:union)

        push!(index_rows, (
            seq_id = seq_id,
            game_id = game_id,
            entry_period = string(ev.period),
            entry_clock = string(ev.clock),
            entry_type = string(ev.detail_1),
            entry_team = entry_team,
            defend_team = defend_team,
            shot_within_horizon = shot_flag,
            n_frames = nrow(seq_df),
            n_puck_points = puck_points,
            fps_estimate = fps
        ))
    end
end

function main()
    cfg = parse_args(ARGS)
    mkpath(cfg.out_dir)

    data_dir = cfg.data_dir
    all_files = readdir(data_dir; join=true)
    event_files = sort(filter(f -> endswith(f, ".Events.csv"), all_files))

    all_rows = DataFrame()
    index_rows = DataFrame(
        seq_id = Int[],
        game_id = String[],
        entry_period = String[],
        entry_clock = String[],
        entry_type = String[],
        entry_team = String[],
        defend_team = String[],
        shot_within_horizon = Bool[],
        n_frames = Int[],
        n_puck_points = Int[],
        fps_estimate = Float64[]
    )

    seq_counter = Ref(0)
    games_seen = 0

    for events_path in event_files
        prefix = game_prefix_from_file(events_path, ".Events.csv")
        shifts_path = joinpath(data_dir, basename(prefix) * ".Shifts.csv")
        if !isfile(shifts_path)
            @warn "Missing shifts file, skipping game" events_path shifts_path
            continue
        end

        tracking_paths = find_tracking_files(data_dir, basename(prefix))
        if isempty(tracking_paths)
            @warn "Missing tracking files, skipping game" events_path
            continue
        end

        games_seen += 1
        process_game!(
            all_rows,
            index_rows,
            events_path,
            shifts_path,
            tracking_paths,
            cfg.horizon,
            cfg.min_frames,
            cfg.min_puck_points,
            seq_counter
        )
    end

    seq_out = joinpath(cfg.out_dir, "neural_ode_sequences_long.csv")
    idx_out = joinpath(cfg.out_dir, "neural_ode_sequence_index.csv")

    CSV.write(seq_out, all_rows)
    CSV.write(idx_out, index_rows)

    println("Processed games: $(games_seen)")
    println("Exported sequences: $(nrow(index_rows))")
    println("Long-form rows: $(nrow(all_rows))")
    println("Saved: $seq_out")
    println("Saved: $idx_out")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
