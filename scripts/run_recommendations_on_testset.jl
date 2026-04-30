using CSV, DataFrames
include("../src/neural_ode_framework.jl")
using .ZoneOptNeuralODEFramework

# Load processed sequences
long_csv = joinpath("processed","neural_ode_sequences_long.csv")
index_csv = joinpath("processed","neural_ode_sequence_index.csv")
long_df, index_df = ZoneOptNeuralODEFramework.load_sequence_tables(long_csv, index_csv)
examples = ZoneOptNeuralODEFramework.build_sequence_examples(long_df, index_df)
train_set, val_set, test_set = ZoneOptNeuralODEFramework.split_sequences(examples; seed=42)

println("Loaded sequences: train=$(length(train_set)) val=$(length(val_set)) test=$(length(test_set))")

# Build model structure (must match checkpoint architecture)
ctx_len = length(first(examples).context)
model = ZoneOptNeuralODEFramework.build_model(ctx_len, train_set; hidden_dim=64, depth=3)

# Restore latest checkpoint - change path if you want another
checkpoint = "models/epoch_140.jls"
println("Restoring checkpoint: ", checkpoint)
payload = ZoneOptNeuralODEFramework.restore_model_checkpoint!(model, checkpoint)

out_dir = "visualizations"
mkpath(out_dir)
out_csv = joinpath(out_dir, "policy_decisions_epoch$(payload.epoch)_test.csv")

rows = Vector{Dict{String,Any}}()

for (i, seq) in enumerate(test_set)
    rec = ZoneOptNeuralODEFramework.recommend_action(model, seq)
    probs = rec.probabilities
    raw = rec.raw_scores
    push!(rows, Dict(
        "seq_index" => i,
        "seq_id" => seq.seq_id,
        "game_id" => seq.game_id,
        "best_strategy" => string(rec.best_strategy),
        "best_label" => rec.best_label,
        "confidence" => rec.confidence,
        "prob_carry" => get(probs, :carry, missing),
        "prob_dump" => get(probs, :dump, missing),
        "prob_pass" => get(probs, :pass, missing),
        "prob_shoot" => get(probs, :shoot, missing),
        "raw_carry" => get(raw, :carry, missing),
        "raw_dump" => get(raw, :dump, missing),
        "raw_pass" => get(raw, :pass, missing),
        "raw_shoot" => get(raw, :shoot, missing),
    ))
end

df = DataFrame(rows)
CSV.write(out_csv, df)
println("Wrote decisions to: ", out_csv)
