#!/usr/bin/env julia
# Quick validation: analyze rink coordinates and preprocessing output

using CSV, DataFrames, Statistics

println("=" ^ 60)
println("VALIDATION TASK 1: Rink Coordinate Analysis")
println("=" ^ 60)

# Read sample tracking data
track = CSV.read("data/2025-10-11.Team.A.@.Team.D.Tracking_P1.csv", DataFrame)
puck_data = filter(row -> row[Symbol("Player or Puck")] == "Puck", track)

# Extract coordinates, convert to Float64, filter out NaN/missing
x_raw = puck_data[!, Symbol("Rink Location X (Feet)")]
y_raw = puck_data[!, Symbol("Rink Location Y (Feet)")]
z_raw = puck_data[!, Symbol("Rink Location Z (Feet)")]

# Convert to numeric and filter valid values
x = [Float64(v) for v in x_raw if v isa Number && !isnan(Float64(v))]
y = [Float64(v) for v in y_raw if v isa Number && !isnan(Float64(v))]
z = [Float64(v) for v in z_raw if v isa Number && !isnan(Float64(v))]

if isempty(x) || isempty(y) || isempty(z)
    println("⚠️  Warning: No valid puck coordinates found in sample data")
    println("    Proceeding with defaults...")
    x = [-100.0, 100.0]
    y = [-30.0, 30.0]
    z = [0.0, 10.0]
end

println("\nPuck coordinate bounds (sample game):")
println("  X (attack direction): $(minimum(x)) to $(maximum(x)) feet")
println("  Y (width): $(minimum(y)) to $(maximum(y)) feet")
println("  Z (height): $(minimum(z)) to $(maximum(z)) feet")
println("  X range: $(maximum(x) - minimum(x)) feet")
println("  Y range: $(maximum(y) - minimum(y)) feet")

println("\n📌 Recommended reward thresholds:")
half_x = (minimum(x) + maximum(x)) / 2
offensive_threshold = half_x + (maximum(x) - minimum(x)) / 4
net_y = 0  # Assume net centered at y=0
danger_distance = 10

println("  zone_score threshold (offensive zone starts): x ≈ $offensive_threshold")
println("  danger_score threshold (distance to net): y ≈ $danger_distance")

println("\n" ^ 2)
println("=" ^ 60)
println("VALIDATION TASK 2: Run Preprocessing on Sample Game")
println("=" ^ 60)

# This will be done separately with the actual script
println("\nRun preprocessing with: julia preprocess_neural_ode.jl --data-dir data --out-dir processed")
println("Then inspect: processed/neural_ode_sequences_long.csv")
