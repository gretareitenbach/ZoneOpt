#!/usr/bin/env julia
# Validate preprocessing output and run training/evaluation pipeline

using CSV, DataFrames
using Dates

timestamp() = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")

function log_step(msg)
    println("[$(timestamp())] $msg")
    flush(stdout)
end

start_time = time()

println("=" ^ 80)
println("STEP 1: Validating Preprocessing Output")
println("=" ^ 80)
log_step("Loading processed CSV files")

try
    long_df = CSV.read("processed/neural_ode_sequences_long.csv", DataFrame)
    index_df = CSV.read("processed/neural_ode_sequence_index.csv", DataFrame)
    log_step("CSV load complete")
    
    println("\n✅ Successfully loaded CSV files")
    println("   Sequences: $(nrow(index_df))")
    println("   Long-form rows: $(nrow(long_df))")
    println("   Long-form columns: $(ncol(long_df))")
    println("   Index columns: $(ncol(index_df))")
    
    # Check for required columns
    log_step("Validating required columns")
    required_cols = [:seq_id, :entry_type, :shot_within_horizon, :puck_x, :puck_y, :puck_z]
    present_cols = Set(Symbol.(names(long_df)))
    missing_cols = [c for c in required_cols if !(c in present_cols)]
    
    if isempty(missing_cols)
        println("\n✅ All required columns present in long-form data")
    else
        println("\n⚠️  Missing columns: $missing_cols")
        exit(1)
    end
    
    # Check data completeness
    log_step("Computing completeness metrics")
    total_rows = nrow(long_df)
    puck_x_valid = count(!ismissing, long_df.puck_x)
    puck_y_valid = count(!ismissing, long_df.puck_y)
    puck_z_valid = count(!ismissing, long_df.puck_z)
    
    println("\n📊 Data completeness:")
    println("   Puck X: $(puck_x_valid) / $total_rows ($(round(100*puck_x_valid/total_rows))%)")
    println("   Puck Y: $(puck_y_valid) / $total_rows ($(round(100*puck_y_valid/total_rows))%)")
    println("   Puck Z: $(puck_z_valid) / $total_rows ($(round(100*puck_z_valid/total_rows))%)")
    
    shot_count = count(index_df.shot_within_horizon)
    println("\n🎯 Sequences with shots: $(shot_count) / $(nrow(index_df))")
    
    println("\n" ^ 2)
    println("=" ^ 80)
    println("STEP 2: Running Training Pipeline")
    println("=" ^ 80)
    log_step("Including run_training.jl")
    
    include("run_training.jl")
    log_step("Training script finished")
    
catch e
    println("\n❌ Error during validation: $e")
    println(stacktrace(catch_backtrace()))
    exit(1)
end

println("\n" ^ 2)
println("=" ^ 80)
println("✅ Pipeline Complete!")
println("=" ^ 80)
elapsed_minutes = round((time() - start_time) / 60; digits = 2)
log_step("Total pipeline elapsed minutes: $(elapsed_minutes)")
