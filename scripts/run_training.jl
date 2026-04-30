using Pkg
using Dates

timestamp() = string(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))

println("[$(timestamp())] Activating Julia project")
flush(stdout)
Pkg.activate(".")

required_packages = [
    "CSV",
    "DataFrames",
    "Flux",
    "OrdinaryDiffEq",
    "DiffEqFlux",
    "SciMLSensitivity",
    "ProgressMeter",
]

project_deps = Set(String.(keys(Pkg.project().dependencies)))
missing_packages = [pkg for pkg in required_packages if !(pkg in project_deps)]

if !isempty(missing_packages)
    println("[$(timestamp())] Installing missing Julia packages: $(join(missing_packages, ", "))")
    flush(stdout)
    Pkg.add(missing_packages)
    println("[$(timestamp())] Package install complete")
    flush(stdout)
end

println("[$(timestamp())] Instantiating environment")
flush(stdout)
Pkg.instantiate()
println("[$(timestamp())] Environment ready")
flush(stdout)

include("../src/neural_ode_framework.jl")
using .ZoneOptNeuralODEFramework

default_epochs = 50
epochs = try
    parse(Int, get(ENV, "ZONEOPT_EPOCHS", string(default_epochs)))
catch
    default_epochs
end

default_lr_patience = 8
lr_patience = try
    parse(Int, get(ENV, "ZONEOPT_LR_PATIENCE", string(default_lr_patience)))
catch
    default_lr_patience
end

default_early_stopping_patience = 20
early_stopping_patience = try
    parse(Int, get(ENV, "ZONEOPT_EARLY_STOPPING_PATIENCE", string(default_early_stopping_patience)))
catch
    default_early_stopping_patience
end

default_min_learning_rate = 1e-5
min_learning_rate = try
    parse(Float64, get(ENV, "ZONEOPT_MIN_LEARNING_RATE", string(default_min_learning_rate)))
catch
    default_min_learning_rate
end

default_lr_decay_factor = 0.5
lr_decay_factor = try
    parse(Float64, get(ENV, "ZONEOPT_LR_DECAY_FACTOR", string(default_lr_decay_factor)))
catch
    default_lr_decay_factor
end

model_dir = get(ENV, "ZONEOPT_MODEL_DIR", joinpath(pwd(), "models"))
resume_checkpoint = get(ENV, "ZONEOPT_RESUME_CHECKPOINT", "")
resume_checkpoint = isempty(strip(resume_checkpoint)) ? nothing : resume_checkpoint

println("[$(timestamp())] Training epochs configured: $(epochs) (override with ZONEOPT_EPOCHS)")
println("[$(timestamp())] Checkpoint directory: $(model_dir) (override with ZONEOPT_MODEL_DIR)")
println("[$(timestamp())] LR patience: $(lr_patience) (override with ZONEOPT_LR_PATIENCE)")
println("[$(timestamp())] Early stopping patience: $(early_stopping_patience) (override with ZONEOPT_EARLY_STOPPING_PATIENCE)")
println("[$(timestamp())] Min learning rate: $(min_learning_rate) (override with ZONEOPT_MIN_LEARNING_RATE)")
println("[$(timestamp())] LR decay factor: $(lr_decay_factor) (override with ZONEOPT_LR_DECAY_FACTOR)")
if !isnothing(resume_checkpoint)
    println("[$(timestamp())] Resume checkpoint: $(resume_checkpoint) (override with ZONEOPT_RESUME_CHECKPOINT)")
end
flush(stdout)

println("[$(timestamp())] Starting training pipeline...")
flush(stdout)
result = run_training_pipeline(
    hidden_dim=64,
    depth=3,
    config=TrainingConfig(
        epochs=epochs,
        lr_patience=lr_patience,
        early_stopping_patience=early_stopping_patience,
        min_learning_rate=min_learning_rate,
        lr_decay_factor=lr_decay_factor,
        checkpoint_dir=model_dir,
        checkpoint_every=1,
    ),
    resume_checkpoint=resume_checkpoint,
)

println("\nTraining complete!")
println("Test RMSE: $(result.test_rmse)")
println("Test Loss: $(result.test_loss)")
println("Decision Validity: $(result.decision_validity)")
println("Splits: train=$(result.splits.train), val=$(result.splits.val), test=$(result.splits.test)")
