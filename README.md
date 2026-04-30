# ZoneOpt.jl

ZoneOpt.jl is a Julia-based framework optimizing ice hockey zone entry strategies through Neural ODEs. By extracting high-frequency tracking data, mapping defensive formations into a context vector, and simulating puck physics over time, ZoneOpt.jl provides AI-agentic recommendations for tactical play execution (dump vs. carry).

## Project Structure
- `src/`: Core framework, model definition, and data preprocessing logic (`neural_ode_framework.jl`, `preprocess_neural_ode.jl`).
- `scripts/`: Executable scripts for training, testing, and visualizations.
- `models/`: Saved model checkpoints.
- `processed/`: Processed CSV tracking datasets.
- `data/`: Raw event and tracking logs.
- `presentation/`: Slides for project defense.

## Quickstart

### 1. Preprocess Data
Run the preprocessing pipeline to transform raw tracking streams into 5-second entry trajectories:
```bash
julia src/preprocess_neural_ode.jl
```

### 2. Train the Model
Using the Adjoint method, optimize Neural ODE parameters:
```bash
julia scripts/run_training.jl
```
Or run the full pipeline verifying outputs: `julia scripts/run_pipeline.jl`

### 3. Agentic Optimization & Decisions
Evaluate decision validity and recommend tactical strategies:
```bash
julia scripts/run_recommendations_on_testset.jl
```
Refine puck velocity for peak reward via Zygote:
```bash
julia scripts/agent_optimization.jl
```

### 4. Sensitivity Analysis
Evaluate model Jacobian outputs w.r.t initial velocity:
```bash
julia scripts/sensitivity_analysis.jl
```

## Documentation
See `FINAL_REPORT.md` for methodology, architecture details, and results. Also refer to `IMPLEMENTATION_VALIDATION.md`.
