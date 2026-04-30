# ZoneOpt.jl — Project Structure & Operations Guide

Welcome to the comprehensive directory map and operational guide for **ZoneOpt.jl**, an ice hockey zone entry optimization framework built using Julia and Scientific Machine Learning (SciML).

This document serves as the master index for the entire workspace (`/home/gretar/6_7320`), explaining what each folder does, what each script is responsible for, and how the core technical workflow operates seamlessly together.

---

## 1. Directory Breakdown

### 📂 `src/` — The Core Logic
This folder drives the computational framework, mathematics, and data staging.
- **`neural_ode_framework.jl`**: The core library. Contains the mathematical formulation for the Universal Differential Equation ($du/dt = f(u,p,t) + NN(u, \theta)$). It defines the physics priors (drag, gravity), the neural network architecture, Adjoint backpropagation parameters, and the agentic reward function evaluation metric.
- **`preprocess_neural_ode.jl`**: The data engineering pipeline. It cross-references the game event logs against the high-frequency continuous tracking streams to extract clean, synchronous 5-second trajectories matching each "Zone Entry" and constructs the 23-dimensional contextual state tensors (player coordinate maps).

### 📂 `scripts/` — Execution Scripts
Scripts intended to be executed from the terminal sequentially or independently:
- **`run_training.jl`**: Runs the primary parallel computing environment via `EnsembleProblem`s. Trains the Neural ODE using `Zygote` and `SciMLSensitivity`.
- **`run_pipeline.jl`**: A master wrapper script simulating end-to-end data processing and model evaluation.
- **`agent_optimization.jl`**: Uses `ForwardDiff.jl` iteratively against the trained Neural ODE environment to calculate Continuous Action Policy Gradients (tuning $u_0$ velocity vectors for optimal reward).
- **`run_recommendations_on_testset.jl`**: Evaluates the model's tactical choices across a held-out test set against historically successful decisions.
- **`evaluate_neural_ode.jl` / `quick_evaluate_and_visualize.jl` / `quick_checkpoint_eval.jl`**: Modular evaluation scripts fetching saved `.jls` network topologies to calculate test RMSE and compare prediction against actual baseline outputs.
- **`visualize_trajectory_predictions.jl` / `generate_report_figures.jl`**: Generates multi-frame plots, pathing visualizations, and GIFs using `Makie.jl`.
- **`sensitivity_analysis.jl`**: Evaluates model Jacobians with respect to various input parameters for analytical reporting.
- **`validate_coordinates.jl` / `validate_preprocessed.jl`**: Coordinate boundary checks for verifying Stathletes dimensions vs. our mathematical reward thresholds.
- **`train_neural_ode.slurm`**: The SLURM job submission script for executing remote HPC training.

### 📂 `data/` — Raw Data
Contains the extracted datasets from the Stathletes Big Data Cup (e.g. `2025-10-11.Team.A.@.Team.D.Events.csv`, `Shifts.csv`, `Tracking.csv`). These are unprocessed, raw events and spatial mappings directly from the tracking systems.

### 📂 `processed/` — Engineered Outputs
The resulting datasets after running `preprocess_neural_ode.jl`:
- **`neural_ode_sequences_long.csv`**: Contains the full synchronized 6D ($x, y, z, v_x, v_y, v_z$) continuous paths interpolating exactly 5 seconds.
- **`neural_ode_sequence_index.csv`**: Metadata indexing which sequences correspond to which entry strategies, game IDs, and context states.

### 📂 `models/` — Learned Architectures
Holds the serialized SciML weights.
- **`epoch_XXX.jls` / `epoch_XXX_best.jls`**: Saved parameter snapshots indexed chronologically utilizing `BSON.jl` or `JLD2.jl` serializers.
- **`SELECTED_CHECKPOINT.md`**: Text indicating the designated optimal network weights chosen for agent utilization.

### 📂 `evaluation/` & `evaluation_full/` — Experimental Metrics
Contains metrics collected throughout evaluation phases (`evaluate_neural_ode.jl`).
- **`baseline_summary.csv`**, **`physics_calibration_grid.csv`**, **`quick_checkpoint_summary.csv`**: Metrics for test RMSE, error by entry type (Carry vs. Dump), and accuracy baselining.
- **`BASELINE_README.md` / `CALIBRATION_SUMMARY.md`**: Quick reference context for evaluation results.

### 📂 `visualizations/` — Rendered Media
The resulting plots from the visualization scripts. Includes trajectory comparison graphics (`epoch140_preview/`), tactical model decision logs (`policy_decisions_epoch140_test.csv`), and multi-state sequential animations (`policy_gifs/`).

### 📂 `logs/` — Remote Computing Logs
Outputs from the cluster compute allocation via `train_neural_ode.slurm` (`training_XXXX.out` and `.err`).

### 📂 `presentation/` — Final Deliverables Media
Contains `SLIDES.md` and related images (`figures/`) exclusively utilized in the final presentation deck.

---

## 2. Root Files & Documentation
- **Julia Environment Files**: 
  - `Project.toml` / `Manifest.toml` / `LocalPreferences.toml`: Defines the strict versions and dependency trees for packages (e.g., `DiffEqFlux`, `OrdinaryDiffEq`, `Zygote`, `ForwardDiff`, `DataFrames`).
- **Reports and Proposals**:
  - `final_project_proposal.md`: The original structural proposal for the architecture.
  - `FINAL_REPORT.md` / `DETAILED_PROJECT_REPORT.md`: Comprehensive breakdowns of methodology, mechanics, and results.
  - `TECHNICAL_PROJECT_REPORT_18337.md`: A heavily specialized document breaking algorithmic choices (Adjoint scaling, dual numbers, VJPs) aimed directly at the parallel computing / SciML class.
  - `IMPLEMENTATION_VALIDATION.md` & `PROJECT_CHECKLIST.md`: Project management debugging checklists and verification logs.
  - `README.md`: Short description and root directory reference.

---

## 3. Operations Workflow: How Everything Connects

If running everything from scratch, the system executes as follows:

1. **Environment Setup:** Start Julia in the workspace root environment (`julia --project=.`).
2. **Data Pipeline:** We orchestrate the timeline utilizing `julia src/preprocess_neural_ode.jl`. The Raw JSON/CSV tracker metrics from `/data/` are distilled, converted into 23-dimensional relative coordinate contexts seamlessly synchronized against actual target coordinates, and deposited entirely into `/processed/`.
3. **Training the UDE:** Execute `julia scripts/run_training.jl`. The Neural ODE accesses the `/processed/` cache, wraps the batch sequentially via `EnsembleProblem` computing layers across available threads, generates gradients sequentially across the solver using `InterpolatingAdjoint` backwards, steps optimizing utilizing *ADAM/BFGS*, calculates continuous trajectory $L2$ errors, and flushes iteration outputs directly into `/models/` (and logs `/logs/`).
4. **Metric Logging & Inference Validation:** Running `julia scripts/evaluate_neural_ode.jl` loads the best model via BSON, evaluates its capacity to recreate testing traces via test-RMSE metrics, scales to physical calibration boundaries, and places all summary tracking outputs into `/evaluation/`.
5. **Agent Inference Optimization:** Through `julia scripts/agent_optimization.jl`, the proxy ODE locks its neural weights $\theta$. We inject varied tactic initialization conditions $u_0$ directly against rigid defensive mappings. The script parses the resulting forward path's depth utilizing our zone scoring configurations. Through the `ForwardDiff` framework, tactical trajectories are derived to calculate gradients directly maximizing expected success on net.
6. **Visual Review:** Through `scripts/visualize_trajectory_predictions.jl`, the 3D plots are rendered displaying the paths against virtual rink configurations in `/visualizations/`.