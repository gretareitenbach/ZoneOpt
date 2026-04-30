#!/bin/bash
#SBATCH --job-name=zoneopt_training
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

set -e
set -o pipefail

ts() {
    date '+%Y-%m-%d %H:%M:%S'
}

cd /home/gretar/6_7320
mkdir -p logs

echo "[$(ts)] Starting ZoneOpt Neural ODE training and evaluation..."
echo "[$(ts)] Job ID: $SLURM_JOB_ID"
echo "[$(ts)] Host: $(hostname)"
echo "[$(ts)] Working directory: $(pwd)"
echo "[$(ts)] CPU request: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "[$(ts)] Memory request: ${SLURM_MEM_PER_NODE:-unknown}"
echo "[$(ts)] Start time: $(date)"

echo "[$(ts)] Sourcing ~/.bashrc"
# Do not use nounset while sourcing system/user bashrc files.
set +u
source ~/.bashrc
set -u

ensure_julia() {
	if command -v julia >/dev/null 2>&1; then
		return 0
	fi

	if command -v module >/dev/null 2>&1; then
		for module_name in julia Julia julia/1.12.6 Julia/1.12.6; do
			if module load "$module_name" >/dev/null 2>&1; then
				break
			fi
		done
	fi

	command -v julia >/dev/null 2>&1
}

if ! ensure_julia; then
	echo "[$(ts)] ERROR: julia is not available on PATH and no module load succeeded"
	echo "[$(ts)] End time: $(date)"
	exit 127
fi

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export ZONEOPT_MODEL_DIR=${ZONEOPT_MODEL_DIR:-"$(pwd)/models"}
export ZONEOPT_RESUME_EPOCH=${ZONEOPT_RESUME_EPOCH:-70}
export ZONEOPT_EPOCHS=${ZONEOPT_EPOCHS:-250}
export ZONEOPT_LR_PATIENCE=${ZONEOPT_LR_PATIENCE:-8}
export ZONEOPT_EARLY_STOPPING_PATIENCE=${ZONEOPT_EARLY_STOPPING_PATIENCE:-20}

# Auto-resume behavior:
# 1) If ZONEOPT_RESUME_CHECKPOINT is set, use it directly.
# 2) Otherwise, find the highest-epoch checkpoint in ZONEOPT_MODEL_DIR.
if [ -z "${ZONEOPT_RESUME_CHECKPOINT:-}" ]; then
	latest_checkpoint=""
	latest_epoch=-1
	latest_is_best=0

	for checkpoint_path in "$ZONEOPT_MODEL_DIR"/epoch_*.jls; do
		[ -e "$checkpoint_path" ] || continue
		checkpoint_name=$(basename "$checkpoint_path")

		if [[ "$checkpoint_name" =~ ^epoch_([0-9]+)(_best)?\.jls$ ]]; then
			epoch_num=$((10#${BASH_REMATCH[1]}))
			is_best=0
			if [ -n "${BASH_REMATCH[2]}" ]; then
				is_best=1
			fi

			if [ "$epoch_num" -gt "$latest_epoch" ]; then
				latest_epoch=$epoch_num
				latest_checkpoint="$checkpoint_path"
				latest_is_best=$is_best
			elif [ "$epoch_num" -eq "$latest_epoch" ] && [ "$latest_is_best" -eq 1 ] && [ "$is_best" -eq 0 ]; then
				# If both exist for the same epoch, prefer epoch_XXX.jls.
				latest_checkpoint="$checkpoint_path"
				latest_is_best=0
			fi
		fi
	done

	if [ -n "$latest_checkpoint" ]; then
		export ZONEOPT_RESUME_CHECKPOINT="$latest_checkpoint"
		export ZONEOPT_RESUME_EPOCH=$latest_epoch
	fi
fi

echo "[$(ts)] JULIA_NUM_THREADS=$JULIA_NUM_THREADS"
echo "[$(ts)] ZONEOPT_EPOCHS=$ZONEOPT_EPOCHS"
echo "[$(ts)] ZONEOPT_MODEL_DIR=$ZONEOPT_MODEL_DIR"
echo "[$(ts)] ZONEOPT_RESUME_EPOCH=$ZONEOPT_RESUME_EPOCH"
echo "[$(ts)] ZONEOPT_LR_PATIENCE=$ZONEOPT_LR_PATIENCE"
echo "[$(ts)] ZONEOPT_EARLY_STOPPING_PATIENCE=$ZONEOPT_EARLY_STOPPING_PATIENCE"
if [ -n "${ZONEOPT_RESUME_CHECKPOINT:-}" ]; then
	echo "[$(ts)] ZONEOPT_RESUME_CHECKPOINT=$ZONEOPT_RESUME_CHECKPOINT"
else
	echo "[$(ts)] No resume checkpoint found; training starts from epoch 1"
fi
echo "[$(ts)] Using Julia at: $(command -v julia)"
echo "[$(ts)] Launching Julia pipeline"
set +e
julia --project=. run_pipeline.jl
status=$?
set -e
echo "[$(ts)] Julia process exited with code $status"

if [ $status -ne 0 ]; then
	echo "[$(ts)] Training pipeline failed with exit code $status"
	echo "[$(ts)] End time: $(date)"
	exit $status
fi

echo "[$(ts)] Training complete!"
echo "[$(ts)] End time: $(date)"
echo "[$(ts)] Check results in current directory:"
ls -lh *.out *.err *.csv models/*.jls 2>/dev/null || echo "No results found"
