#!/usr/bin/env bash
# Convert the sim double-insert HDF5 datasets to LeRobot as a SLURM array (smallest-first via
# sim_double_insert_dirs.txt). Each task converts ONE dataset into NODE-LOCAL /scratch (LeRobot's
# per-frame PNG churn stays off the shared nas6 NFS), then rsyncs the final compact dataset to
# $DATA_ROOT/lerobot/<name>. HF push is a SEPARATE step (auth not wired here).
#   sbatch ~/hdf5_to_lerobot/convert_di_array.sbatch
#SBATCH --job-name=diconv
#SBATCH --partition=cpu
#SBATCH --array=0-26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/diconv_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/diconv_%A_%a.err
set -uo pipefail
source /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/env.sh

PROMPT="insert both pegs into the sockets"

# Task-unique node-local paths (venv + dataset output), cleaned at exit.
export UV_PROJECT_ENVIRONMENT="/scratch/$USER/venv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOCAL_OUT="/scratch/$USER/di_out_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_LEROBOT_HOME="$LOCAL_OUT"          # convert on local NVMe, NOT nas6
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT" "$LOCAL_OUT"' EXIT
cd "$PROJECT_DIR"

mapfile -t DIRS < "$PROJECT_DIR/sim_double_insert_dirs.txt"
d="${DIRS[$SLURM_ARRAY_TASK_ID]}"
name="${d%_hdf5}"
DEST="$DATA_ROOT/lerobot/$name"

echo "[$(date)] node=$(hostname) task=$SLURM_ARRAY_TASK_ID  $d -> $name  (local: $LOCAL_OUT)"
uv sync --frozen 2>&1 | tail -3

uv run python convert_xarm_sim_double_insert_to_lerobot.py \
    --raw-dir "$DATA_ROOT/$d" --repo-id "$name" --prompt "$PROMPT"
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date)] CONVERT FAILED $name rc=$rc"; exit $rc; fi

echo "[$(date)] rsync $LOCAL_OUT/$name -> $DEST"
mkdir -p "$DEST"
rsync -a --exclude='images' --delete "$LOCAL_OUT/$name/" "$DEST/"
rc=$?
echo "[$(date)] DONE $name rc=$rc  size=$(du -sh "$DEST" 2>/dev/null | cut -f1)"
exit $rc
