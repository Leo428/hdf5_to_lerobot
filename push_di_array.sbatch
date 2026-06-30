#!/usr/bin/env bash
# Phase 2: push each converted sim double-insert LeRobot dataset (on NFS) to a PUBLIC HF dataset
# repo huzheyuan/<name>. One array task per dataset. hf upload is idempotent (re-runs skip unchanged
# files), and creates the repo PUBLIC by default (no --private). Runs on cpu nodes (NFS + internet).
#   sbatch ~/hdf5_to_lerobot/push_di_array.sbatch
#SBATCH --job-name=di_push
#SBATCH --partition=cpu
#SBATCH --array=0-26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/di_push_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/di_push_%A_%a.err
set -uo pipefail
DATA_ROOT=/data/group_data/rl/dexterous_robot_data
HF=~/miniforge3/bin/hf
export HF_TOKEN=$(tr -d '[:space:]' < ~/hf_key_2026.txt)   # never printed
unset HF_HUB_OFFLINE 2>/dev/null || true

mapfile -t DIRS < "$DATA_ROOT/hdf5_to_lerobot/sim_double_insert_dirs_bysize.txt"
d="${DIRS[$SLURM_ARRAY_TASK_ID]}"
name="${d%_hdf5}"
SRC="$DATA_ROOT/lerobot/$name"

echo "[$(date)] node=$(hostname) push $name -> huzheyuan/$name  (src $SRC)"
$HF upload "huzheyuan/$name" "$SRC" . --repo-type=dataset
rc=$?
echo "[$(date)] DONE $name rc=$rc"
exit $rc
