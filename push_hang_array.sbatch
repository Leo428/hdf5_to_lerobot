#!/usr/bin/env bash
# Multi-node raw-hang upload: ONE round dir per array task -> PUBLIC HF dual-account
# (huzheyuan/ + CMU-AIRe/ real_hang_hdf5, per-round subfolders). Idempotent/resumable
# (hf re-hashes, skips blobs already on HF). Submit as two disjoint arrays for max nodes:
#   sbatch --partition=cpu                       --array=0-12  push_hang_array.sbatch
#   sbatch --partition=general --gres=gpu:L40S:1 --array=13-24 push_hang_array.sbatch
#SBATCH --job-name=hangraw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/hangraw_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/hangraw_%A_%a.err
set -uo pipefail
export CUDA_VISIBLE_DEVICES=""   # dummy GPU (general-QOS eligibility); harmless on cpu nodes
export HF_TOKEN=$(tr -d '[:space:]' < ~/hf_key_2026.txt)   # never printed
unset HF_HUB_OFFLINE 2>/dev/null || true
export HF_HOME=/scratch/$USER/hf_${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}
export HF_XET_CACHE=$HF_HOME/xet
mkdir -p "$HF_XET_CACHE"
trap 'rm -rf "$HF_HOME"' EXIT
HF=~/miniforge3/bin/hf
DR=/data/group_data/rl/dexterous_robot_data
REPO=real_hang_hdf5
LOGDIR="$DR/hdf5_to_lerobot/slurm_logs/rawpush_real_hang"
mkdir -p "$LOGDIR"
mapfile -t DIRS < "$DR/hdf5_to_lerobot/hang_dirs_by_size.txt"
d="${DIRS[$SLURM_ARRAY_TASK_ID]:-}"
[ -z "$d" ] && { echo "no dir for index $SLURM_ARRAY_TASK_ID"; exit 0; }
echo "[$(date)] node=$(hostname) task=$SLURM_ARRAY_TASK_ID dir=$d"
rc=0
for ns in huzheyuan CMU-AIRe; do
  if "$HF" upload "$ns/$REPO" "$DR/$d" "$d" --repo-type=dataset --commit-message="add $d" > "$LOGDIR/${ns}__${d}.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK   $ns/$REPO/$d"
  else
    echo "[$(date +%H:%M:%S)] FAIL $ns/$REPO/$d (see rawpush_real_hang/${ns}__${d}.log)"; rc=1
  fi
done
echo "[$(date)] DONE task=$SLURM_ARRAY_TASK_ID $d rc=$rc"
exit $rc
