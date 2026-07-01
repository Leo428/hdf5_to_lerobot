#!/usr/bin/env bash
# Back up a raw HDF5 family to PUBLIC HF, one-repo-per-task w/ per-round subfolders, to BOTH
# huzheyuan/ and CMU-AIRe/ (2nd account dedups via HF cross-repo LFS, ~free). Idempotent.
#   sbatch ~/hdf5_to_lerobot/push_raw.sbatch real_hang
#   sbatch ~/hdf5_to_lerobot/push_raw.sbatch sim_double_insert
#SBATCH --job-name=rawpush
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/rawpush_%x_%j.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/rawpush_%x_%j.err
set -uo pipefail
FAMILY="${1:?usage: sbatch push_raw.sbatch <family>}"
export HF_TOKEN=$(tr -d '[:space:]' < ~/hf_key_2026.txt)   # never printed
unset HF_HUB_OFFLINE 2>/dev/null || true
export HF_HOME=/scratch/$USER/hf_${SLURM_JOB_ID:-$$}
export HF_XET_CACHE=$HF_HOME/xet
mkdir -p "$HF_XET_CACHE"
trap 'rm -rf "$HF_HOME"' EXIT
HF=~/miniforge3/bin/hf
DR=/data/group_data/rl/dexterous_robot_data
REPO="${FAMILY}_hdf5"
MAXJOBS=${MAXJOBS:-3}
LOGDIR="$DR/hdf5_to_lerobot/slurm_logs/rawpush_${FAMILY}"
mkdir -p "$LOGDIR"
mapfile -t DIRS < <(ls -d $DR/${FAMILY}_*_hdf5 2>/dev/null | xargs -n1 basename)
echo "[$(date)] node=$(hostname) family=$FAMILY repo=$REPO rounds=${#DIRS[@]} maxjobs=$MAXJOBS"
upload_round() {
  local d="$1"
  for ns in huzheyuan CMU-AIRe; do
    if "$HF" upload "$ns/$REPO" "$DR/$d" "$d" --repo-type=dataset --commit-message="add $d" > "$LOGDIR/${ns}__${d}.log" 2>&1; then
      echo "[$(date +%H:%M:%S)] OK   $ns/$REPO/$d"
    else
      echo "[$(date +%H:%M:%S)] FAIL $ns/$REPO/$d (see $LOGDIR/${ns}__${d}.log)"
    fi
  done
}
export -f upload_round; export HF DR REPO LOGDIR
printf '%s\n' "${DIRS[@]}" | xargs -P "$MAXJOBS" -I{} bash -c 'upload_round "$@"' _ {}
echo "[$(date)] ALL DONE family=$FAMILY"
