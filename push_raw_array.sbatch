#!/usr/bin/env bash
# Back up a real lid/burger RAW HDF5 family to PUBLIC HF, ONE round dir per array task, to BOTH
# huzheyuan/ and CMU-AIRe/ (repo = real_<fam>_hdf5, per-round subfolders; 2nd account dedups via
# HF cross-repo LFS, ~free). Idempotent/resumable (hf re-hashes, skips blobs already on HF).
# --mem=128G FROM THE START (hf upload balloons RSS on big rounds; low mem -> cgroup stall).
#   sbatch --export=ALL,FAMILY=lid    --array=0-18%8 ~/hdf5_to_lerobot/push_raw_array.sbatch
#   sbatch --export=ALL,FAMILY=burger --array=0-13%8 ~/hdf5_to_lerobot/push_raw_array.sbatch
# (add --partition=general --gres=gpu:L40S:1 on a 2nd disjoint array for +8 nodes.)
#SBATCH --job-name=lbraw
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/lbraw_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/lbraw_%A_%a.err
set -uo pipefail
export CUDA_VISIBLE_DEVICES=""   # dummy GPU var (general-QOS eligibility); harmless on cpu nodes
: "${FAMILY:?set FAMILY=lid or FAMILY=burger via --export=ALL,FAMILY=...}"
case "$FAMILY" in
  lid|burger) : ;;
  *) echo "bad FAMILY='$FAMILY' (want lid|burger)"; exit 2 ;;
esac
export HF_TOKEN=$(tr -d '[:space:]' < ~/hf_key_2026.txt)   # never printed
unset HF_HUB_OFFLINE 2>/dev/null || true
export HF_HOME=/scratch/$USER/hf_${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}
export HF_XET_CACHE=$HF_HOME/xet
mkdir -p "$HF_XET_CACHE"
trap 'rm -rf "$HF_HOME"' EXIT
HF=~/miniforge3/bin/hf
DR=/data/group_data/rl/dexterous_robot_data
REPO="real_${FAMILY}_hdf5"
DIRLIST="$DR/hdf5_to_lerobot/${FAMILY}_dirs.txt"
LOGDIR="$DR/hdf5_to_lerobot/slurm_logs/rawpush_real_${FAMILY}"
mkdir -p "$LOGDIR"
mapfile -t DIRS < "$DIRLIST"
d="${DIRS[$SLURM_ARRAY_TASK_ID]:-}"
[ -z "$d" ] && { echo "no dir for index $SLURM_ARRAY_TASK_ID in $DIRLIST"; exit 0; }
echo "[$(date)] node=$(hostname) task=$SLURM_ARRAY_TASK_ID FAMILY=$FAMILY repo=$REPO dir=$d"
rc=0
for ns in huzheyuan CMU-AIRe; do
  if "$HF" upload "$ns/$REPO" "$DR/$d" "$d" --repo-type=dataset --commit-message="add $d" > "$LOGDIR/${ns}__${d}.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK   $ns/$REPO/$d"
  else
    echo "[$(date +%H:%M:%S)] FAIL $ns/$REPO/$d (see rawpush_real_${FAMILY}/${ns}__${d}.log)"; rc=1
  fi
done
echo "[$(date)] DONE task=$SLURM_ARRAY_TASK_ID $d rc=$rc"
exit $rc
