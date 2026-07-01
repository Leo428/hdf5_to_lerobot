#!/usr/bin/env bash
# Upload converted lid/burger LeRobot datasets to PUBLIC HF, ONE REPO PER DATASET (LeRobot-native),
# to BOTH huzheyuan/<name> and CMU-AIRe/<name> (2nd account dedups via HF cross-repo LFS, ~free).
# Each array task = one dataset dir ($DATA_ROOT/lerobot/<name>) -> repo root. Idempotent/resumable.
# --mem=128G from the start (hf upload RSS balloons; low mem -> cgroup stall).
#   sbatch --export=ALL,FAMILY=lid    --array=0-18%8 ~/hdf5_to_lerobot/push_lerobot_array.sbatch
#   sbatch --export=ALL,FAMILY=burger --array=0-13%8 ~/hdf5_to_lerobot/push_lerobot_array.sbatch
#SBATCH --job-name=lblero
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/lblero_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/lblero_%A_%a.err
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
DIRLIST="$DR/hdf5_to_lerobot/${FAMILY}_dirs.txt"
LOGDIR="$DR/hdf5_to_lerobot/slurm_logs/leropush_real_${FAMILY}"
mkdir -p "$LOGDIR"
mapfile -t DIRS < "$DIRLIST"
d="${DIRS[$SLURM_ARRAY_TASK_ID]:-}"
[ -z "$d" ] && { echo "no dir for index $SLURM_ARRAY_TASK_ID in $DIRLIST"; exit 0; }
name="${d%_hdf5}"
SRC="$DR/lerobot/$name"
if [ ! -d "$SRC" ]; then echo "MISSING dataset $SRC (conversion failed?) -- skipping"; exit 3; fi
echo "[$(date)] node=$(hostname) task=$SLURM_ARRAY_TASK_ID FAMILY=$FAMILY dataset=$name src=$SRC size=$(du -sh "$SRC" 2>/dev/null | cut -f1)"
rc=0
for ns in huzheyuan CMU-AIRe; do
  if "$HF" upload "$ns/$name" "$SRC" . --repo-type=dataset --commit-message="add $name (LeRobot v2)" > "$LOGDIR/${ns}__${name}.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK   $ns/$name"
  else
    echo "[$(date +%H:%M:%S)] FAIL $ns/$name (see leropush_real_${FAMILY}/${ns}__${name}.log)"; rc=1
  fi
done
echo "[$(date)] DONE task=$SLURM_ARRAY_TASK_ID $name rc=$rc"
exit $rc
