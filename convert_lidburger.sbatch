#!/usr/bin/env bash
# Convert the real lid / burger HDF5 datasets to LeRobot as a SLURM array. Parameterized by FAMILY
# (passed via --export=ALL,FAMILY=lid|burger), which selects the dir list + task prompt. Each task
# converts ONE dir into NODE-LOCAL /scratch (LeRobot's per-frame PNG churn stays off shared nas6),
# then rsyncs the final compact dataset to $DATA_ROOT/lerobot/<name>. HF push is a SEPARATE step.
# is_intervention: round0 dirs -> all-True (auto via repo_id), full_success -> all-False (no key),
# roundN -> metadata/interventions. All 3 verified against the data before launch.
#   sbatch --export=ALL,FAMILY=lid    --array=0-18%10 ~/hdf5_to_lerobot/convert_lidburger.sbatch
#   sbatch --export=ALL,FAMILY=burger --array=0-13%10 ~/hdf5_to_lerobot/convert_lidburger.sbatch
#SBATCH --job-name=lbconv
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/lbconv_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/lbconv_%A_%a.err
set -uo pipefail
source /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/env.sh

: "${FAMILY:?set FAMILY=lid or FAMILY=burger via --export=ALL,FAMILY=...}"
case "$FAMILY" in
  lid)    DIRLIST="$PROJECT_DIR/lid_dirs.txt";    PROMPT="put the lid on the container and seal it" ;;
  burger) DIRLIST="$PROJECT_DIR/burger_dirs.txt"; PROMPT="pack the burger into the takeout box and close the lid" ;;
  *) echo "bad FAMILY='$FAMILY' (want lid|burger)"; exit 2 ;;
esac

# Task-unique node-local paths (venv + dataset output), cleaned at exit.
export UV_PROJECT_ENVIRONMENT="/scratch/$USER/venv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOCAL_OUT="/scratch/$USER/lb_out_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_LEROBOT_HOME="$LOCAL_OUT"          # convert on local NVMe, NOT nas6
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT" "$LOCAL_OUT"' EXIT
cd "$PROJECT_DIR"

mapfile -t DIRS < "$DIRLIST"
d="${DIRS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "${d:-}" ]; then echo "no dir at idx $SLURM_ARRAY_TASK_ID in $DIRLIST"; exit 0; fi
name="${d%_hdf5}"
DEST="$DATA_ROOT/lerobot/$name"

echo "[$(date)] node=$(hostname) task=$SLURM_ARRAY_TASK_ID FAMILY=$FAMILY  $d -> $name  (local: $LOCAL_OUT)"
echo "         PROMPT='$PROMPT'"
uv sync --frozen 2>&1 | tail -3

uv run python convert_xarm_hdf5_to_lerobot.py --raw-dir "$DATA_ROOT/$d" --repo-id "$name" --prompt "$PROMPT"
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date)] CONVERT FAILED $name rc=$rc"; exit $rc; fi

echo "[$(date)] rsync $LOCAL_OUT/$name -> $DEST"
mkdir -p "$DEST"
rsync -a --exclude='images' --delete "$LOCAL_OUT/$name/" "$DEST/"
rc=$?
echo "[$(date)] DONE $name rc=$rc  size=$(du -sh "$DEST" 2>/dev/null | cut -f1)"
exit $rc
