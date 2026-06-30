#!/usr/bin/env bash
# Per-round conversion as a SLURM array, ordered SMALLEST-FIRST (hang_dirs_by_size.txt) so the quick
# rounds get scheduled first and the bulk of the datasets finish ASAP; the few giant rounds (e.g.
# round0_jasmine) run last / whenever capacity frees -- they're allowed to take longer.
#
# Each task converts into NODE-LOCAL /scratch, then rsyncs ONLY the final compact dataset to NFS.
# Why: LeRobot writes every frame as an individual PNG to disk, then ffmpeg-encodes it to mp4 and
# deletes it -- millions of tiny files. Keeping that churn on local NVMe (not the shared nas6 NFS)
# is the big speed + be-nice-to-other-users win; NFS then sees one sequential rsync per round.
#   sbatch /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/convert_array.sbatch
#SBATCH --job-name=hangconv
#SBATCH --partition=cpu
#SBATCH --array=0-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/hangconv_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/hangconv_%A_%a.err
set -uo pipefail
source /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/env.sh

# Task-unique node-local paths (venv + dataset output), cleaned at exit.
export UV_PROJECT_ENVIRONMENT="/scratch/$USER/venv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOCAL_OUT="/scratch/$USER/lerobot_out_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_LEROBOT_HOME="$LOCAL_OUT"          # convert on local NVMe, NOT nas6
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT" "$LOCAL_OUT"' EXIT
cd "$PROJECT_DIR"

mapfile -t DIRS < "$PROJECT_DIR/hang_dirs_by_size.txt"
d="${DIRS[$SLURM_ARRAY_TASK_ID]}"
name="${d%_hdf5}"
DEST="$DATA_ROOT/lerobot/$name"

echo "[$(date)] node=$(hostname) task=$SLURM_ARRAY_TASK_ID  $d -> $name  (local: $LOCAL_OUT)"
uv sync --frozen 2>&1 | tail -3

uv run python convert_xarm_hdf5_to_lerobot.py --raw-dir "$DATA_ROOT/$d" --repo-id "$name"
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date)] CONVERT FAILED $name rc=$rc"; exit $rc; fi

echo "[$(date)] rsync $LOCAL_OUT/$name -> $DEST"
mkdir -p "$DEST"
rsync -a --exclude='images' --delete "$LOCAL_OUT/$name/" "$DEST/"
rc=$?
echo "[$(date)] DONE $name rc=$rc  size=$(du -sh "$DEST" 2>/dev/null | cut -f1)"
exit $rc
