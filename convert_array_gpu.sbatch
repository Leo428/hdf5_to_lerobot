#!/usr/bin/env bash
# SECOND array, on the general partition (normal qos), to add ~8 more concurrent slots on top of
# cpu_qos's 10. We request a dummy L40S GPU we never touch (CUDA_VISIBLE_DEVICES="") purely to be
# eligible for the general partition. Covers indices 13-24 of hang_dirs_by_size.txt (the LARGER
# rounds, incl. round0_jasmine); the cpu-partition array covers 0-12 -> disjoint, no races.
#   sbatch /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/convert_array_gpu.sbatch
#SBATCH --job-name=hangconv_g
#SBATCH --partition=general
#SBATCH --array=13-24
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/hangconv_gpu_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/hangconv_gpu_%A_%a.err
set -uo pipefail
source /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/env.sh
export CUDA_VISIBLE_DEVICES=""               # allocated a GPU for qos eligibility; never use it

export UV_PROJECT_ENVIRONMENT="/scratch/$USER/venv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOCAL_OUT="/scratch/$USER/lerobot_out_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_LEROBOT_HOME="$LOCAL_OUT"
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT" "$LOCAL_OUT"' EXIT
cd "$PROJECT_DIR"

mapfile -t DIRS < "$PROJECT_DIR/hang_dirs_by_size.txt"
d="${DIRS[$SLURM_ARRAY_TASK_ID]}"
name="${d%_hdf5}"
DEST="$DATA_ROOT/lerobot/$name"

echo "[$(date)] node=$(hostname) task=$SLURM_ARRAY_TASK_ID  $d -> $name  (general/L40S, local: $LOCAL_OUT)"
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
