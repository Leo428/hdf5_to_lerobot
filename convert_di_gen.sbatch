#!/usr/bin/env bash
# GENERAL+L40S half of the split conversion: the 18 LARGEST datasets (indices 0-17 of the
# largest-first sim_double_insert_dirs_bysize.txt). Requests a dummy L40S GPU purely to be eligible
# for the general partition's ~8 nodes/user (CUDA_VISIBLE_DEVICES="" -> the CPU-only AV1 encode never
# touches it). Disjoint indices vs convert_di_cpu.sbatch (18-26) -> no output races.
#   sbatch ~/hdf5_to_lerobot/convert_di_gen.sbatch
#SBATCH --job-name=di_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/di_gen_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/di_gen_%A_%a.err
set -uo pipefail
source /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/env.sh
export CUDA_VISIBLE_DEVICES=""          # dummy GPU only for scheduling; AV1 encode is CPU-only
PROMPT="insert both pegs into the sockets"
LIST="$PROJECT_DIR/sim_double_insert_dirs_bysize.txt"

export UV_PROJECT_ENVIRONMENT="/scratch/$USER/venv_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOCAL_OUT="/scratch/$USER/di_out_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export HF_LEROBOT_HOME="$LOCAL_OUT"
trap 'rm -rf "$UV_PROJECT_ENVIRONMENT" "$LOCAL_OUT"' EXIT
cd "$PROJECT_DIR"

mapfile -t DIRS < "$LIST"
d="${DIRS[$SLURM_ARRAY_TASK_ID]}"
name="${d%_hdf5}"
DEST="$DATA_ROOT/lerobot/$name"

echo "[$(date)] node=$(hostname) task=$SLURM_ARRAY_TASK_ID  $d -> $name"
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
