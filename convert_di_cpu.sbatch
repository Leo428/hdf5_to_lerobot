#!/usr/bin/env bash
# CPU half of the split sim double-insert conversion: the 9 SMALLEST datasets (indices 18-26 of the
# largest-first sim_double_insert_dirs_bysize.txt). Runs alongside convert_di_gen.sbatch (general+L40S,
# the 18 largest, indices 0-17) over DISJOINT index ranges -> no output races.
#   sbatch ~/hdf5_to_lerobot/convert_di_cpu.sbatch
#SBATCH --job-name=di_cpu
#SBATCH --partition=cpu
#SBATCH --array=18-26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/di_cpu_%A_%a.out
#SBATCH --error=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/slurm_logs/di_cpu_%A_%a.err
set -uo pipefail
source /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/env.sh
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
