#!/usr/bin/env bash
# Smoke test: convert 2 episodes from EVERY hang dir into throwaway datasets on /scratch
# (no NFS pollution), validating the full pipeline (load + 6D + gripper + video encode) on each
# folder so per-folder quirks surface before we spend real compute. Run on a compute node:
#   srun --partition=cpu --cpus-per-task=16 --mem=48G --time=02:00:00 \
#        bash /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/smoke_all.sh
set -uo pipefail
source /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/env.sh
export UV_PROJECT_ENVIRONMENT="/scratch/$USER/hdf5_to_lerobot_smokevenv"
export HF_LEROBOT_HOME="/scratch/$USER/hdf5_to_lerobot_smoke"     # throwaway; keep NFS clean
rm -rf "$HF_LEROBOT_HOME"; mkdir -p "$HF_LEROBOT_HOME"
cd "$PROJECT_DIR"

echo "Syncing venv on $(hostname):$UV_PROJECT_ENVIRONMENT ..."
uv sync --frozen 2>&1 | tail -3

pass=0; fail=0; failed=()
while read -r d; do
  [ -z "$d" ] && continue
  name="${d%_hdf5}"
  echo; echo "================ SMOKE $name ================"
  if uv run python convert_xarm_hdf5_to_lerobot.py \
       --raw-dir "$DATA_ROOT/$d" --repo-id "smoke_$name" --max-episodes 2; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1)); failed+=("$name")
  fi
done < "$PROJECT_DIR/hang_dirs.txt"

echo; echo "===== SMOKE RESULT: $pass passed, $fail failed ====="
if [ "$fail" -gt 0 ]; then printf '  FAILED: %s\n' "${failed[@]}"; else echo "  all folders OK"; fi
rm -rf "$UV_PROJECT_ENVIRONMENT"
