#!/usr/bin/env bash
# Populate the uv cache (data-NFS) + create uv.lock, ONCE, on a compute node (needs internet +
# data-NFS, both of which compute nodes have; the login node cannot see data-NFS).
# Usage:  srun --partition=debug --cpus-per-task=8 --mem=16G --time=01:00:00 \
#              bash /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/warm_cache.sh
set -euo pipefail
source /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/env.sh
cd "$PROJECT_DIR"
export UV_PROJECT_ENVIRONMENT="/scratch/$USER/hdf5_to_lerobot_warmvenv"
rm -rf "$UV_PROJECT_ENVIRONMENT"
echo "Warming uv cache -> $UV_CACHE_DIR (this resolves + downloads lerobot@0cf8648 + deps) ..."
uv sync
echo
echo "ffmpeg check (needed for LeRobot video encoding):"
uv run python -c "import importlib.util as u; print('av (PyAV):', u.find_spec('av') is not None)"
command -v ffmpeg >/dev/null && echo "system ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)" || echo "system ffmpeg: NOT on PATH"
echo
echo "Warm done. uv.lock:"; ls -l "$PROJECT_DIR/uv.lock"
rm -rf "$UV_PROJECT_ENVIRONMENT"
