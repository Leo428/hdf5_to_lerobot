#!/usr/bin/env bash
# Common environment for the xArm HDF5 -> LeRobot conversion on Babel. `source` me.
#
# Storage rules on Babel:
#   - HOME (nas5) is 98% full -> keep ALL caches off it.
#   - data-NFS (nas6) holds the raw data, the project, the output datasets, and the uv cache
#     (persistent, visible to every compute node; compute nodes have internet to populate it).
#   - /scratch is node-local NVMe -> the venv lives here so Python imports are off local disk,
#     not the network (per the "don't import code over NFS" rule).
export PATH="$HOME/.local/bin:$PATH"                         # uv lives here

export DATA_ROOT=/data/group_data/rl/dexterous_robot_data
export PROJECT_DIR="$DATA_ROOT/hdf5_to_lerobot"
export HF_LEROBOT_HOME="$DATA_ROOT/lerobot"                  # output LeRobot datasets land here

export UV_CACHE_DIR=/data/group_data/rl/zheyuanh/.uv_cache   # persistent, off home, internet-warmed
export HF_HOME=/data/group_data/rl/zheyuanh/.hf_home         # keep HF hub cache off home
export HF_HUB_OFFLINE=1                                      # never reach the Hub during conversion
export XDG_CACHE_HOME="/scratch/$USER/.cache"                # node-local scratch for misc caches

# Per-node local NVMe venv (fast imports). Launchers override this with a task-unique path to
# avoid two concurrent array tasks racing on the same venv dir.
: "${UV_PROJECT_ENVIRONMENT:=/scratch/$USER/hdf5_to_lerobot_venv}"
export UV_PROJECT_ENVIRONMENT
mkdir -p "$XDG_CACHE_HOME" 2>/dev/null || true
