# xArm HDF5 → LeRobot conversion (Babel)

Converts the old real dual-xArm `episode_*.hdf5` datasets under
`/data/group_data/rl/dexterous_robot_data/` into LeRobot v2 datasets, **one per round dir**.

- Converter: `convert_xarm_hdf5_to_lerobot.py` (transplants the real-npz converter's pose/gripper/
  B1-offset math; adds a per-frame `is_intervention` bool; one HDF5 episode → one LeRobot episode).
- Prompt (all hang dirs): `"put shirt on hanger and place hanger on rack"`.
- Output: `$HF_LEROBOT_HOME = .../dexterous_robot_data/lerobot/<round_name>` (round dir minus `_hdf5`).

## Babel storage model (see `env.sh`)
- venv → node-local `/scratch` (fast imports, not over NFS).
- uv cache + HF cache → data-NFS (HOME is 98% full).
- All NFS data ops must run on a **compute node** (login can't see data-NFS). Compute nodes have internet.

## Steps
```bash
# 0) one-time: warm the uv cache + create uv.lock (compute node)
srun --partition=debug --cpus-per-task=8 --mem=16G --time=01:00:00 \
     bash /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/warm_cache.sh

# 1) smoke test: 2 episodes from EVERY hang dir (throwaway, on /scratch)
srun --partition=cpu --cpus-per-task=16 --mem=48G --time=02:00:00 \
     bash /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/smoke_all.sh

# 2) full per-round conversion (array, 8 concurrent)
sbatch /data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/convert_array.sbatch
```

`hang_dirs.txt` is the round list (array index → dir). Edit `%8` in `convert_array.sbatch` to
change NFS-friendly concurrency.
