# Converting real xArm HDF5 datasets to LeRobot on Babel

A runbook for turning the **old real dual-xArm `episode_*.hdf5` datasets** (burger / hang / lid /
… ) into LeRobot v2 datasets for openpi training, on the **Babel HPC cluster**. Written after
converting the full `hang` family (25 dirs, 2028 episodes, 59 GB) on 2026-06-17.

The runnable project lives on Babel at:

```
/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/
```

> This is a **standalone uv project on Babel**, NOT part of the openpi repo (it only needs
> `lerobot` + a few deps, not all of openpi). This doc is the version-controlled copy; keep the
> two in sync if you edit the scripts.

---

## 1. TL;DR — run a new family in 5 steps

```bash
# (on Babel; everything that touches /data must run on a compute node — see §4)
P=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot

# 0. one-time per machine: warm the uv cache (builds lerobot+torch into the NFS cache)
srun --partition=debug --cpus-per-task=8 --mem=16G --time=01:00:00 bash $P/warm_cache.sh

# 1. build the dir list for the family, smallest-first (see §6), into $P/<family>_dirs.txt
# 2. smoke test 2 eps/dir (throwaway on /scratch) — catches per-folder quirks
sbatch --partition=cpu --cpus-per-task=16 --mem=48G --time=02:00:00 \
       --output=$HOME/smoke_%j.log $HOME/hdf5_to_lerobot/smoke_all.sh   # edit list inside first

# 3. launch the two arrays (cpu + general/L40S) over disjoint index ranges (§4, §7)
sbatch $HOME/hdf5_to_lerobot/convert_array.sbatch       # cpu, indices 0..N1
sbatch $HOME/hdf5_to_lerobot/convert_array_gpu.sbatch   # general/L40S, indices N1+1..N2

# 4. monitor; re-run any failed index (§8); verify (§9)
```

Datasets land at `…/dexterous_robot_data/lerobot/<round_name>/`.

---

## 2. The data format (and why it needs a bespoke converter)

Each `episode_N.hdf5` is a **hybrid** that neither existing xArm converter
(`convert_xarm_data_to_lerobot.py` = real-npz, `convert_xarm_sim_data_to_lerobot.py` = sim-hdf5)
handles as-is. Structure (`h5ls -r episode_0.hdf5`):

```
/actions/global_action      {T,14}   world-frame command (pos-delta + rot-delta + gripper per arm)
/actions/relative_action    {T,14}   EE/body-frame command
/obses/state/{left,right}/
    target_tcp_pose          {T,7}    pos(3) + quaternion(4, scalar-first wxyz)  <- the action anchor
    tcp_pose                 {T,7}    current pose (pos+quat)
    gripper_pos              {T,1}    ACHIEVED gripper (real range ~80-840)
    target_gripper_pos       {T,1}    DEAD (constant) -- do not use
    joint_qpos/qvel/torque, ee_pose, ...                                 (unused)
/obses/images/{left,right}/{top,wrist}  {T, maxlen} uint8   JPEG bytes, zero-padded to fixed width
/obses/state/timestamp      {T}
/metadata/task              scalar   UNRELIABLE global label (see below)
/metadata/interventions     {K}      step indices where a human took over (DAgger rounds; absent on clean data)
/metadata/horizon, og_filename
/rewards, /dones, /truncateds                                            (unused)
```

### The 6 gotchas the converter handles

1. **B1 off-by-one (obs/action timestep offset).** The collection loop appends `obs` *after*
   `env.step(action)`, so `obs[i]` is the state `action[i]` produced. **Verified empirically on this
   data**: `target_tcp_pose[i] - target_tcp_pose[i-1] == global_action[i]` to ~1e-6 (NOT
   `relative_action`, which is the EE/body frame). Therefore the command issued *from* `obs[i]`
   reaches target `i+1`, i.e. **`action[i] = state_pose[i+1]`** (last frame holds → zero delta).
   Implemented by `shift_pose_b1`.

2. **Quaternion → 6D rotation rep.** Poses are 7-d `pos(3) + quat(4)`, **scalar-first wxyz**
   (same as the real npz data — its converter's `quats_wxyz_to_6d` uses
   `Rotation.from_quat(..., scalar_first=True)`). We convert to the network-ready 6D rep (first two
   columns of R) → the model's **20-d state/action** = per arm `[pos(3), 6D(6), gripper(1)]`, L then R.

3. **Dead gripper → reconstruct.** `target_gripper_pos` is constant (env never writes it back), so we
   replay the env formula (D6) on `action[i+1]`'s gripper dim: `dg = action[grip]`; if `|dg| > 0.05`
   command `clip(gripper_pos - dg*80, 80, 840)`, else sentinel `0.0` == hold. State stores the
   **achieved** `gripper_pos`; action stores the **reconstructed absolute target**. Real range
   ~80–840 (NOT the sim's [0,1]).

4. **Cameras: BGR, decoded as RGB; drop one.** 4 cams (`left/top`, `left/wrist`, `right/top`,
   `right/wrist`), 360×640. We use **`right/top` as the base view and DROP `left/top`** → 3 cams
   (`base`, `left_wrist`, `right_wrist`). The frames are BGR (RealSense/ZED), but `cv2.imencode`
   consumed that convention, so **PIL decodes them as true RGB — no channel swap**. Resized 224×224
   (stretched, no aspect preservation), stored as `dtype: video`.

5. **`metadata/task` is UNRELIABLE.** Every dataset — even lid — reports
   `real_xarms_shirt_hang_variations`. It's a stale global label. **Derive the prompt from the folder
   name / pass it explicitly via `--prompt`.** Do NOT read `metadata/task`.

6. **Interventions → per-frame bool.** `metadata/interventions` is the list of human-takeover step
   indices (DAgger correction rounds; absent on clean/`full_success` rounds). The converter adds a
   per-frame boolean feature **`is_intervention`** (True where `i ∈ interventions`, all-False where
   the key is absent). It labels timestep `i` and is **not** B1-shifted. One HDF5 episode → one
   LeRobot episode (no subtask annotations exist for this data).

---

## 3. The converter

`convert_xarm_hdf5_to_lerobot.py` — transplants the proven math from the real-npz converter
(`quats_wxyz_to_6d`, `reconstruct_gripper_target`, `shift_pose_b1`, the BGR-via-PIL decode,
right-top base) and changes only the input (HDF5, read whole arrays vectorized; `global_action` is
the world-frame anchor) and output (one ep per file, `is_intervention` feature, fixed `--prompt`).

```bash
HF_LEROBOT_HOME=<out_root> uv run python convert_xarm_hdf5_to_lerobot.py \
    --raw-dir <.../real_hang_round1_0612_hdf5> \
    --repo-id real_hang_round1_0612 \
    --prompt "put shirt on hanger and place hanger on rack" \
    [--max-episodes 2]          # smoke
```

- Output dataset → `$HF_LEROBOT_HOME/<repo-id>` (bare repo-id with no `namespace/` is fine here).
- `FPS=60`, video codec = **SVT-AV1** (LeRobot default; see §5 for the perf implication).
- `lerobot` pinned to openpi's rev `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` (in `pyproject.toml`).
- Self-check prints on the first episode of each dir (B1 per-step `|dpos|`, 6D round-trip, gripper recon, interv %).

---

## 4. Babel environment & constraints (the important part)

| Thing | Reality on Babel | What we do about it |
|---|---|---|
| **NFS data** (`/data/group_data/rl/...`) | **Not visible from the login node.** Only from compute/debug nodes. | All file ops on data run via `srun`/`sbatch`. From a non-interactive ssh use `srun ... bash -c '<cmd>'` (not `--pty`). |
| **`cpu` partition** | **`sbatch`-only** (interactive `srun` errors: "Only sbatch jobs allowed"). | Use `--partition=debug` for interactive `srun` (has GPUs, 12 h, fine for inspection). Real conversion = `sbatch` on `cpu`. |
| **HOME** (`nas5`) | Was **98% full**; importing big venvs over NFS is slow. | Keep ALL caches off home. Put the **venv on node-local `/scratch`** (fast imports). `conda clean -a` frees ~15 GB if home is tight. |
| **Concurrency** | **`cpu_qos` MaxJobsPU = 10** — hard cap of 10 running jobs/user (pending reason `QOSMaxJobsPerUserLimit`). NOT a CPU/node limit. | Tap a 2nd QOS for ~8 more slots (next row). |
| **2nd QOS trick** | `normal` qos (general partition) also allows 10 jobs, but caps **GPUs at 8/user**. | Run a 2nd array on `--partition=general --gres=gpu:L40S:1` with `CUDA_VISIBLE_DEVICES=""` (a **dummy GPU we never use**, just to be eligible) → ~8 more concurrent. ~18 total. L40S are easy to get. |
| **Intermediate I/O** | LeRobot writes **every frame as an individual PNG to disk**, then ffmpeg-encodes to mp4 and deletes it → millions of tiny files. | **Convert into node-local `/scratch`, then `rsync` only the final dataset to NFS.** Never point `HF_LEROBOT_HOME` at NFS during conversion — the PNG churn cripples the shared `nas6` server. |
| **Compute-node internet** | Compute nodes **have internet** + see `/data` + see HOME. | `uv sync` works on compute nodes; warm the cache once so parallel tasks don't all download torch at once. |
| **uv** | `uv 0.9.x` is at `~/.local/bin/uv`. | `env.sh` puts it on PATH. |

`env.sh` (sourced by every script) sets the canonical paths:

```bash
DATA_ROOT=/data/group_data/rl/dexterous_robot_data
PROJECT_DIR=$DATA_ROOT/hdf5_to_lerobot
HF_LEROBOT_HOME=$DATA_ROOT/lerobot          # final output root (overridden to /scratch in jobs)
UV_CACHE_DIR=/data/group_data/rl/zheyuanh/.uv_cache   # persistent, off home
HF_HOME=/data/group_data/rl/zheyuanh/.hf_home ; HF_HUB_OFFLINE=1
UV_PROJECT_ENVIRONMENT=/scratch/$USER/...    # node-local venv (per-task unique in arrays)
```

---

## 5. Performance notes

- **Throughput is encode-bound.** LeRobot hard-codes **SVT-AV1 @ crf30** (`encode_video_frames`
  vcodec defaults; `create()` only exposes the *decode* `video_backend`). At 224² it's ~**30–65 s
  per episode at 32 CPUs** (depends on episode length). For the `hang` family: most rounds finished
  in minutes–~1 h; the two `round0` giants (240 / 360 eps) were ~4–5 h.
- **Per-task CPUs:** 32 is a good default. SVT-AV1 at 224² plateaus ~16–32 threads, so 48/64 give
  diminishing returns; concurrency (more jobs) matters more — but that's capped at 10+8 (§4).
- **Want it faster?** Switch the codec to `libx264` (a 1-line monkeypatch of `encode_video_frames`'s
  `vcodec`) — ~5–10× faster encode, slightly larger files, decodes identically for training. We kept
  AV1 for the hang run to match the existing datasets' format; revisit if speed matters more.
- **Output size:** ~`107 MB` per ~13 k frames → the full hang family (7.0 M frames) = **59 GB**.
  Budget ~8 KB/frame (3 cams, AV1). Check `nas6` free space first (it's a shared 3 TB fs).

---

## 6. Building the dir list (smallest-first)

Order matters under the 10-job cap / when there's contention: **smallest-first** so the many quick
rounds finish first and the few giants trail (they're allowed to lag). Scan the family, then sort by
frame count ascending. The hang scan helper printed `#episodes / total bytes / avg horizon /
intervention %` per dir — reuse it. Put the result (one `*_hdf5` dir name per line) in
`<family>_dirs.txt`; the array indexes into it by `$SLURM_ARRAY_TASK_ID`.

---

## 7. Launching: two disjoint arrays

To get ~18 concurrent, split the index range across two arrays over the **same** dir list (disjoint
ranges → no races on output):

- `convert_array.sbatch` — `--partition=cpu`, `--array=0-N1`, smaller rounds.
- `convert_array_gpu.sbatch` — `--partition=general --gres=gpu:L40S:1`, `--array=N1+1-N2`,
  `CUDA_VISIBLE_DEVICES=""`, larger rounds (incl. the giants).

Both: `--cpus-per-task=32 --mem=96G --time=24:00:00`, venv + output on per-task `/scratch`, then
`rsync -a --exclude='images' --delete $LOCAL_OUT/<name>/ $DATA_ROOT/lerobot/<name>/`. Split ~13/12
for 25 dirs (general side effectively runs ~8 at once due to the 8-GPU cap).

---

## 8. Monitoring & re-running failures

```bash
# per-task states (works from the login node — sacct hits the accounting DB, no NFS):
sacct -j <cpuJobId>,<genJobId> -n --format=JobID%20,JobName%12,State,Elapsed | grep -E '_[0-9]+ '

# re-run a single failed index (CLI --array overrides the directive; cpu partition is usually free
# once the small rounds finish):
sbatch --array=<N> --partition=cpu $HOME/hdf5_to_lerobot/convert_array.sbatch
```

**Known transient:** a task that shows `FAILED` with `Elapsed 00:00:01` **and no log files written**
is a **node/launch failure** (Slurm couldn't start it on its node) — infra, not a bug. Just re-run
that index. (Happened once to `round0_robyn` in the hang run; the re-run was clean.)

Array log files go to `$PROJECT_DIR/slurm_logs/` (on NFS → read them via a `debug` `srun`). Submit
arrays from the **HOME copy** of the sbatch so the login node can read the script; outputs/paths
inside resolve on the compute node.

---

## 9. Verifying the result

```bash
# via a debug srun, over $DATA_ROOT/lerobot/<family>_*/:
#   - every dir has meta/info.json   (valid LeRobot dataset)
#   - total_episodes matches the source dir's episode count (no drops)
#   - the is_intervention feature is present
for d in .../lerobot/real_<family>_*/; do
  info=$d/meta/info.json
  grep -o '"total_episodes": *[0-9]*' $info; grep -c is_intervention $info
done
```

For the hang family: 25/25 valid, episode counts matched the source exactly (2028 total), all had
`is_intervention`.

---

## 10. Adapting to a new family / dataset

The converter is generic for this HDF5 layout. To do `burger`, `lid`, or any new `real_*` family:

1. **Prompt:** decide the task language string (the analog of "put shirt on hanger…"). The folder
   name is the reliable task indicator, not `metadata/task`.
2. **Dir list:** scan the family, build `<family>_dirs.txt` smallest-first (§6).
3. **Edit the two sbatch files** to point at the new list and split the index range; set `--prompt`
   in the converter call (or change the `DEFAULT_PROMPT`).
4. Smoke 2 eps/dir → launch both arrays → monitor → verify (§§2,7,8,9).

**Different schema?** If a new dataset's `h5ls -r` differs (missing `target_tcp_pose`, different cam
names, 6D instead of quat, etc.), check it against §2 before trusting the converter — older/newer
collection code has varied. The `metadata/interventions` key is simply absent on clean rounds (→
all-False `is_intervention`), which is expected and fine.

---

## File manifest (`hdf5_to_lerobot/`)

| File | Purpose |
|---|---|
| `convert_xarm_hdf5_to_lerobot.py` | The converter (§3). |
| `pyproject.toml` / `uv.lock` | uv project; `lerobot` pinned to openpi rev `0cf8648`. |
| `env.sh` | Canonical paths + Babel storage rules (§4). |
| `warm_cache.sh` | One-time: populate uv cache + create `uv.lock` (run on a compute node). |
| `smoke_all.sh` | Convert 2 eps/dir to throwaway `/scratch` — per-folder sanity. |
| `convert_array.sbatch` | cpu-partition array (smaller half). |
| `convert_array_gpu.sbatch` | general/L40S array (dummy GPU; larger half). |
| `<family>_dirs.txt` | Smallest-first dir list the arrays index into. |
| `slurm_logs/` | Per-task array logs. |

Reference converters in the openpi repo: `examples/xarm/convert/convert_xarm_data_to_lerobot.py`
(real npz) and `convert_xarm_sim_data_to_lerobot.py` (sim hdf5).
