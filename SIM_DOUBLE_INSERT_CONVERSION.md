# Converting sim dual-xArm *double-insert* HDF5 datasets to LeRobot (+ HF) on Babel

A runbook for turning the **old sim `double_insert` `episode_*.hdf5` datasets** into LeRobot v2
datasets and publishing them to the HuggingFace Hub, on the **Babel HPC cluster**. Written after
converting + pushing the full set (27 datasets, 1419 episodes) on 2026-06-30.

Sibling runbooks: `BABEL_HDF5_CONVERSION.md` (real-xArm HDF5) and `SIM_PACKING_CONVERSION.md`
(sim packing). Converter: `convert_xarm_sim_double_insert_to_lerobot.py`. The runnable project lives
on Babel at `/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot/` (and the login-accessible
copy `~/hdf5_to_lerobot/`); see §6 for why there are two.

---

## 1. TL;DR

```bash
# on Babel. data + the 27 raw dirs live under $DATA_ROOT; outputs land in $DATA_ROOT/lerobot/<name>.
P=/data/group_data/rl/dexterous_robot_data/hdf5_to_lerobot

# 1. (one-time) warm the uv cache:  srun --partition=debug -c8 --mem=16G -t1:00:00 bash $P/warm_cache.sh
# 2. smallest-first / largest-first dir list -> $P/sim_double_insert_dirs_bysize.txt
# 3. convert, split across cpu + general(+dummy L40S) over DISJOINT index ranges (§4):
sbatch ~/hdf5_to_lerobot/convert_di_gen.sbatch   # general+L40S, 18 largest (idx 0-17)
sbatch ~/hdf5_to_lerobot/convert_di_cpu.sbatch   # cpu,           9 smallest (idx 18-26)
# 4. verify (§5), then push every dataset PUBLIC to huzheyuan/<name>:
sbatch ~/hdf5_to_lerobot/push_di_array.sbatch
```

Wall-time ~1 h (split, ~11 concurrent). Output datasets: `$DATA_ROOT/lerobot/sim_double_insert_*`.

---

## 2. The data

`/data/group_data/rl/dexterous_robot_data/sim_double_insert_*_hdf5` — **27 datasets, 1419 episodes**,
flat `episode_N.hdf5`, no annotations, no baseline/adversarial split (an earlier project than the
packing sim). Same mujoco dual-xArm env as packing (VERIFIED below). Three families:

| family | datasets | `is_intervention` |
|---|---|---|
| pure human teleop | `0226`, `jasmine_0226`, `riya_0312`, `robyn_0228`, `zheyuan_0508` | **all True** |
| autonomous `full_success` | `r3v0`, `r4`, `r4v0`, `r5` | **all False** |
| DAgger rounds + corrections | `round1..6` (+ date/`_v2` variants), `zheyuan_correction_*` | True at recorded indices |

> A partial Dec-2025 duplicate of a 16-dataset subset sits in
> `/data/group_data/rl/zheyuanh/dual_sim_arm_data/` — **ignore it** (older/incomplete). Also ignore
> the 1-episode `sim_action_coord_frame_bugfix_hdf5`.

### The format (`h5ls -r episode_0.hdf5`) and gotchas

```
/actions/global_action      {T,14}   WORLD-frame command (pos-delta + euler-delta + grip per arm)  <- mocap source
/actions/relative_action    {T,14}   EE/body-frame command (NOT used)
/obses/state/{left,right}/
    tcp_pose                 {T,7}    achieved pose pos(3)+quat(4)  -- quat is scalar-LAST (xyzw)!
    gripper_pos              {T,1}    [0,1] setpoint
    og_action                         ABSENT (this is the key difference vs packing)
    ego_/wrist_/relative2_/joint_qpos/...                                  (unused)
/obses/images/{left,right}/{top,wrist}   {T, maxlen} uint8   JPEG bytes (mujoco RGB via cv2.imencode)
/metadata/interventions     {K}      DAgger human-takeover step indices (absent on teleop/full_success)
/metadata/task              scalar   UNRELIABLE ("real_xarms_shirt_hang_variations" on every file)
/metadata/horizon, og_filename ; /rewards /dones /truncateds                (unused)
```

1. **World action = `actions/global_action`, NOT `og_action`.** Packing reconstructs the mocap from
   `obses/state/{side}/og_action`, which is **absent** here. `actions/global_action` is the world-frame
   delta instead (VERIFIED: `tcp_pose[k]-tcp_pose[k-1] == global_action[k]` to sub-mm; `relative_action`
   is EE-frame and diverges on integration). It already has the `(L[pos,eul,grip], R[pos,eul,grip])`
   14-d layout `reconstruct_mocap` expects, so it's fed in directly.
2. **Same env as packing -> reuse the packing mocap constants.** `reconstruct_mocap` (HOME-init +
   `limit_offset_norm` + clip-to-bounds + `R_delta @ R`) with the **packing** `LEFT/RIGHT_HOME`,
   bounds, `MAX_LIN/ANG` reproduces the recorded pose (VERIFIED: `mocap[0] == HOME` exactly;
   recon-vs-achieved `tcp_pose` median ~1.2-1.8 cm — the expected IK lag, slightly larger than packing
   as it's contact-rich). `observation.state` is this reconstructed **target** (NOT achieved tcp_pose).
3. **Quaternion: reconstruction is scalar-first wxyz throughout** (HOME is wxyz, integrate, emit wxyz,
   `quats_wxyz_to_6d(scalar_first=True)`). The *stored* `tcp_pose` quat is serialised **xyzw
   (scalar-last)**, but it is **never read for the state** (only `tcp_pose[:, :3]` for the self-check).
   Do NOT wire the raw `tcp_pose` quat into the state.
4. **Gripper in [0,1]** (sim setpoint), B1-shifted with the pose — no env-formula reconstruction (that's
   the real arm's 80-840 thing).
5. **Cameras: mujoco RGB via `cv2.imencode` -> R/B swap on decode** (VERIFIED visually: wooden table is
   brown with the swap, blue without; uniform across all 27). 3 cams kept: base=`right/top`,
   `left/wrist`, `right/wrist` (drop `left/top`), 224².
6. **`is_intervention` (3 cases; auto-derived from the dataset name in `convert`):**
   key present -> True at the recorded indices; key absent + `full_success` in name -> all False;
   key absent otherwise (teleop) -> **all True**. `metadata/task` is ignored — use a fixed `--prompt`.

---

## 3. The converter

`convert_xarm_sim_double_insert_to_lerobot.py` = the packing converter's pose/gripper/colour math
wrapped in the real-HDF5 converter's structure (one LeRobot episode per file, fixed `--prompt`,
`is_intervention` feature). Self-contained; runs in the Babel `hdf5_to_lerobot` uv project (pinned
`lerobot`). 20-d state/action = per arm `[pos(3), 6D-rot(6), gripper(1)]`, L then R; `action[i]=state[i+1]`.

```bash
HF_LEROBOT_HOME=/scratch/$USER/lerobot uv run convert_xarm_sim_double_insert_to_lerobot.py \
    --raw-dir $DATA_ROOT/sim_double_insert_round1_hdf5 \
    --repo-id sim_double_insert_round1 \
    --prompt "insert both pegs into the sockets"   [--max-episodes 2]   [--push-to-hub]
```

Prints a per-dataset self-check on episode 0 (B1 |dpos|, 6D det, `mocap[0]`~HOME, recon-vs-achieved
tcp_pose, interv %). FPS=60, codec SVT-AV1 (LeRobot default).

---

## 4. Running on Babel (split arrays)

Babel constraints are the same as `BABEL_HDF5_CONVERSION.md` §4 (NFS invisible to login node;
`cpu` is sbatch-only; convert into node-local `/scratch` then rsync the final dataset to NFS to avoid
PNG churn on `nas6`). To go fast, split the conversion across two partitions over **disjoint** index
ranges of `sim_double_insert_dirs_bysize.txt` (largest-first so the 150-ep `zheyuan_0508` starts first):

- `convert_di_gen.sbatch` — `--partition=general --gres=gpu:L40S:1` (dummy GPU, `CUDA_VISIBLE_DEVICES=""`),
  `--array=0-17` (18 largest). ~8 nodes/user available.
- `convert_di_cpu.sbatch` — `--partition=cpu`, `--array=18-26` (9 smallest). ~10 jobs/user (QOS), but
  often contended (saw ~2-6).

Both: `cd $PROJECT_DIR; uv sync --frozen; uv run convert_...py --raw-dir ... --repo-id <name> --prompt ...;
rsync -a --exclude=images --delete $LOCAL_OUT/<name>/ $DATA_ROOT/lerobot/<name>/`. ~11 concurrent -> ~1 h.

**Known transient:** SVT-AV1 can fail one episode with `non monotonically increasing dts to muxer ...
returned 22` (`av.error.ArgumentError`). It's an encoder flake, not bad data — **just re-run that index**
(`sbatch --array=<i> convert_di_cpu.sbatch`). Happened once (round3_0605); the retry was clean. If it
ever reproduces deterministically, switch that dataset to `libx264` (decodes identically).

---

## 5. Verifying

Per dataset (via a `debug` srun, reads `meta/info.json` + `meta/episodes_stats.jsonl`):
`total_episodes` == raw `episode_*.hdf5` count; `is_intervention` feature present; intervention
fraction matches the family (teleop ~1.0, full_success 0.0, rounds/corrections partial 0.0-0.83).
For the full run: 27/27 valid, 1419 episodes, all semantics correct.

---

## 6. Pushing to the HuggingFace Hub

`push_di_array.sbatch` — one array task per dataset, on `cpu` nodes (NFS + internet). Uploads each
NFS dataset to a **public** repo `huzheyuan/<name>` (name = folder minus `_hdf5`):

```bash
export HF_TOKEN=$(tr -d '[:space:]' < ~/hf_key_2026.txt)   # write token; never printed
hf upload "huzheyuan/<name>" "$DATA_ROOT/lerobot/<name>" . --repo-type=dataset
```

`hf upload` (huggingface_hub 0.36) creates the repo **public by default** (only `--private` makes it
private) and is idempotent (re-runs skip unchanged files). The HF CLI on Babel is `~/miniforge3/bin/hf`;
it was NOT `hf auth login`'d, so pass `HF_TOKEN` from the key file. Do NOT source `env.sh` here (it sets
`HF_HUB_OFFLINE=1`, which blocks uploads). Verify each repo public + complete via the API:
`curl -s https://huggingface.co/api/datasets/huzheyuan/<name>` -> `"private":false`, file count =
`episodes*4 + 5` (parquet + 3 cam mp4 per ep, + 5 meta files).

Result: all 27 at `https://huggingface.co/datasets/huzheyuan/sim_double_insert_<variant>`.

---

## 7. Two copies on Babel (why)

`/data/.../hdf5_to_lerobot/` is the **runtime** project (`env.sh` `PROJECT_DIR`; has `uv.lock`,
`slurm_logs/`; compute nodes run from here). `~/hdf5_to_lerobot/` is the **login-accessible** copy used
to submit `sbatch` (login node can't see `/data`) and to stage/edit. They are otherwise identical; keep
them in sync (`cp` edited files to `/data` before submitting).
