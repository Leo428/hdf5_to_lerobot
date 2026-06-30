# Converting sim dual-xArm *packing* HDF5 + human-pass annotations to LeRobot

A runbook for the **sim packing teleop** conversion via `convert_xarm_sim_data_to_lerobot.py`. Unlike
the real-HDF5 and sim-double-insert pipelines (which run as Babel SLURM arrays — see
`BABEL_HDF5_CONVERSION.md` and `SIM_DOUBLE_INSERT_CONVERSION.md`), packing was converted **locally**
on the workstation that holds the raw data (the converter's `DEFAULT_ROOT` is a local path), one
`(proto, round)` dataset at a time.

---

## 1. TL;DR

```bash
# from the openpi repo, on the machine holding the raw data:
HF_LEROBOT_HOME=/media/huzheyuan/data0/lerobot uv run \
  examples/xarm/convert/convert_xarm_sim_data_to_lerobot.py --proto baseline --round round1
# smoke test 2 episodes:
... --proto baseline --round round1 --repo-id local/xarm_sim_smoke --max-episodes 2
```

One LeRobot dataset per `(proto, round)`. `proto ∈ {baseline, adversarial}`. Default raw root:
`/home/huzheyuan/kshitiz2/RAC/dual_xarms/dual_xarms_sim/data/new_data` (override with `--root`).

---

## 2. The data layout

Per `(proto, round)`, the converter pairs each annotated goal with its demo (`collect_episodes`):

```
<root>/<proto>/<round>/demos/<proto>/<goal>/episode_0.hdf5                                  # the demo
<root>/human_pass_annotations/<proto>/<round>/pipeline_a/<proto>/<goal>/<proto>__*__episode_0.json   # annotations
```

`human_pass_annotations` are human-corrected, terse "Pack/Transfer/Remove X into the Y tray."
imperatives. The annotation JSON has `segments` (each `{start_step, end_step (INCLUSIVE), summary}`)
that tile `[0, N)`. **Each segment becomes one LeRobot episode** with `task = summary`, so action
chunks respect subtask boundaries. Segments whose summary is empty or `"bad"` are dropped; segments
shorter than `--min-seg-frames` are dropped.

Per-episode HDF5 (`episode_0.hdf5`): JPEG frames under `obses/images/{right/top,left/wrist,right/wrist}`;
per-step state under `obses/state/{left,right}/{og_action, gripper_pos, tcp_pose, ...}`; a high-level
goal in the `episode_metadata` attr (`goal_spec.description`).

---

## 3. The 4 fixes / gotchas

1. **No recorded `target_tcp_pose` -> reconstruct the mocap.** The sim never logs the IK mocap target,
   but it's a deterministic recurrence over the recorded **world-frame** action
   `obses/state/{l,r}/og_action` (**NOT** `actions/global_action`, which is the EE/body frame here).
   `reconstruct_mocap` replays `PackingEnv.step`: HOME-init, `pos = clip(pos + limit(d_pos, MAX_LIN),
   bounds)`, `R = R(limit(d_eul, MAX_ANG)) @ R`. Validated bit-exact vs the live PackingEnv mocap.
   `observation.state` = this reconstructed **target** (per arm `[pos(3), 6D-rot(6), gripper(1)]`,
   L then R = 20-d); `action[i] = state[i+1]` (B1 off-by-one; last frame holds).
2. **Quaternions are scalar-first wxyz** (mujoco native); HOME is wxyz and `quats_wxyz_to_6d` uses
   `scalar_first=True`. 6D rep = first two columns of R.
3. **Gripper in `[0,1]`** (the tendon-actuator setpoint, `ctrl/255`), not the real arm's 80-840.
   `gripper_pos` is the commanded setpoint, so the action gripper is just the next setpoint (B1 shift)
   — no env-formula reconstruction.
4. **Colour: mujoco renders RGB, then `cv2.imencode` stores it.** A `cv2.imdecode` returns true RGB
   directly; equivalently, decoding with PIL gives the R/B-swapped image so the code swaps `[..., ::-1]`
   back to true RGB (the OPPOSITE of the real BGR cameras, which need no swap). Verified visually
   (wooden table is brown, not blue). 3 cams: base=`right/top`, `left_wrist`, `right_wrist`; 224².

---

## 4. The converter & outputs

`convert_xarm_sim_data_to_lerobot.py` (in the openpi repo, `examples/xarm/convert/`). Key CLI:

```
--proto {baseline,adversarial}   --round <rnd>   [--repo-id <id>]   [--root <path>]
[--max-episodes N]   [--min-seg-frames N]   [--decode-workers 8]
[--image-writer-processes 3]   [--image-writer-threads 4]
```

- Default `repo-id` = `local/xarm_sim_{proto}_{rnd}`; output dataset at `$HF_LEROBOT_HOME/<repo-id>`.
- FPS=60, SVT-AV1 video. 20-d state/action (same layout as the real + double-insert converters).
- Writes a **`meta/high_level_prompts.json`** sidecar: one high-level goal string per LeRobot
  (subtask) episode, for later knowledge-insulation (KI) training. (This is plain Pi0History data +
  the KI sidecar — contrast the rejection converter, which is high-level-goal-only, no sidecar.)
- A one-time per-dataset self-check prints B1 |dpos|, 6D round-trip, `mocap[0]`~HOME, and recon-mocap
  vs achieved `tcp_pose` (expect cm-level tracking).

Run `compute_norm_stats.py` for any new config before training (not produced by this script).
