"""Convert real dual-xArm teleop data (.npz episodes) to a LeRobot v2 dataset.

This is the xArm analog of ``examples/yam/convert/convert_yam_data_to_lerobot.py``. The two
robots differ in three ways that this script handles:

1. Storage. Each raw episode is a single ``np.savez`` ``episode_*.npz`` (~700 MB) with
   keys ``obses`` (object array of per-step obs dicts), ``actions`` (N,14), plus
   ``rews/dones/truncateds/infos``. Camera frames are **JPEG-encoded inside** each
   ``obs["images"][cam]`` (no mp4, no hdf5). Four cameras at 360x640 (H x W):
   ``left/top, right/top, left/wrist, right/wrist`` -- all **BGR** (RealSense bgr8 +
   ZED BGRA[:3]). We use ``right/top`` as the base view and drop ``left/top``.

2. Action space = end-effector Cartesian, not joints. We do NOT train on the raw
   ``actions`` array (per-step deltas in the rotating EE/body frame). Instead we use the
   recorded **commanded target pose** ``obs.state["{side}/target_tcp_pose"]`` (the env
   integrates the action onto it every step and records it -- the exactly-invertible
   "mocap"-equivalent anchor). The stored ``action`` is the *absolute world-frame target
   pose* and the stored ``state`` is the *current* target pose; openpi's (Cartesian,
   rotation-aware) delta transform turns the chunk into a rotvec-delta at load time.

   Orientation is stored as the **6D rotation representation** (Zhou et al.: the first two
   columns of the rotation matrix). 6D is lossless for proper rotations (recover R via
   Gram-Schmidt) and is the network-ready *absolute* form (D4), so the stored 20-d state
   needs no rotation transform at load -- only the action does.

   B1 (off-by-one): the collection loop appends ``obs`` *after* ``env.step(action)``, so
   ``obs[i]`` is the state ``actions[i]`` produced (verified: ``target[i]-target[i-1] ==
   og_action[i]``). The action issued *from* ``obs[i]`` therefore reaches target ``i+1``.
   So ``action[i] = state_pose[i+1]`` (last frame: hold -> zero delta).

3. Annotations. Subtasks come from ``audio_transcriptions.json`` (a list of
   ``{start:{step}, end:{step}, transcription}``). Each segment becomes one LeRobot
   episode with ``task = transcription`` so action chunks respect subtask boundaries.
   Segments whose transcription is exactly ``"bad"`` (case-insensitive) are dropped (D15).
   A ``high_level_prompts.json`` sidecar (one high-level string per episode) is written
   for later knowledge-insulation (KI) training.

Gripper (D6). ``target_gripper_pos`` is dead (the env never writes it back), so we
reconstruct the commanded *absolute* gripper target by replaying the env's own formula
(``rmp_env.py`` step): ``dg = action[grip]``; if ``|dg| > 0.05`` the env commands
``clip(gripper_pos - dg*80, 80, 840)``, else it sends the sentinel ``0.0`` which the real
arm treats as "no command" == hold. So the absolute target is ``clip(...)`` when moving
and the *current* ``gripper_pos`` (hold) in the dead-zone. State stores the achieved
``gripper_pos``; action stores this reconstructed target (kept absolute, like YAM).

Example:
  uv run examples/xarm/convert/convert_xarm_data_to_lerobot.py \
    --raw-root /media/huzheyuan/data0/.../data/ADVERSARIAL \
    --repo-id local/xarm_pack
  # smoke test on one episode:
  uv run examples/xarm/convert/convert_xarm_data_to_lerobot.py \
    --raw-root .../ADVERSARIAL --repo-id local/xarm_oneep \
    --rounds xarm_baseline_round1 --max-episodes 1
"""

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import io
import json
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import tqdm
import tyro

IMAGE_SIZE = 224
FPS = 60

# Gripper env constants (must match rmp_env.py step()): dead-zone, units/cmd, real range.
GRIP_DEADZONE = 0.05
GRIP_SCALE = 80.0
GRIP_MIN, GRIP_MAX = 80.0, 840.0
GRIP_ACTION_IDX = (6, 13)  # raw 14-d action gripper dims: left, right (frame-independent).

# LeRobot camera feature name <- raw npz camera key. right/top is the base view (D10).
CAMERA_SOURCES: dict[str, str] = {
    "base": "right/top",
    "left_wrist": "left/wrist",
    "right_wrist": "right/wrist",
}

# State / action vector layout (D8): per arm [pos(3), 6D(6), gripper(1)] = 10 -> 20, L then R.
# 6D = first two columns of the rotation matrix: (R[:,0], R[:,1]).
ARMS = ("left", "right")
ARM_FIELDS = ("pos_x", "pos_y", "pos_z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")
VECTOR_NAMES = [f"{side}_{f}" for side in ARMS for f in ARM_FIELDS]
VECTOR_DIM = len(VECTOR_NAMES)  # 20

# De-duped default roster (see action_stats.py: round1K is a round1 subset; *_ALL /
# RobynALL are aggregates; round0 and debug_* are excluded by request).
DEFAULT_ROUNDS = (
    "xarm_adversarial_round1",
    "xarm_adversarial_round2",
    "xarm_adversarial_round3",
    "xarm_baseline_round1",
    "xarm_baseline_round1good",
    "xarm_baseline_round2",
    "xarm_baseline_round3",
    "xarm_baseline_round4",
    "xarm_baseline_round5",
    "xarm_baseline_round6",
)


@dataclasses.dataclass
class SubtaskSegment:
    task_text: str
    start_frame: int  # inclusive
    end_frame: int  # exclusive


def decode_jpeg_to_rgb(jpeg_arr: np.ndarray) -> np.ndarray:
    """Decode a cv2-encoded JPEG uint8 array -> (IMAGE_SIZE, IMAGE_SIZE, 3) true RGB.

    The camera frames are BGR (RealSense bgr8 / ZED BGRA[:3]), but ``cv2.imencode``
    consumes that BGR convention when writing the JPEG, so a standard decoder (PIL)
    already returns true RGB -- NO channel swap. Verified against the cv2-written review
    mp4: plain-PIL channel means match it (|d|~6); swapped means do not (|d|~50).
    Resize stretches to a square (no aspect preservation, D11).
    """
    im = Image.open(io.BytesIO(np.asarray(jpeg_arr, dtype=np.uint8).tobytes()))
    return np.asarray(im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC))


def quats_wxyz_to_6d(quats_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-first unit quaternions -> (..., 6) rotation rep (first two cols of R)."""
    flat = quats_wxyz.reshape(-1, 4)
    mats = Rotation.from_quat(flat, scalar_first=True).as_matrix()  # (M, 3, 3)
    sixd = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=-1)  # (M, 6)
    return sixd.reshape(*quats_wxyz.shape[:-1], 6).astype(np.float32)


def sixd_to_matrix(sixd: np.ndarray) -> np.ndarray:
    """Gram-Schmidt inverse of ``quats_wxyz_to_6d`` -> (3, 3). Used for self-checks/deploy."""
    a1, a2 = sixd[:3], sixd[3:6]
    b1 = a1 / np.linalg.norm(a1)
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / np.linalg.norm(a2p)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # columns = basis vectors


def reconstruct_gripper_target(raw_actions: np.ndarray, grip_cur: np.ndarray) -> np.ndarray:
    """Replay the env gripper formula (D6) to recover the absolute commanded gripper target.

    Args:
      raw_actions: (N, 14) raw body-frame deltas; gripper dims at ``GRIP_ACTION_IDX``.
      grip_cur: (N, num_arms) achieved gripper position (the env's base for the command).

    Returns (N, num_arms): for frame i the absolute gripper target commanded by ``action[i+1]``
    (B1 alignment: ``action[i+1]`` drives ``obs[i] -> obs[i+1]``, with base ``grip_cur[i]``). The
    env's ``rmp_env.step`` computes ``dg = -action[grip]``; if ``|dg| > 0.05`` it commands
    ``clip(gripper_pos + dg*80, 80, 840) == clip(gripper_pos - action[grip]*80, ...)`` else the
    sentinel ``0.0`` == "no command" == hold. We store the *absolute* target, i.e. the clip when
    moving and the current ``grip_cur`` (hold) in the dead-zone and on the last frame.
    """
    grip_tgt = np.empty_like(grip_cur)
    for a in range(grip_cur.shape[1]):
        dg = raw_actions[1:, GRIP_ACTION_IDX[a]]  # action[i+1], i in [0, N-2]
        base = grip_cur[:-1, a]
        moving = np.abs(dg) > GRIP_DEADZONE
        moved_target = np.clip(base - dg * GRIP_SCALE, GRIP_MIN, GRIP_MAX)
        grip_tgt[:-1, a] = np.where(moving, moved_target, base)  # dead-zone sentinel 0.0 == hold
        grip_tgt[-1, a] = grip_cur[-1, a]  # last frame: hold
    return grip_tgt


def shift_pose_b1(pose6d: np.ndarray) -> np.ndarray:
    """B1 off-by-one shift: the action pose at frame i is the next frame's target; last holds.

    ``action[i] = state_pose[i+1]`` for i in [0, N-2]; the last frame repeats (zero delta).
    Operates along axis 0, so it works for any trailing shape (e.g. (N, 2, 9)).
    """
    return np.concatenate([pose6d[1:], pose6d[-1:]], axis=0)


def load_episode(npz_path: Path, decode_workers: int = 16):
    """Return (images_per_feature_cam, state (N,20), action (N,20), num_frames).

    state[i]  = [pos(3), 6D(6), gripper_pos(1)] per arm (L,R), absolute world target + achieved grip.
    action[i] = [pos(3), 6D(6), gripper_target(1)] per arm: pose = state_pose[i+1] (B1 shift, hold
                last); gripper = env-formula reconstructed absolute command (D6).
    """
    data = np.load(npz_path, allow_pickle=True)
    obses = data["obses"]
    raw_actions = np.asarray(data["actions"], dtype=np.float32)  # (N, 14) body-frame deltas
    num_frames = len(obses)
    if num_frames < 2:
        return None

    # Gather absolute target poses + achieved gripper for every frame/arm.
    pos = np.zeros((num_frames, len(ARMS), 3), dtype=np.float32)
    quat = np.zeros((num_frames, len(ARMS), 4), dtype=np.float32)  # wxyz
    grip_cur = np.zeros((num_frames, len(ARMS)), dtype=np.float32)
    for t in range(num_frames):
        st = obses[t]["state"]
        for a, side in enumerate(ARMS):
            tp = np.asarray(st[f"{side}/target_tcp_pose"], dtype=np.float32)  # xyz + quat wxyz
            pos[t, a] = tp[:3]
            quat[t, a] = tp[3:7]
            grip_cur[t, a] = float(np.ravel(st[f"{side}/gripper_pos"])[0])

    pose6d = np.concatenate([pos, quats_wxyz_to_6d(quat)], axis=-1)  # (N, 2, 9) absolute, world

    # Reconstruct commanded absolute gripper target (D6) and B1-shift the pose to the next target.
    grip_tgt = reconstruct_gripper_target(raw_actions, grip_cur)  # (N, 2)
    pose_next = shift_pose_b1(pose6d)  # (N, 2, 9)
    state = np.concatenate(
        [np.concatenate([pose6d[:, a], grip_cur[:, a, None]], axis=1) for a in range(len(ARMS))], axis=1
    ).astype(np.float32)
    action = np.concatenate(
        [np.concatenate([pose_next[:, a], grip_tgt[:, a, None]], axis=1) for a in range(len(ARMS))], axis=1
    ).astype(np.float32)
    assert state.shape == (num_frames, VECTOR_DIM), state.shape
    assert action.shape == (num_frames, VECTOR_DIM), action.shape

    images_per_cam: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        for feat, src in CAMERA_SOURCES.items():
            jpegs = [obses[t]["images"][src] for t in range(num_frames)]
            frames = list(pool.map(decode_jpeg_to_rgb, jpegs))
            images_per_cam[feat] = np.stack(frames)  # (N, 224, 224, 3) uint8 RGB

    del data, obses
    return images_per_cam, state, action, num_frames


def parse_subtask_segments(ann_path: Path, episode_length: int) -> tuple[list[SubtaskSegment], int]:
    """audio_transcriptions.json -> (segments, n_bad_dropped) using each [start.step, end.step).

    Out-of-segment frames (e.g. pre-first-utterance settling) are dropped. Entries with
    empty/whitespace transcription are skipped; entries whose transcription is exactly
    "bad" (case-insensitive) are dropped and counted (D15). Steps clip to [0, length).
    """
    if not ann_path.exists():
        return [], 0
    try:
        anns = json.loads(ann_path.read_text())
    except json.JSONDecodeError:
        return [], 0
    if not isinstance(anns, list):
        return [], 0

    segments: list[SubtaskSegment] = []
    n_bad = 0
    for ann in sorted(anns, key=lambda a: a.get("start", {}).get("step", 0)):
        text = (ann.get("transcription") or "").strip()
        if not text:
            continue
        if text.lower() == "bad":  # D15: annotator-marked junk segment.
            n_bad += 1
            continue
        start = int(ann.get("start", {}).get("step", 0))
        end = int(ann.get("end", {}).get("step", episode_length))
        start = max(0, min(start, episode_length))
        end = max(0, min(end, episode_length))
        if end - start >= 1:
            segments.append(SubtaskSegment(task_text=text, start_frame=start, end_frame=end))
    return segments, n_bad


def collect_episode_dirs(raw_root: Path, rounds: tuple[str, ...]) -> list[Path]:
    """Episode dirs (one .npz each) across the selected rounds, excluding FAILED_*."""
    episode_dirs: list[Path] = []
    for rnd in rounds:
        rnd_dir = raw_root / rnd
        if not rnd_dir.is_dir():
            print(f"  WARNING: round dir not found: {rnd_dir}")
            continue
        for ep_dir in sorted(rnd_dir.iterdir()):
            if not ep_dir.is_dir() or ep_dir.name.startswith("FAILED_"):
                continue
            if any(ep_dir.glob("episode_*.npz")):
                episode_dirs.append(ep_dir)
    return episode_dirs


def create_empty_dataset(repo_id: str) -> LeRobotDataset:
    features = {
        "observation.state": {"dtype": "float32", "shape": (VECTOR_DIM,), "names": [VECTOR_NAMES]},
        "action": {"dtype": "float32", "shape": (VECTOR_DIM,), "names": [VECTOR_NAMES]},
    }
    for feat in CAMERA_SOURCES:
        features[f"observation.images.{feat}"] = {
            "dtype": "video",
            "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
            "names": ["channels", "height", "width"],
        }
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="dual_xarm",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=10,
        image_writer_threads=5,
    )


def _self_check(state: np.ndarray, action: np.ndarray, quat0: np.ndarray, ep_name: str, n_seg: int) -> None:
    """One-time numeric sanity print: B1 per-step motion, 6D round-trip, gripper recon."""
    # B1: per-step pos delta = action_pose[i] - state_pose[i] should be mm-cm, not chunk jumps.
    dl = np.linalg.norm(action[:-1, 0:3] - state[:-1, 0:3], axis=1)
    dr = np.linalg.norm(action[:-1, 10:13] - state[:-1, 10:13], axis=1)
    print(f"  [self-check] {ep_name}: segments={n_seg}")
    print(f"    per-step |dpos| median  L={np.median(dl):.4f} m  R={np.median(dr):.4f} m (expect mm-cm)")
    # 6D round-trip on frame 0 left arm vs the source quaternion.
    rt = sixd_to_matrix(state[0, 3:9])
    src = Rotation.from_quat(quat0, scalar_first=True).as_matrix()
    print(f"    6D round-trip max|dR|={np.max(np.abs(rt - src)):.2e}  det={np.linalg.det(rt):+.4f} (expect ~0, +1)")
    # Gripper recon: where a real change is commanded, how far is achieved-next from the target?
    for a, idx in ((0, 9), (1, 19)):
        tgt, cur, nxt = action[:-1, idx], state[:-1, idx], state[1:, idx]
        moved = np.abs(tgt - cur) > 1.0
        side = ARMS[a]
        if moved.any():
            err = np.abs(tgt[moved] - nxt[moved])
            print(
                f"    gripper {side}: cmd-change {moved.mean() * 100:4.1f}% of steps, "
                f"|target-achieved_next| median={np.median(err):.1f} (range {GRIP_MIN:.0f}-{GRIP_MAX:.0f})"
            )
        else:
            print(f"    gripper {side}: no commanded changes (stayed in dead-zone)")


def convert(
    raw_root: Path,
    repo_id: str = "local/xarm_pack",
    rounds: tuple[str, ...] = DEFAULT_ROUNDS,
    high_level_prompt: str = "pack the objects into the boxes",
    max_episodes: int | None = None,
    *,
    push_to_hub: bool = False,
):
    """Convert dual-xArm .npz episodes to a LeRobot v2 dataset (one episode per subtask)."""
    raw_root = Path(raw_root)
    episode_dirs = collect_episode_dirs(raw_root, rounds)
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
    print(f"Found {len(episode_dirs)} raw episodes across {len(rounds)} rounds.")

    dataset = create_empty_dataset(repo_id)
    high_level_per_episode: list[str] = []
    subtask_count = 0
    frames_written = 0
    bad_dropped = 0
    skipped: list[tuple[str, str]] = []
    selfcheck_done = False

    for ep_dir in tqdm.tqdm(episode_dirs, desc="Converting"):
        npz_path = next(iter(ep_dir.glob("episode_*.npz")))
        loaded = load_episode(npz_path)
        if loaded is None:
            skipped.append((ep_dir.name, "too few frames"))
            continue
        images_per_cam, state, action, num_frames = loaded

        for cam, imgs in images_per_cam.items():
            assert imgs.shape[0] == num_frames, f"{ep_dir.name}: {cam} {imgs.shape[0]} != {num_frames}"

        segments, n_bad = parse_subtask_segments(ep_dir / "audio_transcriptions.json", num_frames)
        bad_dropped += n_bad
        if not segments:
            skipped.append((ep_dir.name, "no usable transcription segments"))
            continue

        if not selfcheck_done:
            # quat0 reconstructed from the stored 6D would be circular; reload frame-0 left quat.
            q0 = np.asarray(np.load(npz_path, allow_pickle=True)["obses"][0]["state"]["left/target_tcp_pose"])[3:7]
            _self_check(state, action, q0.astype(np.float32), ep_dir.name, len(segments))
            selfcheck_done = True

        for seg in segments:
            for i in range(seg.start_frame, seg.end_frame):
                frame = {"observation.state": state[i], "action": action[i], "task": seg.task_text}
                for cam in CAMERA_SOURCES:
                    frame[f"observation.images.{cam}"] = images_per_cam[cam][i]
                dataset.add_frame(frame)
            dataset.save_episode()
            high_level_per_episode.append(high_level_prompt)
            subtask_count += 1
            frames_written += seg.end_frame - seg.start_frame

    dataset.stop_image_writer()

    sidecar = HF_LEROBOT_HOME / repo_id / "meta" / "high_level_prompts.json"
    sidecar.write_text(
        json.dumps({"high_level_prompt": high_level_prompt, "per_episode": high_level_per_episode}, indent=2)
    )

    print("\n=== Conversion summary ===")
    print(f"Raw episodes processed: {len(episode_dirs) - len(skipped)} / {len(episode_dirs)}")
    print(f"LeRobot episodes (subtasks): {subtask_count}   frames: {frames_written}")
    print(f"BAD segments dropped (D15): {bad_dropped}")
    print(f"High-level sidecar: {sidecar}")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for name, reason in skipped[:20]:
            print(f"  - {name}: {reason}")
    print(f"Dataset: {HF_LEROBOT_HOME / repo_id}")

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(convert)
