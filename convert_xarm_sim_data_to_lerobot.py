"""Convert SIM dual-xArm packing teleop (HDF5) + human-pass annotations to LeRobot v2.

Sim analog of ``convert_xarm_data_to_lerobot.py``. The pipeline (20-d Cartesian
state/action, one LeRobot episode per subtask, ``task`` = subtask text) is identical to
the real converter; only the *inputs* differ, in four ways this script handles:

1. Storage = HDF5 (not npz). Per episode ``episode_0.hdf5`` with JPEG-encoded camera
   frames under ``obses/images/{right/top,left/wrist,right/wrist}`` (object arrays) and
   per-step state under ``obses/state/{left,right}/{og_action,gripper_pos,tcp_pose,...}``.

2. No recorded ``target_tcp_pose``. The sim never logs the IK mocap target, but it is a
   pure deterministic recurrence over the recorded world-frame action
   ``obses/state/{l,r}/og_action`` (NOT ``actions/global_action``, which is EE-frame), so
   we reconstruct it post-hoc with no physics (``reconstruct_mocap``), HOME-init +
   ``limit_offset_norm`` + clip-to-bounds + ``R_delta @ R_old``. This recovered mocap *is*
   the sim's ``target_tcp_pose`` (the D1 anchor) -- validated bit-exact vs the live
   PackingEnv mocap. Quaternions are scalar-first wxyz (mujoco native).

3. Gripper in [0,1] (the tendon-actuator *setpoint*, ``ctrl/255``), not the real arm's
   80-840 range. ``gripper_pos`` is the commanded setpoint (not achieved), so the action
   target is simply the next setpoint (B1 shift) -- no env-formula reconstruction. Hence
   the whole 20-d action is just the B1-shifted state (``action[i] = state[i+1]``).

4. Color: mujoco renders RGB, then ``cv2.imencode`` stores it, so ``cv2.imdecode`` returns
   true RGB directly (the real cameras are BGR and need the opposite handling). Verified
   visually (wooden table, not blue).

Annotations = ``human_pass_annotations`` (human-corrected, terse "Pack/Transfer/Remove X
into the Y tray." imperatives matching the real dataset). Segments tile [0, N)
contiguously, so no frames are dropped and ``task`` = ``summary`` verbatim.

One LeRobot dataset per (protocol, round). Example:
  HF_LEROBOT_HOME=/media/huzheyuan/data0/lerobot uv run \
    examples/xarm/convert_xarm_sim_data_to_lerobot.py --proto baseline --round round1
  # smoke test:
  ... --proto baseline --round round1 --repo-id local/xarm_sim_smoke --max-episodes 2
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

IMAGE_SIZE = 224
FPS = 60
DEFAULT_ROOT = Path("/home/huzheyuan/kshitiz2/RAC/dual_xarms/dual_xarms_sim/data/new_data")

# --- PackingEnv mocap recurrence constants (verified vs packing_env.py L198-218). ---
LEFT_HOME = np.asarray([-0.35, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0], dtype=np.float64)  # xyz + quat wxyz
RIGHT_HOME = np.asarray([0.35, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0], dtype=np.float64)
LEFT_BOUNDS = (np.asarray([-0.7, 0.2, 0.0]), np.asarray([0.1, 0.6, 0.3]))
RIGHT_BOUNDS = (np.asarray([-0.1, 0.2, 0.0]), np.asarray([0.7, 0.6, 0.3]))
MAX_LIN = 1.0 / FPS  # _MAX_LINEAR_VELOCITY / control_freq
MAX_ANG = (np.pi / 3) / FPS  # _MAX_ANGULAR_VELOCITY / control_freq
# og_action 14-d layout: L[pos(0:3), euler(3:6), grip(6)], R[pos(7:10), euler(10:13), grip(13)].
ARM_OG = ((slice(0, 3), slice(3, 6)), (slice(7, 10), slice(10, 13)))

# LeRobot camera feature name <- HDF5 view key. right/top is the base view (matches env.py).
CAMERA_SOURCES: dict[str, str] = {
    "base": "right/top",
    "left_wrist": "left/wrist",
    "right_wrist": "right/wrist",
}

ARMS = ("left", "right")
ARM_FIELDS = ("pos_x", "pos_y", "pos_z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")
VECTOR_NAMES = [f"{side}_{f}" for side in ARMS for f in ARM_FIELDS]
VECTOR_DIM = len(VECTOR_NAMES)  # 20


def _limit_rows(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Row-wise ``limit_offset_norm``: scale each (N,3) row to ``max_norm`` if it exceeds it."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.where(n > max_norm, max_norm / np.maximum(n, 1e-12), 1.0)


def reconstruct_mocap(og14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """og_action (N,14) -> (pos (N,2,3), quat (N,2,4) scalar-first wxyz) IK mocap targets.

    Deterministic replay of ``PackingEnv.step`` (L1722-1737): per arm
    ``pos = clip(pos + limit(d_pos, MAX_LIN), bounds)`` and ``R = R(limit(d_eul, MAX_ANG)) @ R``,
    HOME-initialised. ``mocap[k]`` is the target AFTER step k (aligns with ``obses[k]``).
    """
    n = og14.shape[0]
    pos = np.zeros((n, 2, 3), dtype=np.float64)
    quat = np.zeros((n, 2, 4), dtype=np.float64)
    for a, (home, bounds) in enumerate(((LEFT_HOME, LEFT_BOUNDS), (RIGHT_HOME, RIGHT_BOUNDS))):
        psl, esl = ARM_OG[a]
        d_pos = _limit_rows(og14[:, psl], MAX_LIN)  # (N,3)
        d_eul = _limit_rows(og14[:, esl], MAX_ANG)  # (N,3)
        delta_r = Rotation.from_euler("xyz", d_eul).as_matrix()  # (N,3,3) batched
        lo, hi = bounds
        # position: clip makes it sequential
        p = home[:3].copy()
        for k in range(n):
            p = np.clip(p + d_pos[k], lo, hi)
            pos[k, a] = p
        # rotation: m[k] = delta_r[k] @ m[k-1]
        m = Rotation.from_quat(home[3:7], scalar_first=True).as_matrix()
        mats = np.empty((n, 3, 3), dtype=np.float64)
        for k in range(n):
            m = delta_r[k] @ m
            mats[k] = m
        quat[:, a] = Rotation.from_matrix(mats).as_quat(scalar_first=True)
    return pos, quat


def quats_wxyz_to_6d(quats_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-first unit quaternions -> (..., 6) rep (first two columns of R)."""
    flat = quats_wxyz.reshape(-1, 4)
    mats = Rotation.from_quat(flat, scalar_first=True).as_matrix()
    sixd = np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=-1)
    return sixd.reshape(*quats_wxyz.shape[:-1], 6).astype(np.float32)


def sixd_to_matrix(sixd: np.ndarray) -> np.ndarray:
    """Gram-Schmidt inverse of ``quats_wxyz_to_6d`` -> (3,3) (for self-checks)."""
    a1, a2 = sixd[:3], sixd[3:6]
    b1 = a1 / np.linalg.norm(a1)
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / np.linalg.norm(a2p)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def decode_jpeg_to_rgb(jpeg_arr: np.ndarray) -> np.ndarray:
    """Decode a cv2-encoded JPEG -> (224,224,3) true RGB.

    The frames are mujoco-RGB written by ``cv2.imencode``, which swaps R/B vs a standard
    PIL decode -- so PIL-decode then swap ``[..., ::-1]`` recovers true RGB (opposite of
    the real BGR cameras, which need no swap). Stretch-resize to a square with PIL BICUBIC
    to match the real converter / env.py (swap commutes with resize).
    """
    im = Image.open(io.BytesIO(np.asarray(jpeg_arr, dtype=np.uint8).tobytes()))
    im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])  # swap R/B -> true RGB


def shift_b1(arr: np.ndarray) -> np.ndarray:
    """B1 off-by-one: ``action[i] = state[i+1]`` (next absolute target); last frame holds."""
    return np.concatenate([arr[1:], arr[-1:]], axis=0)


def load_episode(hdf5_path: Path, decode_workers: int = 8):
    """Return (images_per_cam, state (N,20), action (N,20), N, high_level_prompt).

    state[i]  = per arm [mocap_pos(3), 6D(mocap_quat)(6), gripper_setpoint(1)], L then R.
    action[i] = state[i+1] (B1 shift; last holds) -- next absolute world target + next gripper setpoint.
    """
    with h5py.File(hdf5_path, "r") as h:
        og_l = np.asarray(h["obses/state/left/og_action"][:], dtype=np.float64)  # (N,7)
        og_r = np.asarray(h["obses/state/right/og_action"][:], dtype=np.float64)  # (N,7)
        og14 = np.concatenate([og_l, og_r], axis=1)  # (N,14)
        grip = np.stack(
            [
                np.asarray(h["obses/state/left/gripper_pos"][:]).reshape(-1),
                np.asarray(h["obses/state/right/gripper_pos"][:]).reshape(-1),
            ],
            axis=1,
        ).astype(np.float32)  # (N,2) in [0,1]
        n = og14.shape[0]
        # high-level goal description (for KI sidecar), best-effort.
        prompt = ""
        if "episode_metadata" in h.attrs:
            try:
                prompt = json.loads(h.attrs["episode_metadata"]).get("goal_spec", {}).get("description", "")
            except (json.JSONDecodeError, TypeError, AttributeError):
                prompt = ""
        jpeg_lists = {feat: [h[f"obses/images/{src}"][t] for t in range(n)] for feat, src in CAMERA_SOURCES.items()}

    if n < 2:
        return None

    pos, quat = reconstruct_mocap(og14)  # (N,2,3), (N,2,4) wxyz
    pose6d = np.concatenate([pos, quats_wxyz_to_6d(quat)], axis=-1).astype(np.float32)  # (N,2,9)
    state = np.concatenate([np.concatenate([pose6d[:, a], grip[:, a, None]], axis=1) for a in range(2)], axis=1).astype(
        np.float32
    )  # (N,20)
    action = shift_b1(state)  # (N,20); action[i]=state[i+1]
    assert state.shape == (n, VECTOR_DIM), state.shape
    assert action.shape == (n, VECTOR_DIM), action.shape

    images_per_cam: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        for feat, jpegs in jpeg_lists.items():
            images_per_cam[feat] = np.stack(list(pool.map(decode_jpeg_to_rgb, jpegs)))  # (N,224,224,3) RGB
    return images_per_cam, state, action, n, prompt


def parse_segments(ann_path: Path, n_frames: int, min_frames: int = 1):
    """human_pass annotation json -> list of (task_text, start_frame, end_frame_exclusive).

    Segments tile [0, N) with ``end_step`` INCLUSIVE; clip to the episode, drop empty/'bad',
    and drop segments shorter than ``min_frames`` frames (too short to train on).
    Returns (segments, n_dropped, n_clipped, n_short).
    """
    d = json.loads(ann_path.read_text())
    segs = sorted(d.get("segments", []), key=lambda s: s.get("start_step", 0))
    out = []
    dropped = clipped = short = 0
    for s in segs:
        text = (s.get("summary") or "").strip()
        if not text or text.lower() == "bad":
            dropped += 1
            continue
        start = max(0, int(s.get("start_step", 0)))
        end = int(s.get("end_step", n_frames - 1)) + 1  # inclusive -> exclusive
        if end > n_frames or start > n_frames:
            clipped += 1
        start = min(start, n_frames)
        end = min(end, n_frames)
        dur = end - start
        if dur < 1:
            dropped += 1
        elif dur < min_frames:
            short += 1
        else:
            out.append((text, start, end))
    return out, dropped, clipped, short


def collect_episodes(root: Path, proto: str, rnd: str):
    """Yield (goal_id, hdf5_path, ann_path) for human_pass-annotated episodes with a demo."""
    ann_dir = root / "human_pass_annotations" / proto / rnd / "pipeline_a" / proto
    demo_dir = root / proto / rnd / "demos" / proto
    items = []
    for ann in sorted(ann_dir.glob(f"*/{proto}__*__episode_0.json")):
        goal = ann.parent.name
        hdf5 = demo_dir / goal / "episode_0.hdf5"
        if hdf5.exists():
            items.append((goal, hdf5, ann))
    return items


def create_empty_dataset(repo_id: str, image_writer_processes: int, image_writer_threads: int) -> LeRobotDataset:
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
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def _self_check(state: np.ndarray, action: np.ndarray, pos: np.ndarray, hdf5_path: Path, name: str) -> None:
    """One-time sanity: B1 per-step motion, 6D round-trip, mocap[0]~HOME, recon-vs-achieved tcp_pose."""
    dl = np.linalg.norm(action[:-1, 0:3] - state[:-1, 0:3], axis=1)
    dr = np.linalg.norm(action[:-1, 10:13] - state[:-1, 10:13], axis=1)
    print(f"  [self-check] {name}:")
    print(
        f"    per-step |dpos| median  L={np.median(dl):.4f} m  R={np.median(dr):.4f} m (expect mm-cm, < {MAX_LIN:.4f})"
    )
    rt = sixd_to_matrix(state[0, 3:9])
    print(f"    6D det(frame0 L)={np.linalg.det(rt):+.4f} (expect +1)")
    print(f"    mocap[0] L pos={np.round(pos[0, 0], 4)} vs HOME={np.round(LEFT_HOME[:3], 4)} (expect ~equal, 1 step)")
    # env-free cross-check: reconstructed mocap target should track the recorded achieved tcp_pose (cm).
    with h5py.File(hdf5_path, "r") as h:
        tcp_l = np.asarray(h["obses/state/left/tcp_pose"][:, :3], dtype=np.float64)
        tcp_r = np.asarray(h["obses/state/right/tcp_pose"][:, :3], dtype=np.float64)
    el = np.linalg.norm(pos[:, 0] - tcp_l, axis=1)
    er = np.linalg.norm(pos[:, 1] - tcp_r, axis=1)
    print(
        f"    recon-mocap vs achieved tcp_pose: L median={np.median(el) * 100:.2f}cm max={el.max() * 100:.1f}cm | "
        f"R median={np.median(er) * 100:.2f}cm max={er.max() * 100:.1f}cm (expect cm-level tracking)"
    )


def convert(
    proto: str,
    rnd: str,
    repo_id: str,
    root: Path,
    max_episodes: int | None,
    min_frames: int,
    decode_workers: int,
    image_writer_processes: int,
    image_writer_threads: int,
):
    items = collect_episodes(root, proto, rnd)
    if max_episodes is not None:
        items = items[:max_episodes]
    print(f"[{proto}/{rnd}] {len(items)} annotated episodes -> {repo_id} (min_seg_frames={min_frames})")
    print(f"  HF_LEROBOT_HOME={HF_LEROBOT_HOME}")

    dataset = create_empty_dataset(repo_id, image_writer_processes, image_writer_threads)
    high_level: list[str] = []
    subtask_count = frames_written = total_dropped = total_clipped = total_short = 0
    skipped: list[tuple[str, str]] = []
    selfcheck_done = False

    for gi, (goal, hdf5, ann) in enumerate(items):
        loaded = load_episode(hdf5, decode_workers)
        if loaded is None:
            skipped.append((goal, "too few frames"))
            continue
        images_per_cam, state, action, n, prompt = loaded
        segments, dropped, clipped, short = parse_segments(ann, n, min_frames)
        total_dropped += dropped
        total_clipped += clipped
        total_short += short
        if not segments:
            skipped.append((goal, "no usable segments"))
            continue

        if not selfcheck_done:
            pos = state[:, [0, 1, 2, 10, 11, 12]].reshape(n, 2, 3)  # L,R mocap pos
            _self_check(state, action, pos, hdf5, goal)
            selfcheck_done = True

        for text, start, end in segments:
            for i in range(start, end):
                frame = {"observation.state": state[i], "action": action[i], "task": text}
                for cam in CAMERA_SOURCES:
                    frame[f"observation.images.{cam}"] = images_per_cam[cam][i]
                dataset.add_frame(frame)
            dataset.save_episode()
            high_level.append(prompt)
            subtask_count += 1
            frames_written += end - start
        if (gi + 1) % 10 == 0:
            print(
                f"  ...{gi + 1}/{len(items)} episodes, {subtask_count} subtask-eps, {frames_written} frames", flush=True
            )

    dataset.stop_image_writer()
    sidecar = HF_LEROBOT_HOME / repo_id / "meta" / "high_level_prompts.json"
    sidecar.write_text(json.dumps({"per_episode": high_level}, indent=2))

    print(f"\n=== [{proto}/{rnd}] summary ===")
    print(f"  source episodes: {len(items) - len(skipped)}/{len(items)}")
    print(f"  LeRobot subtask episodes: {subtask_count}   frames: {frames_written}")
    print(
        f"  segments dropped(empty/bad)={total_dropped} short(<{min_frames}f)={total_short} clipped(oob)={total_clipped}"
    )
    if skipped:
        print(f"  skipped {len(skipped)}: {skipped[:10]}")
    print(f"  dataset: {HF_LEROBOT_HOME / repo_id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proto", required=True, choices=["baseline", "adversarial"])
    ap.add_argument("--round", dest="rnd", required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--min-seg-frames", type=int, default=1, help="drop subtask segments shorter than this many frames")
    ap.add_argument("--decode-workers", type=int, default=8)
    ap.add_argument("--image-writer-processes", type=int, default=3)
    ap.add_argument("--image-writer-threads", type=int, default=4)
    args = ap.parse_args()
    repo_id = args.repo_id or f"local/xarm_sim_{args.proto}_{args.rnd}"
    convert(
        args.proto,
        args.rnd,
        repo_id,
        args.root,
        args.max_episodes,
        args.min_seg_frames,
        args.decode_workers,
        args.image_writer_processes,
        args.image_writer_threads,
    )


if __name__ == "__main__":
    main()
