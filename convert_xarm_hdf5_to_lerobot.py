"""Convert OLD real dual-xArm teleop data (.hdf5 episodes) to a LeRobot v2 dataset.

This is the HDF5 sibling of ``examples/xarm/convert/convert_xarm_data_to_lerobot.py`` (the
real-npz converter). The robot, action space, gripper handling, 6D-rotation encoding, BGR
cameras and the B1 off-by-one are all IDENTICAL to that script -- this data is just stored
in an older HDF5 container instead of ``np.savez`` object arrays, and carries no per-subtask
annotations. So we keep the proven pose/gripper/offset math verbatim and change only:

  1. STORAGE = HDF5. One ``episode_N.hdf5`` per episode with per-step JPEG frames in
     ``obses/images/{left,right}/{top,wrist}`` (fixed-width uint8 ``{T, maxlen}`` -- the JPEG is
     zero-padded; PIL stops at the EOI marker), absolute target poses in
     ``obses/state/{side}/target_tcp_pose`` ``{T,7}`` = pos(3) + quat(4, scalar-first wxyz),
     achieved gripper in ``obses/state/{side}/gripper_pos``, and the 14-d world-frame command in
     ``actions/global_action``. We use ``right/top`` as the base view and DROP ``left/top`` (D10),
     exactly like the npz converter.

     B1 (off-by-one) -- VERIFIED on this data: ``target[i] - target[i-1] == global_action[i]`` to
     ~1e-6 (relative_action is the EE/body frame and does NOT satisfy this). So the command issued
     *from* ``obs[i]`` reaches target ``i+1``: ``action[i] = state_pose[i+1]`` (last frame holds).
     Gripper is reconstructed by replaying the env formula on ``action[i+1]`` (the gripper dims are
     frame-independent, so global/relative are identical there).

  2. ONE LeRobot episode per HDF5 episode (no ``audio_transcriptions.json`` exists here). Every
     frame carries the same fixed ``--prompt`` as ``task`` (the in-file ``metadata/task`` is an
     unreliable global label -- it reads ``real_xarms_shirt_hang_variations`` on every dataset,
     incl. lid -- so we ignore it).

  3. NEW per-frame boolean feature ``is_intervention``: True where frame index ``i`` appears in
     ``metadata/interventions`` (the DAgger human-takeover steps). Datasets without that key
     (clean / non-correction rounds) get all-False. The flag labels timestep ``i`` and is NOT
     B1-shifted.

Gripper (D6, unchanged from npz): ``target_gripper_pos`` is dead (constant), so we replay
``rmp_env.py`` step: ``dg = action[grip]``; if ``|dg| > 0.05`` the env commands
``clip(gripper_pos - dg*80, 80, 840)``, else the sentinel ``0.0`` == hold. State stores achieved
``gripper_pos``; action stores the reconstructed absolute target.

Example (run via the sibling uv project; HF_LEROBOT_HOME picks the output root):
  HF_LEROBOT_HOME=/data/group_data/rl/dexterous_robot_data/lerobot \
    uv run convert_xarm_hdf5_to_lerobot.py \
      --raw-dir /data/group_data/rl/dexterous_robot_data/real_hang_round1_0612_hdf5 \
      --repo-id real_hang_round1_0612
  # smoke test on two episodes:
  ... --repo-id smoke/real_hang_round1_0612 --max-episodes 2
"""

from concurrent.futures import ThreadPoolExecutor
import glob
import io
import os
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import tqdm
import tyro

IMAGE_SIZE = 224
FPS = 60
DEFAULT_PROMPT = "put shirt on hanger and place hanger on rack"

# Gripper env constants (must match rmp_env.py step()): dead-zone, units/cmd, real range.
GRIP_DEADZONE = 0.05
GRIP_SCALE = 80.0
GRIP_MIN, GRIP_MAX = 80.0, 840.0
GRIP_ACTION_IDX = (6, 13)  # raw 14-d action gripper dims: left, right (frame-independent).

# LeRobot camera feature name <- raw HDF5 camera group. right/top is the base view (D10);
# left/top is dropped.
CAMERA_SOURCES: dict[str, str] = {
    "base": "right/top",
    "left_wrist": "left/wrist",
    "right_wrist": "right/wrist",
}

# State / action vector layout (D8): per arm [pos(3), 6D(6), gripper(1)] = 10 -> 20, L then R.
ARMS = ("left", "right")
ARM_FIELDS = ("pos_x", "pos_y", "pos_z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")
VECTOR_NAMES = [f"{side}_{f}" for side in ARMS for f in ARM_FIELDS]
VECTOR_DIM = len(VECTOR_NAMES)  # 20


# ---------------------------------------------------------------------------------------------
# Reusable math (verbatim from convert_xarm_data_to_lerobot.py).
# ---------------------------------------------------------------------------------------------
def decode_jpeg_to_rgb(jpeg_arr: np.ndarray) -> np.ndarray:
    """Decode a cv2-encoded JPEG uint8 row -> (IMAGE_SIZE, IMAGE_SIZE, 3) true RGB.

    The camera frames are BGR (RealSense bgr8 / ZED BGRA[:3]), but ``cv2.imencode`` consumed that
    BGR convention when writing the JPEG, so a standard decoder (PIL) already returns true RGB --
    NO channel swap (matches the npz converter's verified result). The row may be zero-padded past
    the JPEG EOI marker; PIL ignores the trailing bytes. Resize stretches to a square (D11).
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
    """Gram-Schmidt inverse of ``quats_wxyz_to_6d`` -> (3, 3). Used for self-checks."""
    a1, a2 = sixd[:3], sixd[3:6]
    b1 = a1 / np.linalg.norm(a1)
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / np.linalg.norm(a2p)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # columns = basis vectors


def reconstruct_gripper_target(raw_actions: np.ndarray, grip_cur: np.ndarray) -> np.ndarray:
    """Replay the env gripper formula (D6) to recover the absolute commanded gripper target.

    For frame i the target is commanded by ``action[i+1]`` (B1), with base ``grip_cur[i]``. The env
    computes ``dg = action[grip]``; if ``|dg| > 0.05`` it commands ``clip(gripper_pos - dg*80, 80,
    840)``, else the sentinel ``0.0`` == hold. We store the absolute target (clip when moving, the
    current ``grip_cur`` when holding / on the last frame).
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
    """B1 off-by-one shift: ``action[i] = state_pose[i+1]`` for i in [0, N-2]; last frame holds."""
    return np.concatenate([pose6d[1:], pose6d[-1:]], axis=0)


# ---------------------------------------------------------------------------------------------
# HDF5 episode loader.
# ---------------------------------------------------------------------------------------------
def load_episode(h5_path: Path, decode_workers: int = 16):
    """Return (images_per_cam, state (T,20), action (T,20), is_intervention (T,), T) or None.

    state[i]  = [pos(3), 6D(6), gripper_pos(1)] per arm (L,R): absolute world target + achieved grip.
    action[i] = [pos(3), 6D(6), gripper_target(1)] per arm: pose = state_pose[i+1] (B1), gripper =
                env-formula reconstructed absolute command (D6).
    """
    with h5py.File(h5_path, "r") as h:
        T = int(h["actions/global_action"].shape[0])
        if T < 2:
            return None

        pos = np.zeros((T, len(ARMS), 3), dtype=np.float32)
        quat = np.zeros((T, len(ARMS), 4), dtype=np.float32)  # wxyz
        grip_cur = np.zeros((T, len(ARMS)), dtype=np.float32)
        for a, side in enumerate(ARMS):
            tp = np.asarray(h[f"obses/state/{side}/target_tcp_pose"], dtype=np.float32)  # (T,7)
            pos[:, a] = tp[:, :3]
            quat[:, a] = tp[:, 3:7]
            grip_cur[:, a] = np.asarray(h[f"obses/state/{side}/gripper_pos"], dtype=np.float32).ravel()

        # Gripper dims are frame-independent; global_action satisfies the B1 pose relation, so use it.
        raw_actions = np.asarray(h["actions/global_action"], dtype=np.float32)  # (T,14)

        # Per-frame intervention flag (absent on clean rounds -> all False). Labels timestep i; not shifted.
        is_interv = np.zeros(T, dtype=bool)
        if "metadata/interventions" in h:
            iv = np.asarray(h["metadata/interventions"]).ravel().astype(np.int64)
            iv = iv[(iv >= 0) & (iv < T)]
            is_interv[iv] = True

        # Read the (zero-padded) JPEG rows for the 3 kept cameras as whole arrays (1 contiguous
        # NFS read each) before leaving the h5 context.
        jpeg_rows: dict[str, np.ndarray] = {}
        for feat, src in CAMERA_SOURCES.items():
            jpeg_rows[feat] = np.asarray(h[f"obses/images/{src}"][:])  # (T, maxlen) uint8

    pose6d = np.concatenate([pos, quats_wxyz_to_6d(quat)], axis=-1)  # (T, 2, 9) absolute, world
    grip_tgt = reconstruct_gripper_target(raw_actions, grip_cur)  # (T, 2)
    pose_next = shift_pose_b1(pose6d)  # (T, 2, 9)
    state = np.concatenate(
        [np.concatenate([pose6d[:, a], grip_cur[:, a, None]], axis=1) for a in range(len(ARMS))], axis=1
    ).astype(np.float32)
    action = np.concatenate(
        [np.concatenate([pose_next[:, a], grip_tgt[:, a, None]], axis=1) for a in range(len(ARMS))], axis=1
    ).astype(np.float32)
    assert state.shape == (T, VECTOR_DIM), state.shape
    assert action.shape == (T, VECTOR_DIM), action.shape

    images_per_cam: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        for feat, rows in jpeg_rows.items():
            frames = list(pool.map(decode_jpeg_to_rgb, [rows[i] for i in range(T)]))
            images_per_cam[feat] = np.stack(frames)  # (T, 224, 224, 3) uint8 RGB

    return images_per_cam, state, action, is_interv, T


def create_empty_dataset(repo_id: str) -> LeRobotDataset:
    features = {
        "observation.state": {"dtype": "float32", "shape": (VECTOR_DIM,), "names": [VECTOR_NAMES]},
        "action": {"dtype": "float32", "shape": (VECTOR_DIM,), "names": [VECTOR_NAMES]},
        "is_intervention": {"dtype": "bool", "shape": (1,), "names": ["is_intervention"]},
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


def _self_check(state: np.ndarray, action: np.ndarray, quat0: np.ndarray, is_interv: np.ndarray, ep_name: str) -> None:
    """One-time numeric sanity print: B1 per-step motion, 6D round-trip, gripper recon, interv%."""
    dl = np.linalg.norm(action[:-1, 0:3] - state[:-1, 0:3], axis=1)
    dr = np.linalg.norm(action[:-1, 10:13] - state[:-1, 10:13], axis=1)
    print(f"  [self-check] {ep_name}: frames={len(state)}  interventions={is_interv.mean() * 100:.1f}%")
    print(f"    per-step |dpos| median  L={np.median(dl):.4f} m  R={np.median(dr):.4f} m (expect mm-cm)")
    rt = sixd_to_matrix(state[0, 3:9])
    src = Rotation.from_quat(quat0, scalar_first=True).as_matrix()
    print(f"    6D round-trip max|dR|={np.max(np.abs(rt - src)):.2e}  det={np.linalg.det(rt):+.4f} (expect ~0, +1)")
    for a, idx in ((0, 9), (1, 19)):
        tgt, cur, nxt = action[:-1, idx], state[:-1, idx], state[1:, idx]
        moved = np.abs(tgt - cur) > 1.0
        if moved.any():
            err = np.abs(tgt[moved] - nxt[moved])
            print(
                f"    gripper {ARMS[a]}: cmd-change {moved.mean() * 100:4.1f}% of steps, "
                f"|target-achieved_next| median={np.median(err):.1f} (range {GRIP_MIN:.0f}-{GRIP_MAX:.0f})"
            )
        else:
            print(f"    gripper {ARMS[a]}: no commanded changes (stayed in dead-zone)")


def episode_files(raw_dir: Path) -> list[Path]:
    fs = glob.glob(str(raw_dir / "episode_*.hdf5"))
    return [Path(p) for p in sorted(fs, key=lambda p: int(Path(p).stem.split("_")[-1]))]


def convert(
    raw_dir: Path,
    repo_id: str,
    prompt: str = DEFAULT_PROMPT,
    max_episodes: int | None = None,
    *,
    push_to_hub: bool = False,
):
    """Convert one dir of dual-xArm ``episode_*.hdf5`` to a LeRobot v2 dataset (1 episode each)."""
    raw_dir = Path(raw_dir)
    files = episode_files(raw_dir)
    if max_episodes is not None:
        files = files[:max_episodes]
    print(f"[{repo_id}] {len(files)} episodes from {raw_dir}  ->  {HF_LEROBOT_HOME / repo_id}")

    dataset = create_empty_dataset(repo_id)
    episodes_written = 0
    frames_written = 0
    interv_frames = 0
    skipped: list[tuple[str, str]] = []
    selfcheck_done = False

    for h5_path in tqdm.tqdm(files, desc=f"Converting {repo_id}"):
        try:
            loaded = load_episode(h5_path)
        except Exception as e:  # noqa: BLE001 -- robustness: skip a bad file, keep going.
            skipped.append((h5_path.name, f"load error: {type(e).__name__}: {e}"))
            continue
        if loaded is None:
            skipped.append((h5_path.name, "too few frames"))
            continue
        images_per_cam, state, action, is_interv, num_frames = loaded

        for cam, imgs in images_per_cam.items():
            assert imgs.shape[0] == num_frames, f"{h5_path.name}: {cam} {imgs.shape[0]} != {num_frames}"

        if not selfcheck_done:
            with h5py.File(h5_path, "r") as h:
                q0 = np.asarray(h["obses/state/left/target_tcp_pose"][0, 3:7], dtype=np.float32)
            _self_check(state, action, q0, is_interv, h5_path.name)
            selfcheck_done = True

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "is_intervention": np.array([is_interv[i]], dtype=bool),
                "task": prompt,
            }
            for cam in CAMERA_SOURCES:
                frame[f"observation.images.{cam}"] = images_per_cam[cam][i]
            dataset.add_frame(frame)
        dataset.save_episode()
        episodes_written += 1
        frames_written += num_frames
        interv_frames += int(is_interv.sum())

    dataset.stop_image_writer()

    print(f"\n=== {repo_id} summary ===")
    print(f"Episodes written: {episodes_written} / {len(files)}   frames: {frames_written}")
    print(f"Intervention frames: {interv_frames} ({100 * interv_frames / max(frames_written, 1):.1f}%)")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for name, reason in skipped[:20]:
            print(f"  - {name}: {reason}")
    print(f"Dataset: {HF_LEROBOT_HOME / repo_id}")

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(convert)
