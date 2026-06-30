"""Convert OLD sim dual-xArm *double-insert* teleop (.hdf5 episodes) to a LeRobot v2 dataset.

The double-insert sim is from an earlier project than the packing sim, but it runs the **same
mujoco dual-xArm env** (verified: reconstructed mocap[0] == the packing HOME exactly, and the
packing ``reconstruct_mocap`` constants reproduce the recorded pose to cm-level). So the 20-d
Cartesian state/action, the mocap reconstruction, the gripper handling and the R/B colour swap are
all IDENTICAL to ``convert_xarm_sim_data_to_lerobot.py``. This script is therefore that converter's
*pose/gripper/colour math* wrapped in the *structure* of the real-HDF5 converter
(``convert_xarm_hdf5_to_lerobot.py``): one LeRobot episode per file, a fixed ``--prompt``, and the
per-frame ``is_intervention`` flag. The only data-path differences vs the packing sim:

  1. WORLD ACTION SOURCE. The packing sim stores the world-frame command in
     ``obses/state/{side}/og_action``; this older data does NOT have ``og_action`` -- it has
     ``actions/global_action`` (T,14) instead. VERIFIED that ``global_action`` is the world-frame
     delta (achieved ``tcp_pose[k]-tcp_pose[k-1] == global_action[k]`` to sub-mm; ``relative_action``
     is the EE/body frame and diverges on integration). It already has the (L[pos,eul,grip],
     R[pos,eul,grip]) 14-d layout ``reconstruct_mocap`` expects, so we feed it in directly.

  2. NO ANNOTATIONS. The packing converter splits each demo into subtask LeRobot episodes from
     ``human_pass_annotations``; none exist here, so each ``episode_N.hdf5`` becomes ONE LeRobot
     episode whose every frame carries the same ``--prompt`` as ``task``. ``metadata/task`` is the
     usual unreliable global label (``real_xarms_shirt_hang_variations`` on every file) -- ignored.

  3. is_intervention (per-frame bool, labels timestep i, NOT B1-shifted). Three cases by dataset:
       * key ``metadata/interventions`` present (DAgger round/correction rollouts): True only at the
         recorded human-takeover step indices.
       * key absent + ``full_success`` in the dataset name (autonomous policy rollouts that
         succeeded): all False.
       * key absent otherwise (pure human teleop -- base ``0226`` and the per-person ``jasmine`` /
         ``riya`` / ``robyn`` / ``zheyuan_0508`` collects): all True -- every frame is human-driven.
     The absent-key fill is decided from the dataset name in ``convert`` (``absent_fill``).

Encoding (identical to the packing sim converter):
  * state[i]  = per arm [mocap_pos(3), 6D(mocap_quat)(6), gripper_setpoint(1)], L then R -> 20-d.
    The pose is the RECONSTRUCTED mocap *target* (the commanded IK anchor), NOT the recorded
    ``tcp_pose`` -- which in this data is serialised scalar-LAST (xyzw) and is never read for the
    state. ``reconstruct_mocap`` integrates from the wxyz HOME and emits wxyz, so the 6D rep is
    scalar-first throughout (mujoco-consistent), matching the packing converter.
  * action[i] = state[i+1] (B1 off-by-one; last frame holds). Gripper is the [0,1] setpoint
    (``gripper_pos``), B1-shifted with the pose -- no env-formula reconstruction (sim, not the real
    80-840 arm).
  * 3 cameras: base=right/top, left_wrist=left/wrist, right_wrist=right/wrist (drop left/top).
    mujoco renders RGB, ``cv2.imencode`` stores it, so PIL-decode then R/B swap recovers true RGB.

Example (run via the Babel ``hdf5_to_lerobot`` uv project; HF_LEROBOT_HOME picks the output root):
  HF_LEROBOT_HOME=/scratch/$USER/lerobot \
    uv run convert_xarm_sim_double_insert_to_lerobot.py \
      --raw-dir /data/group_data/rl/dexterous_robot_data/sim_double_insert_round1_hdf5 \
      --repo-id huzheyuan/sim_double_insert_round1 \
      --prompt "insert both pegs into the sockets"
  # smoke test on two episodes:
  ... --repo-id smoke/sim_double_insert_round1 --max-episodes 2
  # convert + push the dataset public to the Hub:
  ... --repo-id huzheyuan/sim_double_insert_round1 --push-to-hub
"""

from concurrent.futures import ThreadPoolExecutor
import glob
import io
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
DEFAULT_PROMPT = "insert both pegs into the sockets"

# --- mujoco dual-xArm mocap recurrence constants (same env as the packing sim; VERIFIED:
#     reconstruct_mocap[0] == HOME exactly and tracks the recorded tcp_pose to cm-level). ---
LEFT_HOME = np.asarray([-0.35, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0], dtype=np.float64)  # xyz + quat wxyz
RIGHT_HOME = np.asarray([0.35, 0.4, 0.2, 0, 0.7071068, -0.7071068, 0], dtype=np.float64)
LEFT_BOUNDS = (np.asarray([-0.7, 0.2, 0.0]), np.asarray([0.1, 0.6, 0.3]))
RIGHT_BOUNDS = (np.asarray([-0.1, 0.2, 0.0]), np.asarray([0.7, 0.6, 0.3]))
MAX_LIN = 1.0 / FPS  # _MAX_LINEAR_VELOCITY / control_freq
MAX_ANG = (np.pi / 3) / FPS  # _MAX_ANGULAR_VELOCITY / control_freq
# global_action 14-d layout: L[pos(0:3), euler(3:6), grip(6)], R[pos(7:10), euler(10:13), grip(13)].
ARM_OG = ((slice(0, 3), slice(3, 6)), (slice(7, 10), slice(10, 13)))

# LeRobot camera feature name <- HDF5 view key. right/top is the base view; left/top is dropped.
CAMERA_SOURCES: dict[str, str] = {
    "base": "right/top",
    "left_wrist": "left/wrist",
    "right_wrist": "right/wrist",
}

# State / action vector layout: per arm [pos(3), 6D(6), gripper(1)] = 10 -> 20, L then R.
ARMS = ("left", "right")
ARM_FIELDS = ("pos_x", "pos_y", "pos_z", "r00", "r10", "r20", "r01", "r11", "r21", "gripper")
VECTOR_NAMES = [f"{side}_{f}" for side in ARMS for f in ARM_FIELDS]
VECTOR_DIM = len(VECTOR_NAMES)  # 20


# ---------------------------------------------------------------------------------------------
# Reusable math (verbatim from convert_xarm_sim_data_to_lerobot.py).
# ---------------------------------------------------------------------------------------------
def _limit_rows(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Row-wise ``limit_offset_norm``: scale each (N,3) row to ``max_norm`` if it exceeds it."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.where(n > max_norm, max_norm / np.maximum(n, 1e-12), 1.0)


def reconstruct_mocap(og14: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """world-frame action (N,14) -> (pos (N,2,3), quat (N,2,4) scalar-first wxyz) IK mocap targets.

    Deterministic replay of the env step: per arm ``pos = clip(pos + limit(d_pos, MAX_LIN), bounds)``
    and ``R = R(limit(d_eul, MAX_ANG)) @ R``, HOME-initialised. ``mocap[k]`` is the target AFTER step
    k (aligns with ``obses[k]``). Fed ``actions/global_action`` here (``og_action`` is absent).
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
        p = home[:3].copy()
        for k in range(n):  # position: clip makes it sequential
            p = np.clip(p + d_pos[k], lo, hi)
            pos[k, a] = p
        m = Rotation.from_quat(home[3:7], scalar_first=True).as_matrix()
        mats = np.empty((n, 3, 3), dtype=np.float64)
        for k in range(n):  # rotation: m[k] = delta_r[k] @ m[k-1]
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

    The frames are mujoco-RGB written by ``cv2.imencode``, which swaps R/B vs a standard PIL decode
    -- so PIL-decode then swap ``[..., ::-1]`` recovers true RGB (opposite of the real BGR cameras,
    which need no swap). Stretch-resize to a square with BICUBIC (swap commutes with resize).
    """
    im = Image.open(io.BytesIO(np.asarray(jpeg_arr, dtype=np.uint8).tobytes()))
    im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])  # swap R/B -> true RGB


def shift_b1(arr: np.ndarray) -> np.ndarray:
    """B1 off-by-one: ``action[i] = state[i+1]`` (next absolute target); last frame holds."""
    return np.concatenate([arr[1:], arr[-1:]], axis=0)


# ---------------------------------------------------------------------------------------------
# HDF5 episode loader.
# ---------------------------------------------------------------------------------------------
def load_episode(h5_path: Path, absent_fill: bool, decode_workers: int = 16):
    """Return (images_per_cam, state (T,20), action (T,20), is_intervention (T,), T) or None.

    state[i]  = per arm [mocap_pos(3), 6D(mocap_quat)(6), gripper_setpoint(1)], L then R.
    action[i] = state[i+1] (B1 shift; last holds) -- next absolute world target + next gripper.
    ``absent_fill`` = the is_intervention value for episodes WITHOUT ``metadata/interventions``
    (True for pure human teleop, False for autonomous full_success; see module docstring / convert).
    """
    with h5py.File(h5_path, "r") as h:
        og14 = np.asarray(h["actions/global_action"][:], dtype=np.float64)  # (T,14) world-frame
        t = og14.shape[0]
        if t < 2:
            return None

        grip = np.stack(
            [np.asarray(h[f"obses/state/{side}/gripper_pos"][:]).reshape(-1) for side in ARMS], axis=1
        ).astype(np.float32)  # (T,2) in [0,1]

        # Per-frame intervention flag (labels timestep i; not shifted). Key present -> True only at the
        # recorded human-takeover indices; key absent -> the caller's ``absent_fill`` (teleop=True, full_success=False).
        if "metadata/interventions" in h:
            is_interv = np.zeros(t, dtype=bool)
            iv = np.asarray(h["metadata/interventions"]).ravel().astype(np.int64)
            is_interv[iv[(iv >= 0) & (iv < t)]] = True
        else:
            is_interv = np.full(t, absent_fill, dtype=bool)

        jpeg_rows = {feat: np.asarray(h[f"obses/images/{src}"][:]) for feat, src in CAMERA_SOURCES.items()}

    pos, quat = reconstruct_mocap(og14)  # (T,2,3), (T,2,4) wxyz
    pose6d = np.concatenate([pos, quats_wxyz_to_6d(quat)], axis=-1).astype(np.float32)  # (T,2,9)
    state = np.concatenate(
        [np.concatenate([pose6d[:, a], grip[:, a, None]], axis=1) for a in range(len(ARMS))], axis=1
    ).astype(np.float32)  # (T,20)
    action = shift_b1(state)  # (T,20); action[i]=state[i+1] (pose + gripper)
    assert state.shape == (t, VECTOR_DIM), state.shape
    assert action.shape == (t, VECTOR_DIM), action.shape

    images_per_cam: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
        for feat, rows in jpeg_rows.items():
            frames = list(pool.map(decode_jpeg_to_rgb, [rows[i] for i in range(t)]))
            images_per_cam[feat] = np.stack(frames)  # (T,224,224,3) RGB
    return images_per_cam, state, action, is_interv, t


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


def _self_check(state: np.ndarray, h5_path: Path, is_interv: np.ndarray, ep_name: str) -> None:
    """One-time numeric sanity: B1 per-step motion, 6D det, mocap[0]~HOME, recon-vs-achieved, interv%."""
    action = shift_b1(state)
    dl = np.linalg.norm(action[:-1, 0:3] - state[:-1, 0:3], axis=1)
    dr = np.linalg.norm(action[:-1, 10:13] - state[:-1, 10:13], axis=1)
    pos = state[:, [0, 1, 2, 10, 11, 12]].reshape(len(state), 2, 3)  # L,R mocap pos
    print(f"  [self-check] {ep_name}: frames={len(state)}  interventions={is_interv.mean() * 100:.1f}%")
    print(f"    per-step |dpos| median  L={np.median(dl):.4f} m  R={np.median(dr):.4f} m (expect mm-cm, < {MAX_LIN:.4f})")
    print(f"    6D det(frame0 L)={np.linalg.det(sixd_to_matrix(state[0, 3:9])):+.4f} (expect +1)")
    print(f"    mocap[0] L pos={np.round(pos[0, 0], 4)} vs HOME={np.round(LEFT_HOME[:3], 4)} (expect ~equal)")
    # env-free cross-check: reconstructed mocap target should track the recorded achieved tcp_pose (cm).
    with h5py.File(h5_path, "r") as h:
        tcp = {a: np.asarray(h[f"obses/state/{ARMS[a]}/tcp_pose"][:, :3], dtype=np.float64) for a in range(len(ARMS))}
    for a in range(len(ARMS)):
        e = np.linalg.norm(pos[:, a] - tcp[a], axis=1)
        print(
            f"    recon-mocap vs achieved tcp_pose {ARMS[a]}: median={np.median(e) * 100:.2f}cm "
            f"max={e.max() * 100:.1f}cm (expect cm-level)"
        )


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
    """Convert one dir of sim double-insert ``episode_*.hdf5`` to a LeRobot v2 dataset (1 episode each)."""
    raw_dir = Path(raw_dir)
    files = episode_files(raw_dir)
    if max_episodes is not None:
        files = files[:max_episodes]
    # is_intervention fill for episodes lacking metadata/interventions: pure human teleop collects are
    # ALL intervention (True); autonomous full_success rollouts are NONE (False). Keyed datasets ignore this.
    absent_fill = "full_success" not in raw_dir.name
    print(f"[{repo_id}] {len(files)} episodes from {raw_dir}  ->  {HF_LEROBOT_HOME / repo_id}")
    print(f'  prompt="{prompt}"')
    print(f"  is_intervention (no-key episodes) -> {absent_fill}  ('full_success' in name -> False, else teleop -> True)")

    dataset = create_empty_dataset(repo_id)
    episodes_written = frames_written = interv_frames = 0
    skipped: list[tuple[str, str]] = []
    selfcheck_done = False

    for h5_path in tqdm.tqdm(files, desc=f"Converting {repo_id}"):
        try:
            loaded = load_episode(h5_path, absent_fill)
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
            _self_check(state, h5_path, is_interv, h5_path.name)
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
        dataset.push_to_hub(private=False)


if __name__ == "__main__":
    tyro.cli(convert)
