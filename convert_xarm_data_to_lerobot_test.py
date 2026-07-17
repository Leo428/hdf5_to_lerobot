"""Unit tests for the dual-xArm -> LeRobot converter.

Covers the converter's pure, error-prone logic: the 6D rotation encoding (and its
cross-module consistency with the production ``transforms.sixd_to_matrix`` decoder used at
deploy time), the D6 gripper-target reconstruction (vs a literal transcription of the env
formula), the B1 off-by-one pose shift, the subtask/annotation parsing corner cases, and the
JPEG-decode RGB-no-swap claim.

This file lives under ``examples/`` (not in the default pytest ``testpaths``), so run it
explicitly:

    PYTHONPATH= uv run pytest examples/xarm/convert/convert_xarm_data_to_lerobot_test.py

The leading ``PYTHONPATH=`` drops the system ROS paths, which otherwise inject a broken
``launch_testing`` pytest plugin that breaks collection.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import openpi.transforms as _transforms

# The converter is a standalone script (examples/ has no package __init__), so load it by path.
_MODPATH = Path(__file__).resolve().parent / "convert_xarm_data_to_lerobot.py"
_spec = importlib.util.spec_from_file_location("convert_xarm_data_to_lerobot", _MODPATH)
cvt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cvt)


# --- 6D rotation encoding -------------------------------------------------------------------


def _random_wxyz(n: int, seed: int):
    """(rotation matrices (n,3,3), scalar-first wxyz quats (n,4)). Avoids as_quat(scalar_first=)."""
    r = Rotation.random(n, random_state=seed)
    q_wxyz = np.roll(r.as_quat(), 1, axis=-1).astype(np.float32)  # [x,y,z,w] -> [w,x,y,z]
    return r.as_matrix(), q_wxyz


def test_quats_wxyz_to_6d_matches_first_two_columns():
    mats, q_wxyz = _random_wxyz(16, seed=1)
    sixd = cvt.quats_wxyz_to_6d(q_wxyz)
    assert sixd.shape == (16, 6)
    assert sixd.dtype == np.float32
    np.testing.assert_allclose(sixd, np.concatenate([mats[:, :, 0], mats[:, :, 1]], axis=-1), atol=1e-5)


def test_quats_wxyz_to_6d_preserves_leading_shape():
    _, q_wxyz = _random_wxyz(12, seed=2)
    assert cvt.quats_wxyz_to_6d(q_wxyz.reshape(6, 2, 4)).shape == (6, 2, 6)


def test_quats_to_6d_roundtrips_through_transforms_sixd_to_matrix():
    # The deploy/output path reconstructs rotations with transforms.sixd_to_matrix; it must
    # invert the converter's encoding back to the original rotation. This is the contract that
    # ties the converter (encode) to the runtime (decode).
    mats, q_wxyz = _random_wxyz(16, seed=3)
    recon = _transforms.sixd_to_matrix(cvt.quats_wxyz_to_6d(q_wxyz))
    np.testing.assert_allclose(recon, mats, atol=1e-5)


def test_converter_and_transforms_sixd_to_matrix_agree():
    # The converter keeps its own (1-D) sixd_to_matrix for self-checks; it must agree with the
    # vectorized production one to the last bit of intent.
    _, q_wxyz = _random_wxyz(5, seed=4)
    for row in cvt.quats_wxyz_to_6d(q_wxyz):
        np.testing.assert_allclose(cvt.sixd_to_matrix(row), _transforms.sixd_to_matrix(row), atol=1e-6)


# --- Gripper-target reconstruction (D6) -----------------------------------------------------


def _env_gripper_target(action_grip: float, gripper_pos: float):
    """Literal transcription of rmp_env.step (lines 342-372). Returns None for the hold sentinel."""
    dg = -action_grip
    if abs(dg) > 0.05:
        return float(np.clip(gripper_pos + dg * 80, 80, 840))
    return None  # env sends 0.0 == "no command" == hold


def test_reconstruct_gripper_target_moving_and_b1_alignment():
    # action[i+1] drives obs[i]->obs[i+1] with base grip_cur[i] (the B1 alignment).
    raw = np.zeros((4, 14), np.float32)
    grip_cur = np.array([[500, 500], [400, 600], [300, 700], [200, 800]], np.float32)
    raw[1, cvt.GRIP_ACTION_IDX[0]] = 1.0  # left: target[0] = clip(500 - 1.0*80) = 420
    raw[2, cvt.GRIP_ACTION_IDX[1]] = -2.0  # right: target[1] = clip(600 + 2.0*80) = 760
    out = cvt.reconstruct_gripper_target(raw, grip_cur)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[0, 0], 420.0)
    np.testing.assert_allclose(out[1, 1], 760.0)


def test_reconstruct_gripper_target_deadzone_holds():
    raw = np.full((3, 14), 0.03, np.float32)  # everything inside the dead-zone
    grip_cur = np.array([[111, 222], [333, 444], [555, 666]], np.float32)
    out = cvt.reconstruct_gripper_target(raw, grip_cur)
    np.testing.assert_array_equal(out[:-1], grip_cur[:-1])  # held == current
    np.testing.assert_array_equal(out[-1], grip_cur[-1])  # last frame holds


def test_reconstruct_gripper_target_deadzone_boundary_is_strict():
    # rmp_env uses a strict ``abs(dg) > 0.05``: exactly 0.05 holds, just above moves.
    grip_cur = np.full((2, 2), 500.0, np.float32)
    raw = np.zeros((2, 14), np.float32)
    raw[1, cvt.GRIP_ACTION_IDX[0]] = 0.05  # == threshold -> hold
    raw[1, cvt.GRIP_ACTION_IDX[1]] = 0.0500001  # just above -> move
    out = cvt.reconstruct_gripper_target(raw, grip_cur)
    np.testing.assert_allclose(out[0, 0], 500.0)
    assert out[0, 1] < 500.0


def test_reconstruct_gripper_target_clip_saturation():
    grip_cur = np.array([[800.0, 100.0], [800.0, 100.0]], np.float32)
    raw = np.zeros((2, 14), np.float32)
    raw[1, cvt.GRIP_ACTION_IDX[0]] = 100.0  # 800 - 100*80 << GRIP_MIN
    raw[1, cvt.GRIP_ACTION_IDX[1]] = -100.0  # 100 + 100*80 >> GRIP_MAX
    out = cvt.reconstruct_gripper_target(raw, grip_cur)
    np.testing.assert_allclose(out[0, 0], cvt.GRIP_MIN)
    np.testing.assert_allclose(out[0, 1], cvt.GRIP_MAX)


def test_reconstruct_gripper_matches_env_formula_random():
    # Gold cross-check: 200 random steps x 2 arms against the literal env formula.
    rng = np.random.default_rng(7)
    n = 200
    raw = rng.uniform(-3, 3, (n, 14)).astype(np.float32)
    grip_cur = rng.uniform(80, 840, (n, 2)).astype(np.float32)
    out = cvt.reconstruct_gripper_target(raw, grip_cur)
    for a, idx in enumerate(cvt.GRIP_ACTION_IDX):
        for i in range(n - 1):
            env = _env_gripper_target(raw[i + 1, idx], grip_cur[i, a])  # B1: action[i+1], base i
            expected = env if env is not None else grip_cur[i, a]  # hold == current
            np.testing.assert_allclose(out[i, a], expected, atol=1e-4)
        np.testing.assert_allclose(out[n - 1, a], grip_cur[n - 1, a])  # last frame holds


# --- B1 pose shift --------------------------------------------------------------------------


def test_shift_pose_b1():
    pose = np.arange(5 * 2 * 9, dtype=np.float32).reshape(5, 2, 9)
    out = cvt.shift_pose_b1(pose)
    assert out.shape == pose.shape
    np.testing.assert_array_equal(out[:-1], pose[1:])  # action_pose[i] = state_pose[i+1]
    np.testing.assert_array_equal(out[-1], pose[-1])  # last frame holds (zero delta)


# --- Subtask annotation parsing -------------------------------------------------------------


def _write(tmp_path: Path, obj) -> Path:
    p = tmp_path / "audio_transcriptions.json"
    p.write_text(json.dumps(obj))
    return p


def test_parse_segments_missing_file(tmp_path):
    assert cvt.parse_subtask_segments(tmp_path / "nope.json", 100) == ([], 0)


def test_parse_segments_invalid_json(tmp_path):
    p = tmp_path / "a.json"
    p.write_text("{not valid json")
    assert cvt.parse_subtask_segments(p, 100) == ([], 0)


def test_parse_segments_non_list(tmp_path):
    assert cvt.parse_subtask_segments(_write(tmp_path, {"foo": "bar"}), 100) == ([], 0)


def test_parse_segments_drops_bad_case_insensitive_and_counts(tmp_path):
    anns = [
        {"start": {"step": 0}, "end": {"step": 10}, "transcription": "pick the cup"},
        {"start": {"step": 10}, "end": {"step": 20}, "transcription": "bad"},
        {"start": {"step": 20}, "end": {"step": 30}, "transcription": "BAD"},
        {"start": {"step": 30}, "end": {"step": 40}, "transcription": " Bad "},  # stripped -> "bad"
        {"start": {"step": 40}, "end": {"step": 50}, "transcription": "place it"},
    ]
    segs, n_bad = cvt.parse_subtask_segments(_write(tmp_path, anns), 100)
    assert n_bad == 3
    assert [s.task_text for s in segs] == ["pick the cup", "place it"]
    assert (segs[0].start_frame, segs[0].end_frame) == (0, 10)
    assert (segs[1].start_frame, segs[1].end_frame) == (40, 50)


def test_parse_segments_skips_empty_or_missing_transcription(tmp_path):
    anns = [
        {"start": {"step": 0}, "end": {"step": 10}, "transcription": ""},
        {"start": {"step": 10}, "end": {"step": 20}, "transcription": "   "},
        {"start": {"step": 20}, "end": {"step": 30}, "transcription": None},
        {"start": {"step": 30}, "end": {"step": 40}},  # no transcription key
        {"start": {"step": 40}, "end": {"step": 50}, "transcription": "real"},
    ]
    segs, n_bad = cvt.parse_subtask_segments(_write(tmp_path, anns), 100)
    assert n_bad == 0  # empty != bad
    assert [s.task_text for s in segs] == ["real"]


def test_parse_segments_clips_and_drops_empty_spans(tmp_path):
    anns = [
        {"start": {"step": -5}, "end": {"step": 1000}, "transcription": "spans all"},  # -> [0, 50)
        {"start": {"step": 30}, "end": {"step": 30}, "transcription": "zero len"},  # dropped
        {"start": {"step": 60}, "end": {"step": 70}, "transcription": "past end"},  # -> [50,50) dropped
    ]
    segs, _ = cvt.parse_subtask_segments(_write(tmp_path, anns), 50)
    assert len(segs) == 1
    assert (segs[0].task_text, segs[0].start_frame, segs[0].end_frame) == ("spans all", 0, 50)


def test_parse_segments_sorts_by_start(tmp_path):
    anns = [
        {"start": {"step": 40}, "end": {"step": 50}, "transcription": "third"},
        {"start": {"step": 0}, "end": {"step": 10}, "transcription": "first"},
        {"start": {"step": 20}, "end": {"step": 30}, "transcription": "second"},
    ]
    segs, _ = cvt.parse_subtask_segments(_write(tmp_path, anns), 100)
    assert [s.task_text for s in segs] == ["first", "second", "third"]


# --- JPEG decode ----------------------------------------------------------------------------


def test_decode_jpeg_to_rgb_roundtrip_no_swap():
    # The on-disk frames are BGR and were written with cv2.imencode (which consumes BGR), so a
    # plain PIL decode returns true RGB -- NO channel swap. Replicate that path end-to-end.
    cv2 = pytest.importorskip("cv2")
    rgb = np.zeros((64, 80, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 200, 100, 30  # distinct per-channel so a swap is obvious
    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1])  # store as the env does: imencode(BGR)
    assert ok
    out = cvt.decode_jpeg_to_rgb(buf.reshape(-1))
    assert out.shape == (cvt.IMAGE_SIZE, cvt.IMAGE_SIZE, 3)
    assert out.dtype == np.uint8
    cy, cx = cvt.IMAGE_SIZE // 2, cvt.IMAGE_SIZE // 2
    np.testing.assert_allclose(out[cy, cx], [200, 100, 30], atol=25)  # JPEG is lossy
