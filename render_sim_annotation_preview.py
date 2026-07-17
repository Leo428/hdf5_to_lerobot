"""Render sim packing episodes with the Gemini subtask annotation overlaid.

A QA tool for the sim->LeRobot port: it draws each episode's three camera views side
by side, with the currently-active subtask ``summary`` (verbatim Gemini text), the
``action_type``/``object``/``confidence`` metadata, and a colour-coded segment timeline,
so we can eyeball whether the segmentation (esp. ``perturbation_event`` / ``reorganize`` /
``recover`` / ``transition``) makes sense to train a Markovian VLA on.

It does NOT touch the converter or change any annotation text -- it only visualises what
is already recorded (raw ``summary``). Run it with the sim repo's venv (has cv2/h5py)::

    /home/huzheyuan/kshitiz2/RAC/dual_xarms/dual_xarms_sim/.venv/bin/python \
        examples/xarm/render_sim_annotation_preview.py \
        --ep baseline:round1:SP-01 --ep adversarial:round4:SP-01

Data model (verified): annotation segments tile ``[0, n_frames)`` in obs-frame index, so
frame ``i`` is active in the segment with ``start_step <= i <= end_step`` -- no B1 shift for
display. Colour: frames are mujoco-RGB cv2-JPEG-encoded, so ``cv2.imdecode`` returns true
RGB; we swap to BGR (``[..., ::-1]``) for cv2 drawing/writing (visually confirmed).
"""
# ruff: noqa: N806  (image-dimension locals W/H/H_TL/H_CAP read clearest uppercase)

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

DEFAULT_ROOT = Path("/home/huzheyuan/kshitiz2/RAC/dual_xarms/dual_xarms_sim/data/new_data")
DEFAULT_OUT = Path("/media/huzheyuan/data0/sim_annotation_preview")
VIEWS = ("right/top", "left/wrist", "right/wrist")

# action_type -> BGR colour for the timeline / current-phase strip.
ATYPE_COLOR = {
    "pick_and_place": (60, 180, 60),
    "reorganize": (40, 200, 230),
    "recover": (40, 140, 255),
    "remove_clutter": (220, 130, 40),
    "transition": (150, 150, 150),
    "perturbation_event": (40, 40, 230),
}
OTHER_COLOR = (230, 230, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX
IMG_H = 224  # all views resized to this height before hstack


def resolve_paths(root: Path, proto: str, rnd: str, goal: str) -> tuple[Path, Path]:
    hdf5 = root / proto / rnd / "demos" / proto / goal / "episode_0.hdf5"
    ann = root / "annotations" / proto / rnd / "pipeline_a" / proto / goal / f"{proto}__{goal}__episode_0.json"
    return hdf5, ann


def load_segments(ann_path: Path) -> list[dict]:
    d = json.loads(ann_path.read_text())
    return sorted(d.get("segments", []), key=lambda s: s.get("start_step", 0))


def seg_at(segs: list[dict], i: int) -> tuple[int, dict | None]:
    for k, s in enumerate(segs):
        if s.get("start_step", 0) <= i <= s.get("end_step", -1):
            return k, s
    return -1, None


def decode_bgr(raw) -> np.ndarray:
    """JPEG bytes -> true-BGR uint8 image (ready for cv2 drawing / VideoWriter)."""
    arr = np.frombuffer(np.asarray(raw, dtype=np.uint8), dtype=np.uint8)
    rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # cv2.imdecode of a cv2-encoded mujoco-RGB frame = true RGB
    return np.ascontiguousarray(rgb[:, :, ::-1])  # -> true BGR


def fit_h(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    w = round(img.shape[1] * h / img.shape[0])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def wrap_text(text: str, max_w: int, scale: float, thick: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split():
            trial = word if not cur else cur + " " + word
            (tw, _), _ = cv2.getTextSize(trial, FONT, scale, thick)
            if tw <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def draw_timeline(canvas: np.ndarray, y0: int, h: int, w: int, segs: list[dict], cur: int, n_frames: int) -> None:
    for s in segs:
        a = int(s.get("start_step", 0) / max(1, n_frames) * w)
        b = int((s.get("end_step", 0) + 1) / max(1, n_frames) * w)
        color = ATYPE_COLOR.get(s.get("action_type"), OTHER_COLOR)
        cv2.rectangle(canvas, (a, y0), (max(a + 1, b), y0 + h), color, -1)
    x = int(cur / max(1, n_frames) * w)
    cv2.line(canvas, (x, y0 - 2), (x, y0 + h + 2), (255, 255, 255), 2)


def draw_legend(canvas: np.ndarray, x0: int, y0: int) -> None:
    x = x0
    for name, color in ATYPE_COLOR.items():
        cv2.rectangle(canvas, (x, y0 - 10), (x + 16, y0 + 2), color, -1)
        cv2.putText(canvas, name, (x + 20, y0), FONT, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
        x += 26 + cv2.getTextSize(name, FONT, 0.4, 1)[0][0]


def render_episode(hdf5: Path, ann: Path, out_path: Path, stride: int, out_fps: float) -> dict:
    segs = load_segments(ann)
    meta = json.loads(ann.read_text())
    goal = meta.get("goal_id", hdf5.parent.name)
    proto = meta.get("protocol", "?")
    h = h5py.File(hdf5, "r")
    n_frames = h["obses/images/right/top"].shape[0]

    # probe one composed frame to fix canvas size
    def compose(i: int) -> np.ndarray:
        tiles = [fit_h(decode_bgr(h[f"obses/images/{v}"][i]), IMG_H) for v in VIEWS]
        for v, t in zip(VIEWS, tiles, strict=True):
            cv2.putText(t, v, (4, 16), FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        return np.hstack(tiles)

    strip = compose(0)
    W = strip.shape[1]
    H_TL, H_CAP = 26, 176
    H = IMG_H + H_TL + H_CAP
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter failed to open for {out_path}")

    written = 0
    for i in range(0, n_frames, stride):
        canvas = np.zeros((H, W, 3), np.uint8)
        canvas[:IMG_H] = compose(i)
        k, s = seg_at(segs, i)
        color = ATYPE_COLOR.get(s.get("action_type") if s else None, OTHER_COLOR)
        # current-phase strip just under the images
        cv2.rectangle(canvas, (0, IMG_H), (W, IMG_H + 6), color, -1)
        draw_timeline(canvas, IMG_H + H_TL - 12, 12, W, segs, i, n_frames)
        # caption panel
        cy = IMG_H + H_TL + 22
        atype = s.get("action_type", "?") if s else "(none)"
        conf = s.get("confidence", "?") if s else "?"
        obj = s.get("object", "?") if s else "?"
        a = s.get("start_step", "?") if s else "?"
        b = s.get("end_step", "?") if s else "?"
        head = f"{proto}/{goal}  seg {k + 1}/{len(segs)}  [{atype} conf={conf} obj={obj}]  steps {a}-{b}  frame {i}/{n_frames - 1} t={i / 60:.1f}s"
        cv2.putText(canvas, head, (8, cy), FONT, 0.46, (180, 220, 255), 1, cv2.LINE_AA)
        summary = s.get("summary", "(no segment covers this frame)") if s else "(no segment covers this frame)"
        for j, line in enumerate(wrap_text(summary, W - 16, 0.62, 1)):
            cv2.putText(canvas, line, (8, cy + 26 + j * 26), FONT, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        draw_legend(canvas, 8, H - 8)
        writer.write(canvas)
        written += 1
    writer.release()
    h.close()
    return {
        "goal": goal,
        "proto": proto,
        "n_frames": n_frames,
        "n_segs": len(segs),
        "written": written,
        "out": str(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="new_data root")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ep", action="append", default=[], help="proto:round:goal (repeatable)")
    ap.add_argument("--hdf5", type=Path, help="explicit episode hdf5 (with --ann)")
    ap.add_argument("--ann", type=Path, help="explicit annotation json")
    ap.add_argument("--stride", type=int, default=3, help="render every Nth frame (60Hz native)")
    ap.add_argument("--fps", type=float, default=20.0, help="output video fps")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, Path, str]] = []
    if args.hdf5 and args.ann:
        jobs.append((args.hdf5, args.ann, args.hdf5.parent.name))
    for ep in args.ep:
        proto, rnd, goal = ep.split(":")
        hdf5, ann = resolve_paths(args.root, proto, rnd, goal)
        jobs.append((hdf5, ann, f"{proto}_{rnd}_{goal}"))

    if not jobs:
        ap.error("give --ep proto:round:goal (repeatable) or --hdf5 + --ann")

    for hdf5, ann, name in jobs:
        if not hdf5.exists():
            print(f"[skip] missing hdf5: {hdf5}")
            continue
        if not ann.exists():
            print(f"[skip] missing annotation: {ann}")
            continue
        out = args.out_dir / f"{name}.mp4"
        print(f"[render] {name} ...", flush=True)
        info = render_episode(hdf5, ann, out, args.stride, args.fps)
        print(
            f"   {info['proto']}/{info['goal']}: {info['n_segs']} segs, {info['n_frames']} frames "
            f"-> {info['written']} written  {info['out']}"
        )


if __name__ == "__main__":
    main()
