#!/usr/bin/env python3
"""
Wall Sun/Shade Tracker
=====================
Tracks sun exposure across three wall segments from time-lapse photographs.
Assumes fixed ISO/exposure so absolute pixel brightness is meaningful.

Usage:
    python wall_tracker.py <image_directory> [--threshold 128] [--recalculate]

Steps:
  1. Loads all JPGs and sorts by EXIF timestamp.
  2. Prompts for interactive corner selection of 3 wall segments
     (saved to wall_corners.json for reuse).
  3. Calibrates the L-channel threshold: click one sunny pixel and one
     shaded pixel on the reference frame; midpoint is used as the threshold.
     (saved to wall_threshold.json for reuse, or pass --threshold explicitly)
  4. Warps each segment to a rectified rectangle; classifies each pixel as
     sunny (L >= threshold) or shaded (L < threshold).  GPU-accelerated.
  5. Opens an interactive viewer: three graphs + rectified images with overlay.
     Click graphs or use slider/arrow keys to scrub through frames.
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
import json
import sys
import argparse
import queue
import threading
from datetime import datetime
from typing import Any

import torch
import kornia
from tqdm import tqdm

# ── Configuration ───────────────────────────────────────────────────────────

RECT_SIZE          = (300, 200)
DISPLAY_MAX_W      = 1400
SEGMENT_NAMES      = ["Left", "Center", "Right"]
CORNERS_FILENAME   = "wall_corners.json"
THRESHOLD_FILENAME = "wall_threshold.json"
RESULTS_FILENAME   = "wall_results_cache.npz"

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_BATCH_SIZE = 32


# ── Text helpers ─────────────────────────────────────────────────────────────

def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw text with a black outline for readability on any background."""
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX,
               font_scale, (0, 0, 0), thickness + 4)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness)


# ── EXIF helpers ────────────────────────────────────────────────────────────

def get_exif_datetime(path: Path) -> datetime | None:
    """Extract capture datetime from EXIF."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        ifd = exif.get_ifd(0x8769)
        dt_str = ifd.get(36867) or ifd.get(36868) or exif.get(306)
        if dt_str:
            return datetime.strptime(str(dt_str), "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"  EXIF warning ({path.name}): {e}")
    return None


def load_images(directory: Path) -> list[tuple[Path, datetime]]:
    """Return list of (Path, datetime) sorted by capture time."""
    exts = {'.jpg', '.jpeg'}
    files = [f for f in directory.iterdir() if f.suffix.lower() in exts]

    frames: list[tuple[Path, datetime]] = []
    skipped = 0
    for f in files:
        dt = get_exif_datetime(f)
        if dt is None:
            skipped += 1
            continue
        frames.append((f, dt))

    frames.sort(key=lambda x: x[1])
    if skipped:
        print(f"  Skipped {skipped} files without EXIF timestamps.")
    if frames:
        print(f"  {len(frames)} images: "
              f"{frames[0][1]:%Y-%m-%d %H:%M} → {frames[-1][1]:%H:%M}")
    return frames


# ── Corner selection ────────────────────────────────────────────────────────

Corners = list[list[tuple[int, int]]]


class CornerSelector:
    """OpenCV GUI for clicking four corners on each of three wall segments."""

    COLORS = [(0, 255, 0), (0, 220, 255), (255, 180, 0)]  # BGR per segment

    def __init__(self, image_path: Path) -> None:
        self.orig = cv.imread(str(image_path))
        if self.orig is None:
            raise RuntimeError(f"Cannot read {image_path}")
        h, w = self.orig.shape[:2]
        self.scale = min(DISPLAY_MAX_W / w, 1.0)
        self.display_base = cv.resize(self.orig, None,
                                      fx=self.scale, fy=self.scale)
        self.pts: list[tuple[int, int]] = []
        self.segments: list[list[tuple[int, int]]] = []
        self.seg_idx = 0

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        if event == cv.EVENT_LBUTTONDOWN and len(self.pts) < 4:
            self.pts.append((x, y))
        elif event == cv.EVENT_RBUTTONDOWN and self.pts:
            self.pts.pop()

    def _render(self) -> np.ndarray:
        img = self.display_base.copy()
        for i, seg in enumerate(self.segments):
            cv.polylines(img, [np.array(seg)], True, self.COLORS[i], 2)
            for p in seg:
                cv.circle(img, p, 5, self.COLORS[i], -1)
        c = self.COLORS[self.seg_idx] if self.seg_idx < 3 else (255, 255, 255)
        for j, p in enumerate(self.pts):
            cv.circle(img, p, 6, c, -1)
            put_text(img, str(j + 1), (p[0] + 8, p[1] - 8), 0.45, c, 2)
        if len(self.pts) > 1:
            cv.polylines(img, [np.array(self.pts)], len(self.pts) == 4, c, 2)
        name = SEGMENT_NAMES[self.seg_idx] if self.seg_idx < 3 else "?"
        put_text(img,
                 f"Segment {self.seg_idx+1}/3 ({name}): "
                 f"click 4 corners TL TR BR BL  [{len(self.pts)}/4]",
                 (10, 28), 0.6, (255, 255, 255), 2)
        put_text(img, "Right-click=undo  SPACE=confirm  Q=quit",
                 (10, 54), 0.45, (180, 180, 180), 1)
        return img

    def run(self) -> Corners:
        """Returns list of 3 segments, each a list of 4 (x,y) in original coords."""
        win = "Select Wall Corners"
        cv.namedWindow(win, cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(win, 1200, 800)
        cv.setMouseCallback(win, self._on_mouse)

        while self.seg_idx < 3:
            cv.imshow(win, self._render())
            key = cv.waitKey(30) & 0xFF
            if key == ord('q'):
                cv.destroyAllWindows()
                sys.exit(0)
            if key == ord(' ') and len(self.pts) == 4:
                self.segments.append(list(self.pts))
                self.pts = []
                self.seg_idx += 1

        cv.destroyAllWindows()
        return [[(int(x / self.scale), int(y / self.scale)) for x, y in seg]
                for seg in self.segments]


def save_corners(corners: Corners, path: Path) -> None:
    with open(path, 'w') as f:
        json.dump(corners, f, indent=2)


def load_corners(path: Path) -> Corners:
    with open(path) as f:
        return json.load(f)


# ── Image processing ───────────────────────────────────────────────────────

def warp_segment(image: np.ndarray, corners: list[tuple[int, int]]) -> np.ndarray:
    """Perspective-warp a quad to RECT_SIZE rectangle."""
    src = np.array(corners, dtype=np.float32)
    w, h = RECT_SIZE
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype=np.float32)
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(image, M, RECT_SIZE)


# ── GPU processing ──────────────────────────────────────────────────────────

def _bgr_to_L_batch(bgr_batch: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) uint8 BGR → (B,H,W) float32 L channel [0,255]."""
    rgb = bgr_batch.float().div(255.0)[:, [2, 1, 0]]
    lab = kornia.color.rgb_to_lab(rgb)   # L in [0,100]
    return lab[:, 0] * (255.0 / 100.0)  # → [0,255]


SegResult = dict[str, Any]
FrameResults = list[SegResult]


def process_frames_gpu(
    frames: list[tuple[Path, datetime]],
    corners: Corners,
    threshold: int,
) -> list[FrameResults]:
    """
    GPU-accelerated batch processing via producer-consumer.
    Returns all_results: list of N lists, each with 3 dicts {sunny_pct, mask}.
    """
    N = len(frames)
    all_results: list[FrameResults | None] = [None] * N
    dummy_mask = np.zeros((RECT_SIZE[1], RECT_SIZE[0]), np.uint8)

    q: queue.Queue[Any] = queue.Queue(maxsize=GPU_BATCH_SIZE * 2)
    _DONE = object()

    def producer() -> None:
        for i, (path, _) in enumerate(frames):
            img = cv.imread(str(path))
            if img is None:
                q.put((i, None))
            else:
                rects = [warp_segment(img, sc) for sc in corners]
                q.put((i, rects))
        q.put(_DONE)

    threading.Thread(target=producer, daemon=True).start()

    def flush(batch_idxs: list[int], batch_rects: list[np.ndarray]) -> None:
        arr = np.stack([r.transpose(2, 0, 1) for r in batch_rects])
        L = _bgr_to_L_batch(torch.from_numpy(arr).to(DEVICE))
        masks = (L >= threshold).cpu().numpy().astype(np.uint8) * 255
        for k, frame_idx in enumerate(batch_idxs):
            all_results[frame_idx] = [
                {
                    "sunny_pct": 100.0 * float(np.count_nonzero(masks[k * 3 + s]))
                                 / masks[k * 3 + s].size,
                    "mask": masks[k * 3 + s],
                }
                for s in range(3)
            ]

    batch_idxs: list[int] = []
    batch_rects: list[np.ndarray] = []

    print(f"Processing {N} frames on {DEVICE}…")
    with tqdm(total=N, unit="img") as bar:
        while True:
            item = q.get()
            if item is _DONE:
                break
            i, rects = item  # type: ignore[misc]
            bar.update(1)
            if rects is None:
                all_results[i] = [  # type: ignore[index]
                    {"sunny_pct": 0.0, "mask": dummy_mask.copy()}
                    for _ in range(3)
                ]
                continue
            batch_idxs.append(i)  # type: ignore[arg-type]
            batch_rects.extend(rects)  # type: ignore[arg-type]
            if len(batch_idxs) >= GPU_BATCH_SIZE:
                flush(batch_idxs, batch_rects)
                batch_idxs, batch_rects = [], []

    if batch_idxs:
        flush(batch_idxs, batch_rects)

    return all_results  # type: ignore[return-value]


# ── Results cache ────────────────────────────────────────────────────────────

def save_results_cache(
    cache_path: Path,
    frames: list[tuple[Path, datetime]],
    all_results: list[FrameResults],
    threshold: int,
    corners: Corners,
) -> None:
    N = len(frames)
    paths_arr = np.array([str(f[0]) for f in frames])
    sunny = np.array(
        [[all_results[i][s]["sunny_pct"] for s in range(3)] for i in range(N)],
        dtype=np.float32,
    )
    masks = np.stack(
        [[all_results[i][s]["mask"] for s in range(3)] for i in range(N)]
    )  # (N, 3, H, W)
    np.savez_compressed(
        cache_path,
        paths=paths_arr,
        sunny_pct=sunny,
        masks=masks,
        threshold=np.array([threshold]),
        corners=np.array(corners, dtype=np.int32),
    )
    print(f"  Results cached to {cache_path.name}")


def load_results_cache(
    cache_path: Path,
    frames: list[tuple[Path, datetime]],
    threshold: int,
    corners: Corners,
) -> list[FrameResults] | None:
    """Returns all_results or None on cache miss / invalidation."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
    except Exception:
        return None
    if int(data["threshold"][0]) != threshold:
        return None
    if not np.array_equal(data["corners"], np.array(corners, dtype=np.int32)):
        return None
    current_paths = [str(f[0]) for f in frames]
    if list(data["paths"]) != current_paths:
        return None
    N = len(frames)
    sunny: np.ndarray = data["sunny_pct"]
    masks: np.ndarray = data["masks"]
    return [
        [{"sunny_pct": float(sunny[i, s]), "mask": masks[i, s]} for s in range(3)]
        for i in range(N)
    ]


# ── Bimodality scoring ──────────────────────────────────────────────────────

def bimodality_score(
    rects: list[np.ndarray],
) -> tuple[float, int, np.ndarray]:
    L_all = np.concatenate([
        cv.cvtColor(r, cv.COLOR_BGR2LAB)[:, :, 0].ravel()
        for r in rects
    ])
    hist = np.bincount(L_all, minlength=256).astype(np.float64)
    total = hist.sum()
    best_var = 0.0
    best_t = 0
    best_mu0 = best_mu1 = best_w0 = 0.0
    w0 = 0.0
    sum0 = 0.0
    total_sum = np.dot(np.arange(256, dtype=np.float64), hist)
    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * hist[t]
        mu0 = sum0 / w0
        mu1 = (total_sum - sum0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var = var
            best_t = t
            best_mu0, best_mu1 = mu0, mu1
            best_w0 = w0

    sep = abs(best_mu1 - best_mu0)
    frac = best_w0 / total
    balance = min(frac, 1.0 - frac)
    return sep * balance, best_t, hist


def find_best_calibration_frame(
    frames: list[tuple[Path, datetime]],
    corners: Corners,
    sample_every: int = 5,
) -> tuple[int, float, int, np.ndarray]:
    print("Scanning frames for strongest sun/shade bimodality…")
    best_score = -1.0
    best_idx = 0
    best_t = 128
    best_hist: np.ndarray = np.zeros(256, dtype=np.float64)

    indices = list(range(0, len(frames), sample_every))
    if (len(frames) - 1) not in indices:
        indices.append(len(frames) - 1)

    for i in indices:
        img = cv.imread(str(frames[i][0]))
        if img is None:
            continue
        rects = [warp_segment(img, sc) for sc in corners]
        score, t, hist = bimodality_score(rects)
        if score > best_score:
            best_score, best_idx, best_t, best_hist = score, i, t, hist

    path, dt = frames[best_idx]
    print(f"  Best frame: #{best_idx+1}  {dt:%H:%M}  "
          f"({path.name})  score={best_score:.1f}  Otsu-L={best_t}")
    return best_idx, best_score, best_t, best_hist


# ── Threshold calibration ───────────────────────────────────────────────────

class ThresholdCalibrator:
    """
    Click one sunny pixel and one shaded pixel on the reference image.
    Midpoint L value becomes the threshold.
    """

    HIST_H   = 120
    HIST_PAD = 8

    def __init__(
        self,
        image_path: Path,
        scale: float,
        otsu_t: int,
        hist_counts: np.ndarray,
    ) -> None:
        self.orig = cv.imread(str(image_path))
        if self.orig is None:
            raise RuntimeError(f"Cannot read {image_path}")
        self.scale = scale
        self.display = cv.resize(self.orig, None, fx=scale, fy=scale)
        self.lab = cv.cvtColor(self.orig, cv.COLOR_BGR2LAB)
        self.clicks: list[tuple[str, int, int, int]] = []
        self.otsu_t = otsu_t
        self.hist = hist_counts
        self._hist_strip = self._make_hist_strip(self.display.shape[1])

    def _make_hist_strip(self, width: int) -> np.ndarray:
        strip = np.zeros((self.HIST_H, width, 3), np.uint8)
        strip[:] = (30, 30, 30)
        max_count = float(self.hist.max()) if self.hist.max() > 0 else 1.0
        bar_h = self.HIST_H - self.HIST_PAD * 2
        for val in range(256):
            h = int(self.hist[val] / max_count * bar_h)
            if h == 0:
                continue
            x0 = int(val * width / 256)
            x1 = min(int((val + 1) * width / 256), width - 1)
            cv.rectangle(strip,
                         (x0, self.HIST_H - self.HIST_PAD - h),
                         (x1, self.HIST_H - self.HIST_PAD),
                         (160, 160, 160), -1)
        ox = int(self.otsu_t * width / 256)
        cv.line(strip, (ox, self.HIST_PAD), (ox, self.HIST_H - self.HIST_PAD),
                (80, 220, 80), 2)
        put_text(strip, f"Otsu={self.otsu_t}", (ox + 4, self.HIST_PAD + 14),
                 0.38, (80, 220, 80), 1)
        put_text(strip, "L  (0=dark, 255=bright)",
                 (6, self.HIST_H - 4), 0.35, (160, 160, 160), 1)
        return strip

    def _render(self) -> np.ndarray:
        img = self.display.copy()
        prompts = ["Click a SUNNY pixel", "Click a SHADED pixel",
                   "Press SPACE to confirm"]
        done = len(self.clicks)
        put_text(img, prompts[min(done, 2)], (10, 28), 0.7, (255, 255, 255), 2)
        colors: list[tuple[int, int, int]] = [(0, 200, 255), (255, 100, 0)]
        for i, (label, ox, oy, L_val) in enumerate(self.clicks):
            dx, dy = int(ox * self.scale), int(oy * self.scale)
            cv.circle(img, (dx, dy), 8, colors[i], 2)
            put_text(img, f"{label} L={L_val}", (dx + 12, dy - 6),
                     0.45, colors[i], 1)

        strip = self._hist_strip.copy()
        w = strip.shape[1]
        for i, (_, _ox, _oy, L_val) in enumerate(self.clicks):
            lx = int(L_val * w / 256)
            cv.line(strip, (lx, 2), (lx, self.HIST_H - 2), colors[i], 2)

        if done == 2:
            threshold = (self.clicks[0][3] + self.clicks[1][3]) // 2
            put_text(img, f"threshold = {threshold}  (SPACE to confirm)",
                     (10, 54), 0.55, (100, 255, 100), 2)
            tx = int(threshold * w / 256)
            cv.line(strip, (tx, 2), (tx, self.HIST_H - 2), (100, 255, 100), 2)
            put_text(strip, f"thresh={threshold}", (tx + 4, 28),
                     0.38, (100, 255, 100), 1)

        return np.vstack([img, strip])

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        if event != cv.EVENT_LBUTTONDOWN:
            return
        if len(self.clicks) >= 2:
            return
        if y >= self.display.shape[0]:
            return
        assert self.orig is not None
        ox = int(np.clip(x / self.scale, 0, self.orig.shape[1] - 1))
        oy = int(np.clip(y / self.scale, 0, self.orig.shape[0] - 1))
        L_val = int(self.lab[oy, ox, 0])
        label = "sunny" if len(self.clicks) == 0 else "shaded"
        self.clicks.append((label, ox, oy, L_val))
        print(f"  Clicked {label} pixel at ({ox},{oy})  L={L_val}")

    def run(self) -> int:
        """Returns the midpoint L threshold (0-255)."""
        win = "Threshold Calibration  (histogram below)"
        cv.namedWindow(win, cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(win, 1200, 800)
        cv.setMouseCallback(win, self._on_mouse)
        while True:
            cv.imshow(win, self._render())
            key = cv.waitKey(30) & 0xFF
            if key == ord('q'):
                cv.destroyAllWindows()
                sys.exit(0)
            if key == ord(' ') and len(self.clicks) == 2:
                break
            if key == ord('r'):
                self.clicks.clear()
        cv.destroyAllWindows()
        threshold = (self.clicks[0][3] + self.clicks[1][3]) // 2
        print(f"  Threshold set to {threshold}")
        return threshold


# ── Interactive viewer ──────────────────────────────────────────────────────

class Viewer:
    """Matplotlib GUI: three time-series graphs + rectified images + scrubber."""

    GRAPH_COLORS = ['#1976D2', '#F57C00', '#388E3C']
    SUN_TINT   = np.array([0, 180, 255], dtype=np.float32)
    SHADE_TINT = np.array([200, 100, 0],  dtype=np.float32)

    def __init__(
        self,
        frames: list[tuple[Path, datetime]],
        corners: Corners,
        results: list[FrameResults],
    ) -> None:
        self.frames  = frames
        self.corners = corners
        self.results = results
        self.N       = len(frames)
        self.idx     = 0
        self._warp_cache: dict[int, list[np.ndarray]] = {}

        self.datetimes = [f[1] for f in frames]
        self.mpl_times = mdates.date2num(self.datetimes)
        self.multiday = self.datetimes[0].date() != self.datetimes[-1].date()
        self.sunny = [
            [float(results[i][s]["sunny_pct"]) for i in range(self.N)]
            for s in range(3)
        ]
        self._build()

    def _get_rectified(self, idx: int) -> list[np.ndarray]:
        if idx not in self._warp_cache:
            if len(self._warp_cache) > 10:
                del self._warp_cache[next(iter(self._warp_cache))]
            path = self.frames[idx][0]
            img = cv.imread(str(path))
            if img is None:
                dummy = np.zeros((RECT_SIZE[1], RECT_SIZE[0], 3), np.uint8)
                self._warp_cache[idx] = [dummy, dummy, dummy]
            else:
                self._warp_cache[idx] = [warp_segment(img, sc) for sc in self.corners]
        return self._warp_cache[idx]

    def _build(self) -> None:
        self.fig = plt.figure(figsize=(15, 9), num="Wall Sun/Shade Tracker")
        gs = GridSpec(2, 3, figure=self.fig,
                      height_ratios=[1.2, 1], hspace=0.32, wspace=0.28,
                      left=0.06, right=0.97, top=0.93, bottom=0.14)

        self.ax_g = []
        self.vlines = []
        locator = mdates.AutoDateLocator()
        if self.multiday:
            formatter: mdates.DateFormatter | mdates.ConciseDateFormatter = (
                mdates.ConciseDateFormatter(locator)
            )
        else:
            formatter = mdates.DateFormatter('%H:%M')

        for s in range(3):
            ax = self.fig.add_subplot(gs[0, s])
            ax.plot(self.mpl_times, self.sunny[s],
                    color=self.GRAPH_COLORS[s], lw=1.4)
            ax.set_ylim(-5, 105)
            ax.set_ylabel("Sunny %")
            ax.set_title(f"{SEGMENT_NAMES[s]} segment", fontsize=11)
            ax.grid(True, alpha=0.25)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            for label in ax.get_xticklabels():
                label.set_rotation(30)
                label.set_horizontalalignment('right')
            vl = ax.axvline(self.mpl_times[0], color='#E53935',
                            lw=1.2, ls='--', zorder=5)
            self.vlines.append(vl)
            self.ax_g.append(ax)

        self.ax_im = []
        self.im_objs: list[object] = []
        for s in range(3):
            ax = self.fig.add_subplot(gs[1, s])
            ax.set_xticks([])
            ax.set_yticks([])
            self.ax_im.append(ax)
            self.im_objs.append(None)

        sl_ax = self.fig.add_axes((0.10, 0.04, 0.78, 0.025))
        self.slider = Slider(sl_ax, 'Frame', 0, self.N - 1,
                             valinit=0, valstep=1, valfmt='%d')
        self.slider.on_changed(self._on_slider)

        self.ts_label = self.fig.text(0.50, 0.08, '', ha='center',
                                      fontsize=10, family='monospace')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',   self._on_key)

        self._refresh(0)

    def _on_slider(self, val: float) -> None:
        self._refresh(int(val))

    def _on_click(self, ev: Any) -> None:
        for ax in self.ax_g:
            if ev.inaxes is ax and ev.xdata is not None:
                idx = int(np.argmin(np.abs(np.array(self.mpl_times) - ev.xdata)))
                self.slider.set_val(idx)
                return

    def _on_key(self, ev: Any) -> None:
        key = ev.key
        if key == 'right':
            self.slider.set_val(min(self.idx + 1, self.N - 1))
        elif key == 'left':
            self.slider.set_val(max(self.idx - 1, 0))
        elif key == 'shift+right':
            self.slider.set_val(min(self.idx + 10, self.N - 1))
        elif key == 'shift+left':
            self.slider.set_val(max(self.idx - 10, 0))

    def _overlay(self, rect_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        out = rect_bgr.astype(np.float32)
        if mask is not None:
            out[mask == 255] = out[mask == 255] * 0.55 + self.SUN_TINT   * 0.45
            out[mask == 0]   = out[mask == 0]   * 0.55 + self.SHADE_TINT * 0.45
        return np.clip(out, 0, 255).astype(np.uint8)

    def _refresh(self, idx: int) -> None:
        self.idx = idx
        for vl in self.vlines:
            vl.set_xdata([self.mpl_times[idx]])

        rects = self._get_rectified(idx)
        for s in range(3):
            r = self.results[idx][s]
            mask: np.ndarray | None = r["mask"]  # type: ignore[assignment]
            vis = self._overlay(rects[s], mask)
            vis_rgb = cv.cvtColor(vis, cv.COLOR_BGR2RGB)

            if self.im_objs[s] is None:
                self.im_objs[s] = self.ax_im[s].imshow(vis_rgb)
            else:
                self.im_objs[s].set_data(vis_rgb)  # type: ignore[union-attr]

            pct = float(r["sunny_pct"])
            self.ax_im[s].set_title(
                f"{SEGMENT_NAMES[s]}: {pct:.0f}% sun", fontsize=9)

        dt = self.datetimes[idx]
        self.ts_label.set_text(
            f"{dt:%Y-%m-%d %H:%M:%S}   {self.frames[idx][0].name}   "
            f"[{idx+1}/{self.N}]")
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Track sun/shade on wall segments from time-lapse JPGs.")
    ap.add_argument("directory", help="Directory containing JPG images")
    ap.add_argument("--threshold", type=int, default=None,
                    help="Absolute L-channel threshold (0-255). "
                         "If omitted, a calibration UI lets you click "
                         "one sunny and one shaded pixel.")
    ap.add_argument("--reselect", action="store_true",
                    help="Force re-selection of corners even if saved file exists")
    ap.add_argument("--recalculate", action="store_true",
                    help="Ignore cached results and reprocess all frames")
    args = ap.parse_args()

    img_dir = Path(args.directory)
    if not img_dir.is_dir():
        print(f"Error: {img_dir} is not a directory")
        sys.exit(1)

    print("Loading images…")
    frames = load_images(img_dir)
    if len(frames) < 3:
        print("Need at least 3 images with EXIF timestamps.")
        sys.exit(1)

    corners_path   = img_dir / CORNERS_FILENAME
    threshold_path = img_dir / THRESHOLD_FILENAME
    results_path   = img_dir / RESULTS_FILENAME
    corners: Corners | None = None
    threshold: int | None = args.threshold

    ref_idx  = len(frames) // 2
    ref_path = frames[ref_idx][0]
    ref_img  = cv.imread(str(ref_path))
    if ref_img is None:
        print(f"Error: cannot read reference frame {ref_path}")
        sys.exit(1)
    _h, w = ref_img.shape[:2]
    scale = min(DISPLAY_MAX_W / w, 1.0)

    if corners_path.exists() and not args.reselect:
        ans = input(f"Saved corners found ({corners_path.name}). "
                    f"Reuse? [Y/n] ").strip().lower()
        if ans != 'n':
            corners = load_corners(corners_path)
            print("  Reusing saved corners.")

    if corners is None:
        print(f"Opening corner selector on frame {ref_idx+1} ({ref_path.name})…")
        print("  Click TL → TR → BR → BL for each segment, then SPACE.")
        sel = CornerSelector(ref_path)
        corners = sel.run()
        save_corners(corners, corners_path)
        print(f"  Corners saved to {corners_path.name}")

    if threshold is None:
        if threshold_path.exists():
            ans = input(f"Saved threshold found ({threshold_path.name}). "
                        f"Reuse? [Y/n] ").strip().lower()
            if ans != 'n':
                with open(threshold_path) as f:
                    threshold = json.load(f)['threshold']
                print(f"  Reusing saved threshold: {threshold}")

    if threshold is None:
        cal_idx, _score, otsu_t, hist = find_best_calibration_frame(
            frames, corners, sample_every=5)
        cal_path = frames[cal_idx][0]
        print(f"Opening threshold calibrator on frame {cal_idx+1} ({cal_path.name})…")
        print("  Green line = Otsu suggestion.  Click SUNNY then SHADED, "
              "SPACE to confirm, R to reset.")
        cal = ThresholdCalibrator(cal_path, scale, otsu_t, hist)
        threshold = cal.run()
        with open(threshold_path, 'w') as f:
            json.dump({'threshold': threshold}, f)
        print(f"  Threshold saved to {threshold_path.name}")

    print(f"Using L-channel threshold: {threshold}")

    all_results: list[FrameResults] | None = None
    if not args.recalculate:
        all_results = load_results_cache(results_path, frames, threshold, corners)
        if all_results is not None:
            print(f"  Loaded cached results from {results_path.name}")

    if all_results is None:
        all_results = process_frames_gpu(frames, corners, threshold)
        save_results_cache(results_path, frames, all_results, threshold, corners)

    print("Launching viewer…")
    print("  Click graph to jump │ ←/→ step │ Shift + ←/→ skip 10")
    viewer = Viewer(frames, corners, all_results)
    viewer.show()


if __name__ == '__main__':
    main()
