"""Page measurements shared by every tool: ink mask, staff metrics, straightening."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # 600-dpi scans are large on purpose


def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu's threshold, clamped so a blank or all-black page still gives a sane value."""
    step = max(1, int(np.sqrt(gray.size / 2_000_000)))  # a 2-megapixel sample is plenty
    hist = np.bincount(gray[::step, ::step].ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    levels = np.arange(256)
    w0 = np.cumsum(hist)
    w1 = total - w0
    m0 = np.cumsum(hist * levels)
    mean_all = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mean_all * w0 / total - m0) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    return int(np.clip(np.argmax(between), 64, 200))


def ink_mask(gray: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """True where the page has ink."""
    t = otsu_threshold(gray) if threshold is None else threshold
    return gray < t


def paper_tone(gray: np.ndarray, ink: np.ndarray) -> int:
    """Typical paper value, used to fill erased or padded areas so they don't show as patches."""
    paper = gray[~ink]
    if paper.size == 0:
        return 255
    return int(np.median(paper[:: max(1, paper.size // 200_000)]))


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Grow a boolean mask by r pixels (4-connected steps)."""
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _runs(col: np.ndarray):
    d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends


@dataclass
class StaffMetrics:
    line: int   # staff line thickness, px
    space: int  # distance between staff line centres, px
    found: bool  # False when the page gave no usable evidence and these are defaults


def staff_metrics(ink: np.ndarray, samples: int = 80) -> StaffMetrics:
    """Line thickness and staff space from the most common vertical black and white runs."""
    h, w = ink.shape
    cols = np.linspace(w * 0.1, w * 0.9, samples).astype(int)
    blacks, whites = [], []
    for c in cols:
        starts, ends = _runs(ink[:, c])
        if len(starts) < 2:
            continue
        blacks.append(ends - starts)
        whites.append(starts[1:] - ends[:-1])
    if not blacks:
        return StaffMetrics(line=max(1, h // 1500), space=max(8, h // 180), found=False)
    b = np.concatenate(blacks)
    wt = np.concatenate(whites)
    b = b[b < max(4, h // 100)]
    wt = wt[wt < max(8, h // 20)]
    if b.size == 0 or wt.size == 0:
        return StaffMetrics(line=max(1, h // 1500), space=max(8, h // 180), found=False)
    line = int(np.argmax(np.bincount(b)))
    gap = int(np.argmax(np.bincount(wt)))
    return StaffMetrics(line=max(1, line), space=max(4, gap + line), found=True)


def _profile_sharpness(img: Image.Image, angle: float) -> float:
    rot = np.asarray(img.rotate(angle, resample=Image.NEAREST, fillcolor=0), dtype=np.float32)
    prof = rot.sum(axis=1)
    return float(np.square(np.diff(prof)).sum())


def estimate_skew(ink: np.ndarray, max_angle: float = 3.0) -> float:
    """Angle in degrees (PIL convention, counter-clockwise) that levels the staff lines."""
    h, w = ink.shape
    full = Image.fromarray((ink * 255).astype(np.uint8))

    def shrink(px):
        s = min(1.0, px / max(h, w))
        return full.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BOX)

    coarse, fine = shrink(1000), shrink(2000)
    best = 0.0
    for img, step, span in ((coarse, 0.25, max_angle), (fine, 0.05, 0.25), (fine, 0.01, 0.05)):
        candidates = np.arange(best - span, best + span + 1e-9, step)
        # ties go to the smaller correction: never rotate a page that is already level
        best = max(candidates, key=lambda a: (_profile_sharpness(img, a), -abs(a)))
    return float(round(best, 3))


def straighten(gray: np.ndarray, min_angle: float = 0.05) -> tuple[np.ndarray, float]:
    """Rotate a crooked page level. Returns the new page and the angle applied."""
    ink = ink_mask(gray)
    angle = estimate_skew(ink)
    if abs(angle) < min_angle:
        return gray, 0.0
    fill = paper_tone(gray, ink)
    rotated = Image.fromarray(gray).rotate(angle, resample=Image.BICUBIC, fillcolor=fill)
    return np.asarray(rotated), angle
