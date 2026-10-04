"""Template erase: show it one plate number, header or divider, and it erases every match.

Matching is normalised cross-correlation, done coarse-to-fine: a quick pass on a
shrunken page finds candidates, then each is confirmed at full resolution.
By default only the template's own ink shape is erased (grown by a few pixels),
so notation that happens to overlap the matched box survives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import analyze, proof
from .imageio import load_gray, save_gray


@dataclass
class Template:
    name: str
    pixels: np.ndarray  # uint8 grayscale


def template_from_page(page: np.ndarray, box: tuple[int, int, int, int], name: str = "box") -> Template:
    x, y, w, h = box
    crop = page[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError(f"template box {box} is outside the page")
    return Template(name, crop.copy())


def parse_box(text: str) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in text.split(","))
    return x, y, w, h


def _box_sums(a: np.ndarray, h: int, w: int) -> np.ndarray:
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    return ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]


def ncc_map(image: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation for every placement of tmpl fully inside image."""
    image = image.astype(np.float64)
    tmpl = tmpl.astype(np.float64)
    H, W = image.shape
    h, w = tmpl.shape
    if h > H or w > W:
        return np.zeros((0, 0))
    t = tmpl - tmpl.mean()
    tn = np.sqrt((t * t).sum())
    if tn == 0:
        return np.zeros((H - h + 1, W - w + 1))
    fshape = (H + h - 1, W + w - 1)
    corr = np.fft.irfft2(np.fft.rfft2(image, fshape) * np.fft.rfft2(t[::-1, ::-1], fshape), fshape)
    corr = corr[h - 1:H, w - 1:W]
    s1 = _box_sums(image, h, w)
    s2 = _box_sums(image * image, h, w)
    var = np.maximum(s2 - s1 * s1 / (h * w), 0)
    denom = np.sqrt(var) * tn
    out = np.zeros_like(corr)
    ok = denom > 1e-6 * tn * max(1.0, np.sqrt(h * w))
    out[ok] = corr[ok] / denom[ok]
    return out


def _shrink(a: np.ndarray, f: float) -> np.ndarray:
    if f >= 1:
        return a
    h, w = a.shape
    return np.asarray(Image.fromarray(a).resize((max(1, round(w * f)), max(1, round(h * f))), Image.BOX))


def find_matches(page: np.ndarray, tmpl: np.ndarray, threshold: float = 0.7,
                 band: tuple[float, float] | None = None, limit: int = 50):
    """Return [(x0, y0, x1, y1, score)] for every placement scoring at least threshold."""
    H, W = page.shape
    th, tw = tmpl.shape
    y_lo, y_hi = (0, H) if band is None else (int(band[0] * H), int(band[1] * H))
    area = page[y_lo:y_hi]
    if area.shape[0] < th or area.shape[1] < tw:
        return []
    f = min(1.0, max(12 / min(th, tw), 1600 / max(area.shape)))
    coarse = ncc_map(_shrink(area, f), _shrink(tmpl, f))
    if coarse.size == 0:
        return []
    sth, stw = max(1, round(th * f)), max(1, round(tw * f))
    r = int(np.ceil(2 / f)) + 2
    hits = []
    for _ in range(limit):
        cy, cx = np.unravel_index(int(np.argmax(coarse)), coarse.shape)
        if coarse[cy, cx] < threshold - 0.15:
            break
        coarse[max(0, cy - sth // 2):cy + sth // 2 + 1, max(0, cx - stw // 2):cx + stw // 2 + 1] = -1
        # confirm at full resolution
        x0, y0 = max(0, round(cx / f) - r), max(0, round(cy / f) - r)
        window = area[y0:y0 + th + 2 * r, x0:x0 + tw + 2 * r]
        fine = ncc_map(window, tmpl)
        if fine.size == 0:
            continue
        fy, fx = np.unravel_index(int(np.argmax(fine)), fine.shape)
        score = float(fine[fy, fx])
        if score < threshold:
            continue
        bx, by = x0 + fx, y0 + fy + y_lo
        if any(abs(bx - hx) < tw // 2 and abs(by - hy) < th // 2 for hx, hy, *_ in hits):
            continue
        hits.append((int(bx), int(by), int(bx + tw), int(by + th), score))
    return hits


def apply_erase(page: np.ndarray, tmpl: np.ndarray, hit, paper: int, mode: str = "ink", grow: int = 3) -> None:
    """Paint one match out of page, in place."""
    x0, y0, x1, y1, _ = hit
    H, W = page.shape
    if mode == "box":
        page[max(0, y0 - grow):min(H, y1 + grow), max(0, x0 - grow):min(W, x1 + grow)] = paper
        return
    mask = np.zeros((y1 - y0 + 2 * grow, x1 - x0 + 2 * grow), dtype=bool)
    mask[grow:grow + tmpl.shape[0], grow:grow + tmpl.shape[1]] = analyze.ink_mask(tmpl)
    mask = analyze.dilate(mask, grow)
    oy, ox = y0 - grow, x0 - grow
    sy0, sx0 = max(0, -oy), max(0, -ox)
    sy1, sx1 = mask.shape[0] - max(0, oy + mask.shape[0] - H), mask.shape[1] - max(0, ox + mask.shape[1] - W)
    mask = mask[sy0:sy1, sx0:sx1]
    region = page[oy + sy0:oy + sy1, ox + sx0:ox + sx1]
    region[mask] = np.maximum(region[mask], paper)


def erase(pages: list[Path], outdir, templates: list[Template], threshold: float = 0.7,
          band: tuple[float, float] | None = None, mode: str = "ink", grow: int = 3, log=print) -> dict:
    outdir = Path(outdir)
    (outdir / "proof").mkdir(parents=True, exist_ok=True)
    report, entries = [], []
    for p in pages:
        gray = load_gray(p)
        page = gray.copy()
        paper = analyze.paper_tone(gray, analyze.ink_mask(gray))
        page_hits = []
        for t in templates:
            hits = find_matches(gray, t.pixels, threshold, band)
            for h in hits:
                apply_erase(page, t.pixels, h, paper, mode, grow)
            page_hits += [{"template": t.name, "box": list(h[:4]), "score": round(h[4], 3)} for h in hits]
            log(f"erase {p.name}: {len(hits)} × {t.name}" + (f" (scores {', '.join(f'{h[4]:.2f}' for h in hits)})" if hits else ""))
        out_name = Path(p.name).with_suffix(".png").name
        save_gray(page, outdir / out_name)
        proof_name = f"{Path(p.name).stem}-proof.png"
        proof.draw_erase_proof(gray, [(*h["box"], h["score"]) for h in page_hits], outdir / "proof" / proof_name)
        entries.append((proof_name, f"{p.name} · {len(page_hits)} erased"))
        report.append({"source": p.name, "output": out_name, "hits": page_hits})
    manifest = {"tool": "scoretools.erase", "threshold": threshold, "mode": mode, "pages": report}
    (outdir / "erase.json").write_text(json.dumps(manifest, indent=2))
    proof.write_index(outdir / "proof", "Erase proof", entries,
                      "Red boxes are what was erased, with the match score under each. "
                      "Scores near the threshold are the ones to check; pages with no box had no match.")
    return manifest
