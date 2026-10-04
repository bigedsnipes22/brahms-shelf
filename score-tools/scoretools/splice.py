"""Splice: pack cut systems onto any page shape.

Every system is drawn at one shared scale (the widest system fills the column
width), so staff size stays even across the whole score. Page breaks are chosen
together rather than greedily, to spread leftover space evenly instead of
leaving a crowded page next to a near-empty one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from . import analyze, proof

PRESETS: dict[str, dict] = {
    # print, 300 dpi
    "letter": dict(size=(2550, 3300), dpi=300, margin=150),
    "a4": dict(size=(2480, 3508), dpi=300, margin=150),
    # iPad Air / iPad 10th gen screen, 264 ppi
    "ipad": dict(size=(1640, 2360), dpi=264, margin=40),
    "ipad-landscape": dict(size=(2360, 1640), dpi=264, margin=40),
    "ipad-2up": dict(size=(2360, 1640), dpi=264, margin=40, columns=2),
    # 16:9 video frames: centred, no stretching of gaps
    "video": dict(size=(1920, 1080), dpi=144, margin=60, valign="center", halign="center"),
    "video-4k": dict(size=(3840, 2160), dpi=288, margin=120, valign="center", halign="center"),
}


@dataclass
class SpliceOptions:
    size: tuple[int, int] = (1640, 2360)
    dpi: int = 264
    margin: int = 40
    columns: int = 1
    column_gap: int | None = None   # px; default 2x margin
    gap: float = 1.0                # min space between systems, in staff spaces
    max_gap: float = 5.0            # cap when justifying, in staff spaces
    valign: str = "justify"         # top | justify | center
    halign: str = "left"            # left | center
    scale: float | None = None      # output px per source px; default fits the widest system
    extras: str = "first"           # first | all | none
    whiten: bool = True             # lift scan paper tone to white

    @classmethod
    def preset(cls, name: str, **overrides) -> "SpliceOptions":
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}")
        base = replace(cls(), **PRESETS[name])
        return replace(base, **{k: v for k, v in overrides.items() if v is not None})


def parse_size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def pick_items(manifest: dict, extras: str) -> list[dict]:
    items = manifest["items"]
    if extras == "all":
        return list(items)
    if extras == "none":
        return [it for it in items if it["kind"] == "system"]
    first_system = next((i for i, it in enumerate(items) if it["kind"] == "system"), len(items))
    return [it for i, it in enumerate(items) if it["kind"] == "system" or i < first_system]


def break_columns(heights: list[float], column_h: float, gap: float) -> list[tuple[int, int]]:
    """Split items into columns, minimising squared leftover space (the last column is free)."""
    n = len(heights)
    best = [0.0] + [float("inf")] * n
    prev = [0] * (n + 1)
    for j in range(1, n + 1):
        used = -gap
        for i in range(j - 1, -1, -1):
            used += heights[i] + gap
            if used > column_h and i < j - 1:
                break
            slack = max(0.0, column_h - used)
            cost = best[i] + (0.0 if j == n else slack * slack)
            if cost < best[j]:
                best[j], prev[j] = cost, i
    cols, j = [], n
    while j > 0:
        cols.append((prev[j], j))
        j = prev[j]
    return cols[::-1]


def _whiten(img: np.ndarray) -> np.ndarray:
    paper = analyze.paper_tone(img, analyze.ink_mask(img))
    if paper >= 250:
        return img
    return np.clip(img.astype(np.float32) * (255.0 / max(paper, 1)), 0, 255).astype(np.uint8)


def splice(cut_dir, outdir, opts: SpliceOptions, pdf_path=None, log=print) -> dict:
    cut_dir, outdir = Path(cut_dir), Path(outdir)
    manifest = json.loads((cut_dir / "cut.json").read_text())
    items = pick_items(manifest, opts.extras)
    if not items:
        raise ValueError(f"no systems in {cut_dir / 'cut.json'}")

    W, H = opts.size
    C = max(1, opts.columns)
    col_gap = opts.column_gap if opts.column_gap is not None else 2 * opts.margin
    cw = (W - 2 * opts.margin - (C - 1) * col_gap) / C
    ch = H - 2 * opts.margin

    sizes = []
    for it in items:
        with Image.open(cut_dir / it["file"]) as im:
            sizes.append(im.size)
    widest = max(w for (w, _), it in zip(sizes, items) if it["kind"] == "system") if any(
        it["kind"] == "system" for it in items) else max(w for w, _ in sizes)
    s = opts.scale or cw / widest
    space = float(np.median([it.get("staff_space", 20) for it in items])) * s
    gap, max_gap = opts.gap * space, opts.max_gap * space

    # per-item scale: shared, except anything that would overflow its column shrinks alone
    scales = [min(s, cw / w, ch / h) for (w, h) in sizes]
    heights = [h * k for (_, h), k in zip(sizes, scales)]
    columns = break_columns(heights, ch, gap)
    n_pages = -(-len(columns) // C)

    outdir.mkdir(parents=True, exist_ok=True)
    pages, placements, entries = [], [], []
    for pg in range(n_pages):
        canvas = Image.new("L", (W, H), 255)
        for c, (i0, i1) in enumerate(columns[pg * C:(pg + 1) * C]):
            used = sum(heights[i0:i1])
            n = i1 - i0
            slack = ch - used - gap * (n - 1)
            if opts.valign == "justify" and n > 1:
                step = gap + min(slack / (n - 1), max_gap - gap) if slack > 0 else gap
                y = opts.margin
            elif opts.valign == "center":
                step, y = gap, opts.margin + max(0.0, slack) / 2
            else:
                step, y = gap, opts.margin
            x_col = opts.margin + c * (cw + col_gap)
            for i in range(i0, i1):
                it, k = items[i], scales[i]
                w, h = sizes[i]
                tw, th = max(1, round(w * k)), max(1, round(h * k))
                arr = np.asarray(Image.open(cut_dir / it["file"]).convert("L"))
                if opts.whiten:
                    arr = _whiten(arr)
                im = Image.fromarray(arr).resize((tw, th), Image.LANCZOS)
                centred = opts.halign == "center" or it["kind"] == "extra"  # titles sit centred
                x = x_col + ((cw - tw) / 2 if centred else 0)
                canvas.paste(im, (round(x), round(y)))
                placements.append({"page": pg + 1, "column": c + 1, "file": it["file"],
                                   "x": round(x), "y": round(y), "w": tw, "h": th, "scale": round(k, 4)})
                y += th + step
        name = f"page-{pg + 1:03d}.png"
        canvas.save(outdir / name)
        pages.append(canvas)
        entries.append((name, f"Page {pg + 1}"))
        log(f"splice page {pg + 1}: " + ", ".join(p["file"] for p in placements if p["page"] == pg + 1))

    if pdf_path:
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=opts.dpi)
        log(f"wrote {pdf_path} ({n_pages} pages)")

    result = {"tool": "scoretools.splice", "size": [W, H], "dpi": opts.dpi, "columns": C,
              "scale": round(s, 4), "pages": n_pages, "placements": placements}
    (outdir / "splice.json").write_text(json.dumps(result, indent=2))
    proof.write_index(outdir, "Splice output", entries,
                      f"{len(items)} items on {n_pages} pages of {W}×{H}. Click a page for full size.")
    return result
