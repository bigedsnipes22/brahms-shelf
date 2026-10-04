"""Cut pages into systems by the white space between them.

Nothing here reads staves for meaning, so it works on piano, vocal and orchestral
scores alike. The rules:

* A row is blank only if nothing crosses it. A system line or barline joining the
  staves of a system keeps the rows between those staves inked, so a system
  never splits apart at its own inner gaps.
* A block of rows that holds staff lines is a system. Smaller blocks (tempo marks,
  lyrics, dynamics, page numbers) join the system on the far side of the widest
  blank band, or become an "extra" when they sit far from any system
  (titles, running heads, page numbers).
* When notes from neighbouring systems overlap and leave no blank band at all,
  the two systems arrive as one block. Inside a block, each pair of neighbouring
  staves is checked for a system line or barline running from one to the next.
  Where none does, that is a system break, cut with a seam: a crooked path that
  crosses as little ink as possible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import analyze, proof
from .imageio import load_gray, save_gray


@dataclass
class CutOptions:
    straighten: bool = True
    attach: float = 2.0       # staff spaces: how close a small block must be to join a system
    split: bool = True         # seam-cut systems whose notes overlap
    min_lines: int = 4        # staff-like rows a block needs to count as a system


@dataclass
class Block:
    y0: int
    y1: int
    lines: int = 0
    ink: int = 0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass
class PageScan:
    angle: float
    metrics: analyze.StaffMetrics
    x0: int
    x1: int
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Region:
    kind: str           # "system" or "extra"
    top: np.ndarray     # per-column first row (inclusive), full page width
    bottom: np.ndarray  # per-column last row (exclusive)
    seam: bool = False  # True if either edge is a crooked cut


# ---------------------------------------------------------------- pass 1


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def clear_borders(ink: np.ndarray) -> np.ndarray:
    """Blank out dark scanner borders at the page edges so they don't join every row."""
    h, w = ink.shape
    ink = ink.copy()
    colfrac = ink.mean(axis=0)
    for xs in (range(0, w // 10), range(w - 1, w - 1 - w // 10, -1)):
        for x in xs:
            if colfrac[x] < 0.6:
                break
            ink[:, x] = False
    rowfrac = ink.mean(axis=1)
    for ys in (range(0, h // 20), range(h - 1, h - 1 - h // 20, -1)):
        for y in ys:
            if rowfrac[y] < 0.97:
                break
            ink[y, :] = False
    return ink


def _content_span(ink: np.ndarray, line: int) -> tuple[int, int]:
    cols = np.flatnonzero(ink.sum(axis=0) > line)
    if cols.size == 0:
        return 0, ink.shape[1]
    return int(cols[0]), int(cols[-1]) + 1


def count_staff_lines(ink_rows: np.ndarray, strips: int = 8, fill: float = 0.6) -> int:
    """Most long horizontal lines seen in any vertical strip of the block."""
    h, w = ink_rows.shape
    sw = max(1, w // strips)
    best = 0
    for s in range(strips):
        seg = ink_rows[:, s * sw:(s + 1) * sw]
        if seg.shape[1] == 0:
            continue
        best = max(best, len(_runs(seg.mean(axis=1) > fill)))
    return best


def find_blocks(ink: np.ndarray, x0: int, x1: int, m: analyze.StaffMetrics) -> list[Block]:
    area = ink[:, x0:x1]
    prof = area.sum(axis=1)
    raw = [Block(a, b) for a, b in _runs(prof > m.line // 2)]
    min_gap = max(3, m.space // 2)
    blocks: list[Block] = []
    for b in raw:
        if blocks and b.y0 - blocks[-1].y1 < min_gap:
            blocks[-1].y1 = b.y1
        else:
            blocks.append(b)
    for b in blocks:
        b.lines = count_staff_lines(area[b.y0:b.y1])
        b.ink = int(prof[b.y0:b.y1].sum())
    return blocks


def scan_page(gray: np.ndarray, opts: CutOptions):
    """Straighten (optionally) and measure a page. Returns (page, ink, threshold, scan)."""
    angle = 0.0
    if opts.straighten:
        gray, angle = analyze.straighten(gray)
    threshold = analyze.otsu_threshold(gray)
    ink = clear_borders(analyze.ink_mask(gray, threshold))
    m = analyze.staff_metrics(ink)
    x0, x1 = _content_span(ink, m.line)
    scan = PageScan(angle=angle, metrics=m, x0=x0, x1=x1, blocks=find_blocks(ink, x0, x1, m))
    return gray, ink, threshold, scan


# ---------------------------------------------------------------- seams


def find_seam(ink_band: np.ndarray) -> np.ndarray:
    """Left-to-right path through the band crossing the least ink. Returns row per column."""
    bh, w = ink_band.shape
    centre = (np.abs(np.arange(bh) - (bh - 1) / 2) / max(1, bh)).astype(np.float32)
    cost = ink_band.astype(np.float32) * 100 + centre[:, None]
    acc = cost[:, 0].copy()
    back = np.zeros((bh, w), dtype=np.int8)
    inf = np.float32(np.inf)
    for c in range(1, w):
        up = np.concatenate(([inf], acc[:-1])) + 0.05
        down = np.concatenate((acc[1:], [inf])) + 0.05
        stacked = np.stack((up, acc, down))
        idx = np.argmin(stacked, axis=0)
        acc = stacked[idx, np.arange(bh)] + cost[:, c]
        back[:, c] = idx - 1
    path = np.empty(w, dtype=np.int64)
    path[-1] = int(np.argmin(acc))
    for c in range(w - 1, 0, -1):
        path[c - 1] = path[c] + back[path[c], c]
    return path


# ---------------------------------------------------------------- pass 2


@dataclass
class _Seg:
    y0: int
    y1: int
    major: bool
    ink: int = 0
    top_seam: np.ndarray | None = None     # set when the segment starts at a seam
    bottom_seam: np.ndarray | None = None  # set when the segment ends at a seam


def _full(arr_cols: np.ndarray, x0: int, width: int) -> np.ndarray:
    return np.pad(arr_cols, (x0, width - x0 - arr_cols.size), mode="edge")


def find_staves(rows: np.ndarray, space: int, fill: float = 0.4) -> list[tuple[int, int]]:
    """(top, bottom) of each staff in a block: groups of 3+ long lines about a staff space apart."""
    lines = _runs(rows.mean(axis=1) > fill)
    groups: list[list[tuple[int, int]]] = []
    for a, b in lines:
        if groups and a - groups[-1][-1][1] <= 1.5 * space:
            groups[-1].append((a, b))
        else:
            groups.append([(a, b)])
    return [(g[0][0], g[-1][1]) for g in groups if len(g) >= 3]


def system_breaks(rows: np.ndarray, space: int) -> list[tuple[int, int]]:
    """Gaps between staves that nothing joins. Within a system, a system line or barline
    runs unbroken from each staff to the next; between systems, nothing does."""
    staves = find_staves(rows, space)
    out = []
    for (_, a_bot), (b_top, _) in zip(staves, staves[1:]):
        between = rows[a_bot:b_top]
        if between.shape[0] > 0 and not between.all(axis=0).any():
            out.append((a_bot, b_top))
    return out


def _segments(ink: np.ndarray, scan: PageScan, use_lines: bool, opts: CutOptions) -> list[_Seg]:
    width = ink.shape[1]
    tallest = max((b.height for b in scan.blocks), default=0)
    segs: list[_Seg] = []
    for b in scan.blocks:
        major = b.lines >= opts.min_lines if use_lines else b.height >= 0.5 * tallest
        breaks = system_breaks(ink[b.y0:b.y1, scan.x0:scan.x1], scan.metrics.space) if (
            major and use_lines and opts.split) else []
        if not breaks:
            segs.append(_Seg(b.y0, b.y1, major, b.ink))
            continue
        # systems whose notes overlap: no blank row between them, so cut along a seam
        seams = []
        for lo, hi in breaks:
            band = analyze.dilate(ink[b.y0 + lo:b.y0 + hi, scan.x0:scan.x1], 2)  # keep clear of anti-aliased edges
            path = find_seam(band) + b.y0 + lo
            seams.append(_full(path, scan.x0, width))
        edges = [None, *seams, None]
        for j in range(len(edges) - 1):
            top, bot = edges[j], edges[j + 1]
            y0 = b.y0 if top is None else int(top[scan.x0:scan.x1].min())
            y1 = b.y1 if bot is None else int(bot[scan.x0:scan.x1].max())
            segs.append(_Seg(y0, y1, True, b.ink, top_seam=top, bottom_seam=bot))
    return segs


def assemble(ink: np.ndarray, scan: PageScan, use_lines: bool, opts: CutOptions) -> list[Region]:
    h, width = ink.shape
    segs = _segments(ink, scan, use_lines, opts)
    if not segs:
        return []
    majors = [i for i, s in enumerate(segs) if s.major]
    flat = lambda y: np.full(width, int(y), dtype=np.int64)  # noqa: E731

    if not majors:
        return [Region("extra", flat(segs[0].y0), flat(segs[-1].y1))]

    attach = opts.attach * scan.metrics.space
    speck = scan.metrics.space * scan.metrics.space
    real = [i for i, s in enumerate(segs) if s.major or s.ink >= speck]  # specks never decide a cut

    # owner[i] = which system segment i belongs to; -1 = extra above, -2 = extra below
    owner = [i if s.major else 0 for i, s in enumerate(segs)]
    for a, b in zip(majors, majors[1:]):
        if b == a + 1:
            continue
        # the widest blank band between the two systems decides where marks between them go
        between = [i for i in real if a <= i <= b]
        p, q = max(zip(between, between[1:]),
                   key=lambda pq: (segs[pq[1]].y0 - segs[pq[0]].y1, -abs(pq[0] - (a + b) / 2)))
        mid = (segs[p].y1 + segs[q].y0) / 2
        for i in range(a + 1, b):
            owner[i] = a if (i <= p or (i < q and (segs[i].y0 + segs[i].y1) / 2 < mid)) else b

    # marks above the first system and below the last join it while each is close to the
    # previous real mark; past the first far one, the rest are extras (titles, page numbers)
    first, last = majors[0], majors[-1]
    for order, sys_i, edge_of, gap_to, far in (
        (range(first - 1, -1, -1), first, lambda s: s.y0, lambda e, s: e - s.y1, -1),
        (range(last + 1, len(segs)), last, lambda s: s.y1, lambda e, s: s.y0 - e, -2),
    ):
        edge, attaching = edge_of(segs[sys_i]), True
        for i in order:
            close = gap_to(edge, segs[i]) <= attach
            if attaching and close:
                owner[i] = sys_i
                if segs[i].ink >= speck:
                    edge = edge_of(segs[i])
            else:
                owner[i] = far
                if segs[i].ink >= speck:
                    attaching = False

    regions: list[Region] = []
    for key in [-1, *majors, -2]:
        members = [segs[i] for i in range(len(segs)) if owner[i] == key]
        if not members:
            continue
        if key < 0:
            # far from every system: keep real marks (titles, page numbers), drop lone specks
            members = [m for m in members if m.ink >= speck]
            if not members:
                continue
        lead, tail = members[0], members[-1]
        top = lead.top_seam if lead.top_seam is not None else flat(lead.y0)
        bottom = tail.bottom_seam if tail.bottom_seam is not None else flat(tail.y1)
        seam = lead.top_seam is not None or tail.bottom_seam is not None
        regions.append(Region("system" if key >= 0 else "extra", top, bottom, seam))
    return regions


def _span(counts: np.ndarray, gap: int, speck: int):
    """First..last index of real ink along one axis. Specks sitting apart at the edges
    (further than gap from anything, with under `speck` pixels of ink) don't count."""
    runs = _runs(counts > 0)
    if not runs:
        return None
    groups = [list(runs[0])]
    for a, b in runs[1:]:
        if a - groups[-1][1] < gap:
            groups[-1][1] = b
        else:
            groups.append([a, b])
    real = [g for g in groups if counts[g[0]:g[1]].sum() >= speck] or groups
    return real[0][0], real[-1][1]


def extract(gray: np.ndarray, region: Region, x0: int, x1: int, pad: int, fill: int,
            space: int = 20, threshold: int = 128):
    """Crop a region (whitening anything past a seam) and trim it to its ink plus pad."""
    gap, speck = max(2, space // 4), space * space // 4
    y0 = int(region.top[x0:x1].min())
    y1 = int(region.bottom[x0:x1].max())
    crop = gray[y0:y1, x0:x1].copy()
    rows = np.arange(y0, y1)[:, None]
    outside = (rows < region.top[None, x0:x1]) | (rows >= region.bottom[None, x0:x1])
    crop[outside] = fill
    ink = analyze.ink_mask(crop, threshold)
    ys = _span(ink.sum(axis=1), gap, speck)
    xs = _span(ink.sum(axis=0), gap, speck)
    if ys is None or xs is None:
        return None, None
    (ty0, ty1), (tx0, tx1) = ys, xs
    trimmed = crop[ty0:ty1, tx0:tx1]
    out = np.full((trimmed.shape[0] + 2 * pad, trimmed.shape[1] + 2 * pad), fill, dtype=np.uint8)
    out[pad:pad + trimmed.shape[0], pad:pad + trimmed.shape[1]] = trimmed
    bbox = [int(x0 + tx0), int(y0 + ty0), int(x0 + tx1), int(y0 + ty1)]
    return out, bbox


# ---------------------------------------------------------------- driver


def cut(pages: list[Path], outdir, opts: CutOptions | None = None, log=print) -> dict:
    """Cut every page into system images under outdir. Returns the manifest written to cut.json."""
    opts = opts or CutOptions()
    manifest = _cut_pages(pages, Path(outdir), opts, use_lines=True, log=log)
    if manifest is None:
        # no staff lines anywhere (unusual notation, or a very faint scan): fall back to block size
        log("no staff lines found anywhere; treating the tallest blocks on each page as systems")
        manifest = _cut_pages(pages, Path(outdir), opts, use_lines=False, log=log)
    return manifest


def _cut_pages(pages: list[Path], outdir: Path, opts: CutOptions, use_lines: bool, log):
    (outdir / "proof").mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("[0-9][0-9][0-9][0-9].png"):
        old.unlink()
    items = []
    proofs = []
    any_lines = False
    for pi, p in enumerate(pages):
        gray, ink, threshold, scan = scan_page(load_gray(p), opts)
        any_lines |= any(b.lines >= opts.min_lines for b in scan.blocks)
        fill = analyze.paper_tone(gray, ink)
        pad = max(2, scan.metrics.space // 2)
        regions = assemble(ink, scan, use_lines, opts)
        drawn = []
        for r in regions:
            img, bbox = extract(gray, r, scan.x0, scan.x1, pad, fill, scan.metrics.space, threshold)
            if img is None:
                continue
            n = len(items) + 1
            name = f"{n:04d}.png"
            save_gray(img, outdir / name)
            items.append({
                "file": name, "kind": r.kind, "page": pi + 1, "source": p.name,
                "bbox": bbox, "angle": scan.angle, "seam": r.seam,
                "staff_space": scan.metrics.space,
            })
            drawn.append((r, n, bbox))
        proof_name = f"page-{pi + 1:03d}.png"
        proof.draw_cut_proof(gray, drawn, outdir / "proof" / proof_name)
        proofs.append((proof_name, f"Page {pi + 1} · {p.name} · "
                       f"{sum(r.kind == 'system' for r, _, _ in drawn)} systems"
                       + (f" · straightened {scan.angle:+.2f}°" if scan.angle else "")))
        log(f"cut   {p.name}: " + (f"straightened {scan.angle:+.2f}°; " if scan.angle else "") + ", ".join(f"#{n} {r.kind}{' (seam)' if r.seam else ''}" for r, n, _ in drawn))

    if use_lines and not any_lines:
        return None
    manifest = {"tool": "scoretools.cut", "items": items}
    (outdir / "cut.json").write_text(json.dumps(manifest, indent=2))
    proof.write_index(outdir / "proof", "Cut proof", proofs,
                      "Each coloured band is one system, numbered in reading order. Grey bands are extras "
                      "(titles, page numbers). A red line marks a crooked seam cut — check those first.")
    return manifest
