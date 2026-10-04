"""Proof sheets: every tool shows its work over the original page, so nothing ships unchecked."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROOF_WIDTH = 1400
SYSTEM_COLOURS = [(38, 110, 196), (214, 120, 24), (40, 150, 90), (150, 70, 170)]
EXTRA_COLOUR = (120, 120, 120)
SEAM_COLOUR = (220, 30, 30)


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _base(gray: np.ndarray):
    h, w = gray.shape
    s = min(1.0, PROOF_WIDTH / w)
    img = Image.fromarray(gray).resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS).convert("RGB")
    return img, s


def draw_cut_proof(gray: np.ndarray, drawn, path) -> None:
    """drawn: list of (Region, number, bbox) in page coordinates."""
    img, s = _base(gray)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = _font(max(14, int(34 * s * 2)))
    for k, (region, n, bbox) in enumerate(drawn):
        colour = EXTRA_COLOUR if region.kind == "extra" else SYSTEM_COLOURS[k % len(SYSTEM_COLOURS)]
        x0, y0, x1, y1 = (int(v * s) for v in bbox)
        d.rectangle([x0, y0, x1, y1], fill=colour + (46,), outline=colour + (255,), width=3)
        label = str(n) if region.kind == "system" else f"{n} extra"
        tw = d.textlength(label, font=font)
        d.rectangle([x0, y0, x0 + tw + 12, y0 + font.size + 8], fill=colour + (235,))
        d.text((x0 + 6, y0 + 3), label, fill=(255, 255, 255, 255), font=font)
        if region.seam:
            for edge in (region.top, region.bottom):
                if np.ptp(edge) == 0:
                    continue
                xs = np.arange(0, edge.size, max(1, int(1 / s)))
                d.line([(float(x * s), float(edge[x] * s)) for x in xs], fill=SEAM_COLOUR + (255,), width=3)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_erase_proof(gray: np.ndarray, hits, path) -> None:
    """hits: list of (x0, y0, x1, y1, score) in page coordinates, drawn on the page before erasing."""
    img, s = _base(gray)
    d = ImageDraw.Draw(img)
    font = _font(16)
    for x0, y0, x1, y1, score in hits:
        box = [int(x0 * s) - 3, int(y0 * s) - 3, int(x1 * s) + 3, int(y1 * s) + 3]
        d.rectangle(box, outline=SEAM_COLOUR, width=3)
        d.text((box[0], box[3] + 2), f"{score:.2f}", fill=SEAM_COLOUR, font=font)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def write_index(folder, title: str, entries, note: str = "") -> Path:
    """A contact sheet of proof images. entries: list of (filename, caption)."""
    folder = Path(folder)
    cards = "\n".join(
        f'<figure><a href="{html.escape(f)}"><img src="{html.escape(f)}" loading="lazy" alt=""></a>'
        f"<figcaption>{html.escape(c)}</figcaption></figure>"
        for f, c in entries
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#eef0ec;--card:#fff;--ink:#1f221e;--muted:#5c635c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15191a;--card:#1f2424;--ink:#e2e7e1;--muted:#98a099}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,Helvetica,Arial,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} p{{color:var(--muted);margin:0 0 16px;max-width:70ch}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,340px),1fr));gap:16px}}
figure{{margin:0;background:var(--card);border-radius:6px;padding:8px}}
img{{width:100%;height:auto;display:block;background:#fff}}
figcaption{{font-size:13px;color:var(--muted);padding-top:6px}}
</style></head><body>
<h1>{html.escape(title)}</h1><p>{html.escape(note)}</p>
<main>
{cards}
</main></body></html>
"""
    out = folder / "index.html"
    out.write_text(page)
    return out
