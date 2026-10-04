"""Synthetic score pages with known answers, for testing cut / splice / erase."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SPACE = 24  # staff space, px
LINE = 3
STAFF_H = 4 * SPACE + LINE


def font(size: int, bold: bool = False):
    name = "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


KINDS = {
    # staves, gap between staves, barlines run through all staves?
    "piano": (2, 4 * SPACE, True),
    "violin": (1, 0, True),
    "orchestra": (4, 3 * SPACE, False),  # only the left system line joins them
    "song": (3, 4 * SPACE, False),
}


@dataclass
class Truth:
    systems: list[tuple[int, int]] = field(default_factory=list)  # y ranges, before skew
    plate_box: tuple[int, int, int, int] | None = None              # x0, y0, x1, y1
    interleaved_after: int | None = None                            # index of upper system of the overlapping pair


def _head(d, x, y):
    d.ellipse([x, y - SPACE // 2 + 1, x + int(1.3 * SPACE), y + SPACE // 2 - 1], fill=0)


def _note(d, x, y, up):
    _head(d, x, y)
    if up:
        sx = x + int(1.3 * SPACE) - LINE
        d.rectangle([sx, y - int(3.5 * SPACE), sx + LINE - 1, y], fill=0)
    else:
        d.rectangle([x, y, x + LINE - 1, y + int(3.5 * SPACE)], fill=0)


def _ledger_note(d, x, staff_top, steps):
    """A note `steps` staff spaces above (negative) or below (positive) the staff, with ledger lines."""
    if steps < 0:
        y = staff_top + steps * SPACE
        for k in range(1, -steps + 1):
            ly = staff_top - k * SPACE
            d.rectangle([x - 8, ly, x + int(1.3 * SPACE) + 8, ly + LINE - 1], fill=0)
    else:
        y = staff_top + 4 * SPACE + steps * SPACE
        for k in range(1, steps + 1):
            ly = staff_top + 4 * SPACE + k * SPACE
            d.rectangle([x - 8, ly, x + int(1.3 * SPACE) + 8, ly + LINE - 1], fill=0)
    # engraved convention: a note off the staff has its stem pointing back toward it
    if steps > 0:
        stem_top = staff_top + 2 * SPACE
        _head(d, x, y)
        sx = x + int(1.3 * SPACE) - LINE
        d.rectangle([sx, stem_top, sx + LINE - 1, y], fill=0)
        return stem_top, y + SPACE // 2
    stem_bot = staff_top + 2 * SPACE
    _head(d, x, y)
    d.rectangle([x, y, x + LINE - 1, stem_bot], fill=0)
    return y - SPACE // 2, stem_bot


def draw_system(d, rng, x0, x1, y, kind, measures=4):
    staves, gap, through = KINDS[kind]
    tops = [y + i * (STAFF_H + gap) for i in range(staves)]
    bottom = tops[-1] + STAFF_H
    for t in tops:
        for i in range(5):
            d.rectangle([x0, t + i * SPACE, x1, t + i * SPACE + LINE - 1], fill=0)
        for x in range(x0 + 3 * SPACE, x1 - 2 * SPACE, int(2.2 * SPACE)):
            k = rng.randint(1, 7)
            _note(d, x, t + k * SPACE // 2, up=k >= 4)
    d.rectangle([x0, tops[0], x0 + LINE + 1, bottom], fill=0)  # system line
    for m in range(1, measures + 1):
        bx = x0 + (x1 - x0) * m // measures - LINE
        if through:
            d.rectangle([bx, tops[0], bx + LINE, bottom], fill=0)
        else:
            for t in tops:
                d.rectangle([bx, t, bx + LINE, t + STAFF_H], fill=0)
    return tops[0], bottom, tops


def make_page(kind="piano", systems=5, title=False, tempo=False, page_number=None, plate=True,
              dynamics=True, interleave=False, skew=0.0, specks=0, paper=255, seed=0,
              size=(2000, 2600)):
    rng = random.Random(seed)
    W, H = size
    img = Image.new("L", size, 255)
    d = ImageDraw.Draw(img)
    truth = Truth()
    x0, x1 = 160, W - 160
    top = 200
    if page_number is not None:
        d.text((W - 220, 90), str(page_number), fill=0, font=font(40))
    if title:
        d.text((W // 2 - 420, 140), "Sonate in f-moll", fill=0, font=font(96, bold=True))
        d.text((W - 640, 290), "Johannes Brahms", fill=0, font=font(44))
        top = 470
    if tempo:
        top += 60
    bottom_limit = H - 260
    staves, gap, _ = KINDS[kind]
    sys_h = staves * STAFF_H + (staves - 1) * gap
    room = bottom_limit - top
    spacing = (room - systems * sys_h) / max(1, systems - 1) if systems > 1 else 0
    if spacing < 3 * SPACE:
        raise ValueError("too many systems for the page")
    y = top
    for i in range(systems):
        yi = int(y)
        if interleave and i == 2:
            yi -= int(spacing) - 6 * SPACE  # pull up: ledger notes overlap row-wise, but never touch
        s0, s1, tops = draw_system(d, rng, x0, x1, yi, kind)
        lo, hi = s0, s1
        if tempo and i == 0:
            d.text((x0 + 20, s0 - 2 * SPACE - 40), "Allegro maestoso", fill=0, font=font(40, bold=True))
            lo = s0 - 2 * SPACE - 40
        if interleave and i == 1:
            a, b = _ledger_note(d, x0 + 300, tops[-1], 4)
            hi = max(hi, b)
        if interleave and i == 2:
            a, b = _ledger_note(d, x0 + 900, tops[0], -4)
            lo = min(lo, a)
            truth.interleaved_after = 1
        if dynamics and i % 2 == 0:
            d.text((x0 + 400, s1 + SPACE // 2), "p", fill=0, font=font(46, bold=True))
            hi = max(hi, s1 + SPACE // 2 + 56)
        truth.systems.append((lo, hi))
        y += sys_h + spacing
    if plate:
        f = font(30)
        text = "E. F. 1234"
        px, py = x0, H - 150
        d.text((px, py), text, fill=0, font=f)
        l, t, r, b = d.textbbox((px, py), text, font=f)
        truth.plate_box = (l, t, r, b)
    for _ in range(specks):
        sx, sy = rng.randrange(W), rng.randrange(H)
        r = rng.randint(1, 2)
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=0)
    arr = np.asarray(img).astype(np.float32)
    arr = paper - (255 - arr) * (paper - 20) / 235  # ink stays dark, paper takes the given tone
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if skew:
        img = img.rotate(skew, resample=Image.BICUBIC, fillcolor=paper)
    return np.asarray(img), truth
