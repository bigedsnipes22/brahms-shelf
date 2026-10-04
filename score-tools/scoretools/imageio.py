"""Getting pages in and out: image folders, PDFs (via poppler), and grayscale saves."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from . import analyze  # noqa: F401  (lifts Pillow's pixel limit)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}


def natural_key(p: Path):
    """Sort page2 before page10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def expand_inputs(paths) -> list[Path]:
    """Files stay as given; folders become their images in natural order."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted((f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS), key=natural_key))
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def load_gray(path) -> np.ndarray:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return np.asarray(img.convert("L"))


def save_gray(arr: np.ndarray, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, optimize=False)


def _need(tool: str) -> str:
    exe = shutil.which(tool)
    if not exe:
        raise RuntimeError(f"'{tool}' not found. On a Mac: brew install poppler")
    return exe


def pdf_page_count(pdf: Path) -> int:
    out = subprocess.run([_need("pdfinfo"), str(pdf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else 0


def pdf_is_scan(pdf: Path) -> bool:
    """True when every page is exactly one embedded image, so extraction is lossless."""
    out = subprocess.run([_need("pdfimages"), "-list", str(pdf)], capture_output=True, text=True, check=True).stdout
    per_page: dict[int, int] = {}
    for line in out.splitlines()[2:]:
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        if parts[2] != "image":
            return False  # masks and stencils need compositing, so render instead
        per_page[int(parts[0])] = per_page.get(int(parts[0]), 0) + 1
    pages = pdf_page_count(pdf)
    return pages > 0 and len(per_page) == pages and all(n == 1 for n in per_page.values())


def pdf_to_pages(pdf, outdir, dpi: int = 600, mode: str = "auto") -> list[Path]:
    """Turn a PDF into page images. mode: auto | extract | render."""
    pdf = Path(pdf)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if mode == "auto":
        mode = "extract" if pdf_is_scan(pdf) else "render"
    prefix = outdir / "page"
    if mode == "extract":
        subprocess.run([_need("pdfimages"), "-png", "-p", str(pdf), str(prefix)], check=True)
    elif mode == "render":
        subprocess.run([_need("pdftoppm"), "-gray", "-png", "-r", str(dpi), str(pdf), str(prefix)], check=True)
    else:
        raise ValueError(f"unknown PDF mode {mode!r}")
    return sorted(outdir.glob("page*.png"), key=natural_key)


def gather_pages(inputs, workdir, dpi: int = 600, pdf_mode: str = "auto") -> list[Path]:
    """Accept any mix of images, folders and PDFs; PDFs are unpacked into workdir."""
    pages: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.suffix.lower() == ".pdf":
            pages.extend(pdf_to_pages(p, Path(workdir) / p.stem, dpi=dpi, mode=pdf_mode))
        else:
            pages.extend(expand_inputs([p]))
    return pages
