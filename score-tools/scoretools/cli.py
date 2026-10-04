"""Command line: scoretools cut | splice | erase | run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .cut import CutOptions, cut
from .erase import Template, erase, parse_box, template_from_page
from .imageio import gather_pages, load_gray
from .splice import PRESETS, SpliceOptions, parse_size, splice


def _band(text: str | None):
    if not text:
        return None
    a, b = (float(v) for v in text.split(":"))
    return a, b


def _templates(args, pages: list[Path]) -> list[Template]:
    out = [Template(Path(t).stem, load_gray(t)) for t in (args.template or [])]
    for spec in args.from_ or []:
        ref, box = spec.rsplit(":", 1)
        if ref.isdigit():
            page = pages[int(ref) - 1]
        else:
            matches = [p for p in pages if p.name == ref or p.stem == ref]
            if not matches:
                raise SystemExit(f"--from: no page named {ref!r}")
            page = matches[0]
        out.append(template_from_page(load_gray(page), parse_box(box), name=f"{page.stem}@{box}"))
    return out


def _add_input_args(p):
    p.add_argument("inputs", nargs="+", help="page images, folders of them, or PDFs")
    p.add_argument("-o", "--out", required=True, help="output folder")
    p.add_argument("--dpi", type=int, default=600, help="render resolution for vector PDFs (default 600)")
    p.add_argument("--pdf-mode", choices=["auto", "extract", "render"], default="auto",
                   help="scanned PDFs are extracted losslessly, others rendered (default auto)")


def _add_cut_args(p):
    p.add_argument("--no-straighten", action="store_true", help="don't level crooked pages first")
    p.add_argument("--no-split", action="store_true", help="never seam-cut systems whose notes overlap")
    p.add_argument("--attach", type=float, default=2.0,
                   help="staff spaces within which marks above/below join a system (default 2)")


def _add_erase_args(p, prefix=""):
    p.add_argument(f"--{prefix}template", action="append", dest="template", metavar="PNG",
                   help="image of the thing to erase, cropped from a page at the same resolution (repeatable)")
    p.add_argument(f"--{prefix}from", action="append", dest="from_", metavar="PAGE:X,Y,W,H",
                   help="take the template from a box on a page; PAGE is a 1-based number or file name (repeatable)")
    p.add_argument("--threshold", type=float, default=0.7, help="match score needed, 0-1 (default 0.7)")
    p.add_argument("--band", help="only search this vertical slice of each page, as fractions, e.g. 0:0.15")
    p.add_argument("--mode", choices=["ink", "box"], default="ink",
                   help="erase just the template's ink shape, or the whole matched box (default ink)")
    p.add_argument("--grow", type=int, default=3, help="px to grow the erased area (default 3)")


def _add_splice_args(p):
    p.add_argument("--preset", choices=sorted(PRESETS), default="ipad", help="page shape (default ipad)")
    p.add_argument("--size", help="custom page size in px, e.g. 2360x1640 (overrides the preset)")
    p.add_argument("--dpi-out", type=int, dest="dpi_out", help="PDF resolution for the page size")
    p.add_argument("--margin", type=int, help="page margin in px")
    p.add_argument("--columns", type=int, help="systems side by side, e.g. 2")
    p.add_argument("--valign", choices=["top", "justify", "center"])
    p.add_argument("--halign", choices=["left", "center"])
    p.add_argument("--gap", type=float, help="min space between systems, in staff spaces (default 1)")
    p.add_argument("--max-gap", type=float, help="max space when justifying, in staff spaces (default 5)")
    p.add_argument("--scale", type=float, help="output px per source px (default: fit widest system)")
    p.add_argument("--extras", choices=["first", "all", "none"],
                   help="titles/page numbers to keep: first = only the opening title (default)")
    p.add_argument("--no-whiten", action="store_true", help="keep the scan's paper tone")


def _splice_opts(args) -> SpliceOptions:
    return SpliceOptions.preset(
        args.preset,
        size=parse_size(args.size) if args.size else None,
        dpi=args.dpi_out, margin=args.margin, columns=args.columns,
        valign=args.valign, halign=args.halign, gap=args.gap, max_gap=args.max_gap,
        scale=args.scale, extras=args.extras, whiten=False if args.no_whiten else None,
    )


def _cut_opts(args) -> CutOptions:
    return CutOptions(straighten=not args.no_straighten, attach=args.attach,
                      split=not args.no_split)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scoretools", description="Cut, splice and clean up score scans.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cut", help="cut pages into system images")
    _add_input_args(p)
    _add_cut_args(p)

    p = sub.add_parser("splice", help="pack cut systems onto new pages")
    p.add_argument("cut_dir", help="folder written by 'scoretools cut'")
    p.add_argument("-o", "--out", required=True, help="output folder for page images")
    p.add_argument("--pdf", help="also write a PDF here")
    _add_splice_args(p)

    p = sub.add_parser("erase", help="erase every match of a template (plate numbers, headers)")
    _add_input_args(p)
    _add_erase_args(p)

    p = sub.add_parser("run", help="pages -> (erase) -> cut -> splice -> PDF, with proofs at each step")
    _add_input_args(p)
    _add_cut_args(p)
    _add_erase_args(p, prefix="erase-")
    _add_splice_args(p)

    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.cmd == "splice":
        splice(args.cut_dir, out, _splice_opts(args), pdf_path=args.pdf)
        print(f"open {out / 'index.html'}")
        return 0

    pages = gather_pages(args.inputs, out / "pages", dpi=args.dpi, pdf_mode=args.pdf_mode)
    if not pages:
        raise SystemExit("no pages found")

    if args.cmd == "erase":
        templates = _templates(args, pages)
        if not templates:
            raise SystemExit("erase needs --template or --from")
        erase(pages, out, templates, args.threshold, _band(args.band), args.mode, args.grow)
        print(f"check {out / 'proof' / 'index.html'}")
        return 0

    if args.cmd == "cut":
        cut(pages, out, _cut_opts(args))
        print(f"check {out / 'proof' / 'index.html'}")
        return 0

    # run
    templates = _templates(args, pages)
    if templates:
        report = erase(pages, out / "erased", templates, args.threshold, _band(args.band), args.mode, args.grow)
        pages = [out / "erased" / r["output"] for r in report["pages"]]
    cut(pages, out / "systems", _cut_opts(args))
    name = Path(args.inputs[0]).stem
    pdf = out / f"{name}-{args.preset}.pdf"
    splice(out / "systems", out / args.preset, _splice_opts(args), pdf_path=pdf)
    print("\nproof, in order:")
    if templates:
        print(f"  {out / 'erased' / 'proof' / 'index.html'}")
    print(f"  {out / 'systems' / 'proof' / 'index.html'}")
    print(f"  {out / args.preset / 'index.html'}")
    print(f"result: {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
