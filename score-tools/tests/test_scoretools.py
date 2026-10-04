import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from synth import LINE, SPACE, make_page  # noqa: E402

from scoretools import analyze  # noqa: E402
from scoretools.cli import main  # noqa: E402
from scoretools.cut import cut  # noqa: E402
from scoretools.erase import erase, find_matches, template_from_page  # noqa: E402
from scoretools.splice import SpliceOptions, break_columns, splice  # noqa: E402


def write_pages(folder: Path, specs):
    folder.mkdir(parents=True, exist_ok=True)
    paths, truths = [], []
    for i, spec in enumerate(specs):
        arr, truth = make_page(**spec)
        p = folder / f"p{i + 1}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
        truths.append(truth)
    return paths, truths


def systems_per_page(manifest):
    out = {}
    for it in manifest["items"]:
        if it["kind"] == "system":
            out[it["page"]] = out.get(it["page"], 0) + 1
    return out


# ------------------------------------------------------------- analyze


def test_staff_metrics():
    arr, _ = make_page(kind="piano", systems=4)
    m = analyze.staff_metrics(analyze.ink_mask(arr))
    assert m.found
    assert m.line == LINE
    assert m.space == SPACE


@pytest.mark.parametrize("angle", [0.0, 0.5, -0.8, 1.5])
def test_skew_is_recovered(angle):
    arr, _ = make_page(kind="piano", systems=4, skew=angle)
    _, applied = analyze.straighten(arr)
    assert applied == pytest.approx(-angle, abs=0.06)


# ------------------------------------------------------------- cut


@pytest.mark.parametrize("kind,n", [("piano", 5), ("violin", 9), ("orchestra", 2), ("song", 3)])
def test_cut_counts_systems(tmp_path, kind, n):
    pages, _ = write_pages(tmp_path / "in", [dict(kind=kind, systems=n, page_number=3, seed=7)])
    m = cut(pages, tmp_path / "cut", log=lambda *_: None)
    assert systems_per_page(m) == {1: n}


def test_cut_title_tempo_and_plate(tmp_path):
    pages, _ = write_pages(tmp_path / "in", [dict(kind="piano", systems=4, title=True, tempo=True, seed=1)])
    m = cut(pages, tmp_path / "cut", log=lambda *_: None)
    kinds = [it["kind"] for it in m["items"]]
    # title above, plate number below are extras; the tempo mark rides with the first system
    assert kinds == ["extra", "system", "system", "system", "system", "extra"]
    first_system = m["items"][1]
    assert first_system["bbox"][1] < 470  # starts at the tempo mark, above the staff


def test_cut_seams_between_overlapping_systems(tmp_path):
    pages, truths = write_pages(tmp_path / "in", [dict(kind="piano", systems=5, interleave=True, seed=2)])
    m = cut(pages, tmp_path / "cut", log=lambda *_: None)
    systems = [it for it in m["items"] if it["kind"] == "system"]
    assert len(systems) == 5
    assert [it["seam"] for it in systems] == [False, True, True, False, False]
    # each overlapping system keeps its own ledger note: the upper one reaches below
    # the lower one's top, which is only possible with a crooked cut
    upper, lower = systems[1]["bbox"], systems[2]["bbox"]
    assert upper[3] > lower[1]


def test_cut_survives_skew_specks_and_grey_paper(tmp_path):
    specs = [dict(kind="piano", systems=5, interleave=True, seed=2, skew=0.6, specks=40, paper=228)]
    pages, _ = write_pages(tmp_path / "in", specs)
    m = cut(pages, tmp_path / "cut", log=lambda *_: None)
    assert systems_per_page(m) == {1: 5}
    xs = {it["bbox"][0] for it in m["items"] if it["kind"] == "system"}
    assert max(xs) - min(xs) <= 4  # specks beside a system don't throw off its left edge


def test_cut_loses_no_ink(tmp_path):
    pages, _ = write_pages(tmp_path / "in", [dict(kind="song", systems=3, title=True, seed=4)])
    m = cut(pages, tmp_path / "cut", log=lambda *_: None)
    page_ink = analyze.ink_mask(np.asarray(Image.open(pages[0])), 128).sum()
    cut_ink = sum(analyze.ink_mask(np.asarray(Image.open(tmp_path / "cut" / it["file"])), 128).sum()
                  for it in m["items"])
    assert cut_ink == page_ink


def test_cut_writes_proofs(tmp_path):
    pages, _ = write_pages(tmp_path / "in", [dict(kind="violin", systems=6, seed=1)] * 2)
    cut(pages, tmp_path / "cut", log=lambda *_: None)
    assert (tmp_path / "cut" / "proof" / "page-001.png").exists()
    assert (tmp_path / "cut" / "proof" / "page-002.png").exists()
    assert "page-002.png" in (tmp_path / "cut" / "proof" / "index.html").read_text()


# ------------------------------------------------------------- splice


def test_break_columns_balances_and_keeps_order():
    cols = break_columns([100] * 7, column_h=350, gap=10)
    assert cols[0][0] == 0 and cols[-1][1] == 7
    assert all(a[1] == b[0] for a, b in zip(cols, cols[1:]))
    assert all(sum([100] * (j - i)) + 10 * (j - i - 1) <= 350 for i, j in cols)


def test_break_columns_oversized_item_gets_own_column():
    assert break_columns([50, 500, 50], column_h=300, gap=5) == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize("preset", ["ipad", "ipad-2up", "letter", "video"])
def test_splice_places_every_system_once(tmp_path, preset):
    pages, _ = write_pages(tmp_path / "in", [dict(kind="piano", systems=4, title=True, seed=1),
                                             dict(kind="piano", systems=5, page_number=2, seed=2)])
    m = cut(pages, tmp_path / "cut", log=lambda *_: None)
    opts = SpliceOptions.preset(preset)
    r = splice(tmp_path / "cut", tmp_path / "out", opts, pdf_path=tmp_path / "out.pdf", log=lambda *_: None)
    placed = [p["file"] for p in r["placements"]]
    systems = [it["file"] for it in m["items"] if it["kind"] == "system"]
    assert [f for f in placed if f in systems] == systems
    assert m["items"][0]["file"] in placed  # the opening title is kept
    for page in sorted((tmp_path / "out").glob("page-*.png")):
        assert Image.open(page).size == opts.size
    for p in r["placements"]:
        assert p["y"] + p["h"] <= opts.size[1] - opts.margin + 1
    assert (tmp_path / "out.pdf").stat().st_size > 0


def test_splice_shared_scale_and_left_edge(tmp_path):
    pages, _ = write_pages(tmp_path / "in", [dict(kind="violin", systems=9, seed=5)])
    cut(pages, tmp_path / "cut", log=lambda *_: None)
    r = splice(tmp_path / "cut", tmp_path / "out", SpliceOptions.preset("ipad"), log=lambda *_: None)
    assert len({p["scale"] for p in r["placements"]}) == 1
    assert len({p["x"] for p in r["placements"]}) == 1


# ------------------------------------------------------------- erase


def test_erase_plate_number_everywhere(tmp_path):
    specs = [dict(kind="piano", systems=4, seed=s, skew=k) for s, k in ((1, 0), (2, 0.4), (3, -0.3))]
    pages, truths = write_pages(tmp_path / "in", specs)
    x0, y0, x1, y1 = truths[0].plate_box
    box = (x0 - 4, y0 - 4, x1 - x0 + 8, y1 - y0 + 8)
    tmpl = template_from_page(np.asarray(Image.open(pages[0])), box)
    report = erase(pages, tmp_path / "out", [tmpl], log=lambda *_: None)
    for page, src, truth in zip(report["pages"], pages, truths):
        assert len(page["hits"]) == 1
        before = np.asarray(Image.open(src))
        after = np.asarray(Image.open(tmp_path / "out" / page["output"]))
        bx0, by0, bx1, by1 = truth.plate_box
        assert (after[by0 - 20:by1 + 20, bx0 - 20:bx1 + 20] < 128).sum() == 0
        changed = np.argwhere(before != after)
        assert changed[:, 0].min() > by0 - 30  # nothing above the plate line was touched


def test_erase_finds_nothing_when_absent():
    with_plate, truth = make_page(kind="violin", systems=6, seed=1)
    without, _ = make_page(kind="violin", systems=6, seed=1, plate=False)
    x0, y0, x1, y1 = truth.plate_box
    tmpl = with_plate[y0 - 4:y1 + 4, x0 - 4:x1 + 4]
    assert find_matches(without, tmpl) == []


# ------------------------------------------------------------- end to end


def test_run_from_scanned_pdf(tmp_path):
    pages, truths = write_pages(tmp_path / "in", [dict(kind="piano", systems=4, title=True, seed=1),
                                                  dict(kind="piano", systems=5, seed=2)])
    pdf = tmp_path / "score.pdf"
    ims = [Image.open(p) for p in pages]
    ims[0].save(pdf, save_all=True, append_images=ims[1:], resolution=230)
    x0, y0, x1, y1 = truths[0].plate_box
    box = f"1:{x0 - 4},{y0 - 4},{x1 - x0 + 8},{y1 - y0 + 8}"
    assert main(["run", str(pdf), "-o", str(tmp_path / "run"), "--preset", "letter", "--erase-from", box]) == 0
    m = json.loads((tmp_path / "run" / "systems" / "cut.json").read_text())
    assert systems_per_page(m) == {1: 4, 2: 5}
    assert [it["kind"] for it in m["items"]].count("extra") == 1  # title only; plate numbers erased
    assert (tmp_path / "run" / "score-letter.pdf").exists()
