# scoretools

Cut score pages into systems, splice them onto any page shape, and erase repeated
marks like plate numbers. These are the parts of ScoreProcessor (sproc) worth
keeping, rebuilt in Python so they run on a Mac.

Every step writes a **proof**: the original page with what the tool did drawn over
it. Open the proofs before you trust the output.

## Set up on a Mac

```sh
brew install python poppler          # poppler reads PDFs (replaces Ghostscript + pdfimages)
cd score-tools
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
scoretools --help
```

## The one-line version

```sh
scoretools run score.pdf -o out --preset ipad
```

This unpacks the PDF, cuts every page into systems, packs them onto iPad-sized
pages and writes `out/score-ipad.pdf`. It prints the proof pages to check, in order.

Scanned PDFs are extracted losslessly at their own resolution. Other PDFs are
rendered at 600 dpi. Folders of page images (PNG, JPEG, TIFF) work too.

## The three tools

### cut: pages → systems

```sh
scoretools cut pages/ -o systems
open systems/proof/index.html
```

* A system is a run of rows with staff lines in it. The cut is made through the
  white space between systems.
* A system line or barline joining its staves holds a system together. That's why
  orchestral, vocal and piano scores all work without the tool knowing which staff is which.
* Tempo marks, lyrics and dynamics join the nearer system. Titles and page numbers
  that stand apart become **extras** (grey in the proof).
* When two systems' notes overlap so that no straight cut exists, it cuts along a
  crooked **seam** around the notes. Seams are drawn in red on the proof. Check those first.
* Crooked pages are straightened first (`--no-straighten` to skip).

Output: `0001.png, 0002.png, …` plus `cut.json`, which records the page, position
and kind of each piece.

### splice: systems → new pages

```sh
scoretools splice systems -o ipad --preset ipad --pdf score-ipad.pdf
```

| preset | page | notes |
|---|---|---|
| `ipad` | 1640×2360 | portrait, iPad Air screen |
| `ipad-landscape` | 2360×1640 | |
| `ipad-2up` | 2360×1640 | two systems side by side |
| `letter`, `a4` | 300 dpi | 0.5 in margins |
| `video`, `video-4k` | 1920×1080, 3840×2160 | centred, for 16:9 frames |

Any preset takes `--size WxH`, `--columns N`, `--margin`, `--gap` / `--max-gap`
(in staff spaces), `--valign top|justify|center`, and `--extras first|all|none`
(the default keeps just the opening title).

All systems share one scale, so staff size stays even. Page breaks are chosen
together to spread the white space evenly. A scan's grey paper is lifted to white
(`--no-whiten` to keep it).

### erase: show it one, erase them all

Crop one example of the mark from a page at full resolution. In Preview: drag a
box around it, then ⌘C, then File ▸ New from Clipboard, and save as PNG. Then:

```sh
scoretools erase pages/ -o clean --template plate.png
open clean/proof/index.html
```

Or give the box directly: `--from 3:410,6120,700,90` (page 3, x, y, width, height).

* It erases only the template's own ink shape, grown by `--grow` pixels. Notes that
  happen to overlap the box survive. Use `--mode box` to clear the whole box instead.
* `--threshold` (default 0.7) is the match score needed. The proof prints each
  match's score, so you can see how close the near-misses were.
* `--band 0.9:1` searches only the bottom tenth of each page. That's faster, and
  it can't hit anything in the music.

In `run`, use `--erase-template` / `--erase-from` to erase before cutting.

## Known limits

* A page number that sits within about 2 staff spaces of a system rides along with
  it (`--attach` sets the distance). Erasing page numbers is part of the next step,
  page prep.
* A system with no system line, whose staves are joined only by barlines that stop
  short, may come out as separate staves. Check the proof.
* Tested so far on generated pages, not yet on real downloads.

## Tests

```sh
pip install -e '.[test]' && pytest
```

The tests build score pages with known answers (skew, specks, grey paper,
overlapping systems, plate numbers) and check every tool against them.
