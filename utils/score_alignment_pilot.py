#!/usr/bin/env python3
"""
Score forced-alignment output against Open.Bible reference verse timings.

For each language in the alignment pilot this compares the verse boundaries
predicted by ReadAlongs (read from the saved Praat TextGrids) against the
human-authored timing markers shipped with the recordings, and reports the
boundary-error battery used in the forced-alignment literature:

  * median / mean absolute error, and the signed error (systematic drift)
  * tolerance accuracy -- the share of boundaries within 100/250/500/1000 ms
  * gross vs fine split -- gross misalignments (|delta| > GROSS_MS) reported
    separately from the median error of the remainder, because pooling them
    makes both numbers uninterpretable (cf. McAuliffe et al. 2017, where a
    comparable median but a much worse mean diagnosed gross failures)

Tolerances are far looser than the 10-25 ms used for phone boundaries: verse
boundaries fall inside inter-verse pauses, so the reference marker and the
aligner can both be "right" a few hundred ms apart.

The markers are production artefacts for audio-Bible navigation, not phonetic
annotation, so this measures AGREEMENT BETWEEN TWO INDEPENDENT SEGMENTATIONS
rather than accuracy against ground truth.
"""

import argparse
import os
import re
import statistics as st
import sys
from pathlib import Path

GROSS_MS = 1000.0
TOLERANCES_MS = (100, 250, 500, 1000)

# Marker files vary: most use "Verse 1", Hiligaynon uses lowercase zero-padded
# "verse 01". Times use either a comma or a period as the decimal separator.
VERSE_RE = re.compile(r"^verse\s+(\d+)\s+(\d+):(\d+):(\d+)[,.](\d+)", re.IGNORECASE)
INTERVAL_RE = re.compile(
    r'intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"'
)


def read_reference(path):
    """Verse number -> start time in seconds, from a timing marker file."""
    marks = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = VERSE_RE.match(line.strip())
            if m:
                v = int(m.group(1))
                h, mi, s = int(m.group(2)), int(m.group(3)), int(m.group(4))
                marks[v] = h * 3600 + mi * 60 + s + float("0." + m.group(5))
    return marks


def read_textgrid_sentences(path):
    """Start times of non-empty intervals on the Sentence tier."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    start = text.find('name = "Sentence"')
    if start < 0:
        return []
    end = text.find('name = "Word"', start)
    block = text[start : end if end > start else len(text)]
    return [
        float(xmin)
        for xmin, _xmax, label in INTERVAL_RE.findall(block)
        if label.strip()
    ]


def score_chapter(tg_path, ref_path):
    """Return (deltas_ms, note). deltas are predicted - reference."""
    ref = read_reference(ref_path)
    if not ref:
        return [], "no reference markers"
    sentences = read_textgrid_sentences(tg_path)
    if not sentences:
        return [], "no sentences in TextGrid"

    verses = sorted(ref)
    n = len(verses)
    # prepare_verse_text_file writes a chapter-intro placeholder line first, so
    # the usual layout is one leading sentence followed by the verses in order.
    if len(sentences) == n + 1:
        predicted = sentences[1:]
    elif len(sentences) == n:
        predicted = sentences
    elif len(sentences) > n:
        predicted = sentences[len(sentences) - n :]
    else:
        return [], f"count mismatch (ref={n}, tg={len(sentences)})"

    return [(p - ref[v]) * 1000.0 for p, v in zip(predicted, verses)], ""


def summarize(deltas):
    a = [abs(d) for d in deltas]
    gross = [d for d in a if d > GROSS_MS]
    fine = [d for d in a if d <= GROSS_MS]
    row = {
        "n": len(a),
        "median": st.median(a) if a else float("nan"),
        "mean": st.mean(a) if a else float("nan"),
        "bias": st.mean(deltas) if deltas else float("nan"),
        "gross_pct": 100.0 * len(gross) / len(a) if a else float("nan"),
        "fine_median": st.median(fine) if fine else float("nan"),
    }
    for t in TOLERANCES_MS:
        row[f"within_{t}"] = 100.0 * sum(x <= t for x in a) / len(a) if a else float("nan")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-root", required=True, help="alignment pilot output root")
    ap.add_argument("--repo-root", default=".", help="repo root holding data/audios")
    ap.add_argument("--per-chapter-csv", default=None)
    args = ap.parse_args()

    pilot = Path(args.pilot_root)
    repo = Path(args.repo_root)
    languages = sorted(p.name for p in pilot.iterdir() if p.is_dir())

    rows, per_chapter, notes = {}, [], {}
    for lang in languages:
        deltas, skipped = [], []
        # Most languages keep marker files flat in the bundle; Nepali nests them
        # under release/timingfiles/timingfiles/<CODE>/. Index by filename.
        bundle = repo / "data/audios" / lang / "Timing Files"
        ref_index = {f.stem: f for f in bundle.rglob("*.txt")} if bundle.is_dir() else {}
        for tg in sorted((pilot / lang).rglob("*.TextGrid")):
            ref = ref_index.get(tg.stem)
            if ref is None:
                skipped.append(f"{tg.stem}: no reference file")
                continue
            d, note = score_chapter(tg, ref)
            if note:
                skipped.append(f"{tg.stem}: {note}")
                continue
            deltas.extend(d)
            s = summarize(d)
            per_chapter.append((lang, tg.stem, s["n"], s["median"], s["bias"], s["gross_pct"]))
        if deltas:
            rows[lang] = summarize(deltas)
        if skipped:
            notes[lang] = skipped

    hdr = (f"{'Language':<18}{'n':>6}{'med':>8}{'mean':>8}{'bias':>8}"
           f"{'gross%':>8}{'fine':>7}" + "".join(f"{'<'+str(t):>7}" for t in TOLERANCES_MS))
    print(hdr)
    print("-" * len(hdr))
    for lang in sorted(rows, key=lambda L: rows[L]["median"]):
        r = rows[lang]
        print(f"{lang:<18}{r['n']:>6}{r['median']:>8.0f}{r['mean']:>8.0f}{r['bias']:>+8.0f}"
              f"{r['gross_pct']:>8.1f}{r['fine_median']:>7.0f}"
              + "".join(f"{r['within_'+str(t)]:>7.0f}" for t in TOLERANCES_MS))

    if rows:
        allv = [rows[L] for L in rows]
        print("-" * len(hdr))
        print(f"{'MEDIAN OF LANGS':<18}{sum(r['n'] for r in allv):>6}"
              f"{st.median([r['median'] for r in allv]):>8.0f}"
              f"{st.median([r['mean'] for r in allv]):>8.0f}"
              f"{st.median([r['bias'] for r in allv]):>+8.0f}"
              f"{st.median([r['gross_pct'] for r in allv]):>8.1f}"
              f"{st.median([r['fine_median'] for r in allv]):>7.0f}"
              + "".join(f"{st.median([r['within_'+str(t)] for r in allv]):>7.0f}"
                        for t in TOLERANCES_MS))
    print("\nAll values in milliseconds. med/mean/fine = |predicted - reference|;")
    print("bias = signed mean; gross% = share beyond "
          f"{GROSS_MS:.0f} ms; fine = median of the rest; <N = % within N ms.")

    if notes:
        print("\nChapters skipped:")
        for lang, items in sorted(notes.items()):
            print(f"  {lang}: {len(items)}")
            for it in items[:3]:
                print(f"      {it}")

    if args.per_chapter_csv:
        with open(args.per_chapter_csv, "w", encoding="utf-8") as fh:
            fh.write("language,chapter,n,median_ms,bias_ms,gross_pct\n")
            for r in per_chapter:
                fh.write(f'"{r[0]}",{r[1]},{r[2]},{r[3]:.1f},{r[4]:.1f},{r[5]:.1f}\n')
        print(f"\nPer-chapter detail: {args.per_chapter_csv}")


if __name__ == "__main__":
    sys.exit(main())
