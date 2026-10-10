#!/usr/bin/env bash
#SBATCH --job-name=align_retry
#SBATCH --partition=long-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.guzman@mila.quebec

# ---------------------------------------------------------------------------
# Retry the books that failed in jobs/align_full_timing_languages.sh.
#
# Reads every manifest.csv under OUT_ROOT, collects the books whose status is
# not "ok", and re-runs only those. Nothing is hardcoded: re-running this after
# a partially successful retry picks up whatever is still failing.
#
# WHY THE RESOURCE SHAPE IS DIFFERENT FROM THE FULL RUN
# The full run died with MaxRSS 31.4 GB against a 32 GB request. Three books
# were OOM-killed, each on the LONGEST CHAPTER of its book:
#     Nepali/PSA        Psalm 119     176 verses (longest chapter in the Bible)
#     Chhattisgarhi/EZK Ezekiel 16     63 verses
#     Assamese/JER      Jeremiah 50-52 up to 64 verses
# SoundSwallower exhausts an internal block array on very long utterances and
# then loops emitting
#     ERROR: "blkarray_list.c": Block array (NxN) exhausted
# while allocating -- 41.6 million times for Chhattisgarhi/EZK, which is why
# that single align.log reached 2.8 GB. With 16 workers sharing 32 GB, one
# runaway starves the other 15 and kills the whole allocation.
#
# So this job runs FEWER workers with MUCH more memory each: 4 workers over
# 96 GB is ~24 GB per worker, versus ~2 GB in the full run. Fewer CPUs are
# needed anyway -- there are only a handful of books to redo.
#
# PER_BOOK_TIMEOUT bounds the runaway case: if the block-array loop is truly
# unbounded, more memory only delays it, so a book that exceeds the timeout is
# killed and marked failed rather than burning the job's walltime.
#
# The other four failures are NOT memory-related. They ended cleanly with
#     "Word sequence 1 had N tokens but produced 0 segments"
# which is the aligner rejecting a chapter for audio/text mismatch. Those are
# retried too, but expect them to fail again -- that is a finding about the
# data, not an infrastructure problem.
#
# Submit from THIS directory:
#   cd jobs && sbatch align_retry_failed.sh
#
# Environment overrides:
#   DRY_RUN=1                  list what would be retried, run nothing
#   PER_BOOK_TIMEOUT=6h        per-book wall clock cap (default 4h)
#   KEEP_AUDIO=1               keep the re-cut verse WAV/TXT
#   JOBS=2                     worker-pool size (default 4)
#   OUT_ROOT=/some/path        output location
#   SKIP_SCORING=1             do not re-score at the end
# ---------------------------------------------------------------------------

set -uo pipefail

START_TIME=$(date +%s)
echo "Job ${SLURM_JOB_ID:-local} starting on $(hostname) at $(date)"

REPO="$(cd "${SLURM_SUBMIT_DIR:-$PWD}/.." 2>/dev/null && pwd)"
if [[ ! -f "$REPO/utils/force_align_book.py" ]]; then
  REPO="/home/mila/g/guzmand/scratch/Repositories/open-bible-resources"
fi

OUT_ROOT="${OUT_ROOT:-$REPO/openbibletts_alignment_full}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_AUDIO="${KEEP_AUDIO:-0}"
JOBS="${JOBS:-4}"
PER_BOOK_TIMEOUT="${PER_BOOK_TIMEOUT:-4h}"
SKIP_SCORING="${SKIP_SCORING:-0}"

# Must match jobs/align_full_timing_languages.sh or the retried books are not
# comparable with the rest of the corpus.
G2P_LANG="und"
CHAPTER_INTRO="chapter introduction"

##################################################################
# Environment
##################################################################
module load miniconda/3
module load gcc/9.3.0

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

conda activate ReadAlongs

cd "$REPO" || { echo "ERROR: cannot cd to $REPO" >&2; exit 2; }
[[ -d "$OUT_ROOT" ]] || { echo "ERROR: $OUT_ROOT not found -- run the full job first." >&2; exit 2; }
echo "Repo root : $REPO"
echo "Out root  : $OUT_ROOT"

##################################################################
# Phase 1: collect failed books from the manifests
##################################################################
declare -a R_LANG=() R_CODE=() R_AUDIO=() R_USX=() R_OUT=()

while IFS=$'\t' read -r lang code audio usx out; do
  [[ -z "$lang" ]] && continue
  R_LANG+=("$lang"); R_CODE+=("$code"); R_AUDIO+=("$audio")
  R_USX+=("$usx"); R_OUT+=("$out")
done < <(python3 - "$OUT_ROOT" <<'PY'
import csv, glob, os, sys
root = sys.argv[1]
for mf in sorted(glob.glob(os.path.join(root, "*", "manifest.csv"))):
    with open(mf, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                print("\t".join([row["language"], row["book_code"],
                                 row["audio_dir"], row["usx_file"],
                                 row["output_dir"]]))
PY
)

if (( ${#R_LANG[@]} == 0 )); then
  echo "Nothing to retry: every book in every manifest is already ok."
  exit 0
fi

echo "=========================================================="
echo "Books to retry : ${#R_LANG[@]}"
for i in "${!R_LANG[@]}"; do
  echo "  ${R_LANG[$i]}/${R_CODE[$i]}"
done
echo "Workers        : $JOBS   (~$(( 96 / JOBS )) GB each)"
echo "Per-book cap   : $PER_BOOK_TIMEOUT"
echo "=========================================================="

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 -- nothing run."
  exit 0
fi

##################################################################
# Phase 2: re-run them
##################################################################
for i in "${!R_LANG[@]}"; do
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n 2>/dev/null || break; done
  (
    book_out="${R_OUT[$i]}"
    # Clear the previous attempt: stale TextGrids from the partial run would
    # otherwise be indistinguishable from this attempt's output, and the old
    # align.log can be multi-GB.
    rm -rf "$book_out"
    mkdir -p "$book_out"

    if timeout "$PER_BOOK_TIMEOUT" python utils/force_align_book.py \
          --audio_folder "${R_AUDIO[$i]}" \
          --book_usx "${R_USX[$i]}" \
          --output "$book_out" \
          --language "$G2P_LANG" \
          --chapter-intro "$CHAPTER_INTRO" \
          --save-temps > "$book_out/align.log" 2>&1; then
      echo ok > "$book_out/.status"
    else
      rc=$?
      if (( rc == 124 )); then
        echo "timeout" > "$book_out/.status"
      else
        echo failed > "$book_out/.status"
      fi
    fi
    if [[ "$KEEP_AUDIO" != "1" ]]; then
      find "$book_out" -maxdepth 1 \( -name '*_Verse_*.wav' -o -name '*_Verse_*.txt' \) -delete
    fi
  ) &
done
wait

##################################################################
# Phase 3: update manifests in place and report
##################################################################
fail=0
echo "=========================================================="
printf "%-18s %-6s %-9s %10s %10s\n" "Language" "Book" "status" "TextGrids" "verses"
for i in "${!R_LANG[@]}"; do
  lang="${R_LANG[$i]}"; code="${R_CODE[$i]}"; book_out="${R_OUT[$i]}"
  status="$(cat "$book_out/.status" 2>/dev/null || echo failed)"
  [[ "$status" != "ok" ]] && fail=1
  tg=$(find "$book_out" -name '*.TextGrid' 2>/dev/null | wc -l)
  v=$(grep -haoE 'Extracted [0-9]+ verse' "$book_out/align.log" 2>/dev/null | awk '{s+=$2} END{print s+0}')
  printf "%-18s %-6s %-9s %10s %10s\n" "$lang" "$code" "$status" "$tg" "$v"

  python3 - "$OUT_ROOT/$lang/manifest.csv" "$code" "$status" <<'PY'
import csv, sys, os
path, code, status = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
for r in rows:
    if r["book_code"] == code:
        r["status"] = status
tmp = path + ".tmp"
with open(tmp, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys(), quoting=csv.QUOTE_NONNUMERIC)
    w.writeheader()
    w.writerows(rows)
os.replace(tmp, path)
PY
done

##################################################################
# Phase 4: re-score the whole corpus with the retried books included
##################################################################
if [[ "$SKIP_SCORING" != "1" ]]; then
  echo "=========================================================="
  echo "Re-scoring alignment against reference timing markers..."
  SCORE_TXT="$OUT_ROOT/alignment_scores.txt"
  SCORE_CSV="$OUT_ROOT/alignment_scores_per_chapter.csv"
  if python utils/score_alignment_pilot.py \
        --pilot-root "$OUT_ROOT" --repo-root "$REPO" \
        --per-chapter-csv "$SCORE_CSV" > "$SCORE_TXT" 2>&1; then
    cat "$SCORE_TXT"
  else
    echo "WARNING: scoring failed; see $SCORE_TXT"
    tail -20 "$SCORE_TXT" 2>/dev/null
  fi
fi

echo "=========================================================="
echo "Elapsed : $(( ($(date +%s) - START_TIME) / 60 )) min"
echo "Job ${SLURM_JOB_ID:-local} finished at $(date) with fail=$fail"
exit $fail
