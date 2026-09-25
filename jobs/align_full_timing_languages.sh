#!/usr/bin/env bash
#SBATCH --job-name=align_full
#SBATCH --partition=long-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# ---------------------------------------------------------------------------
# Full-corpus alignment validation.
#
# The same measurement as jobs/align_pilot_500_verses.sh, but over the WHOLE
# Bible instead of 6 New Testament books: every chapter of all 66 books, for
# each of the 26 languages that ship Open.Bible timing metadata. Those timing
# files give reference verse boundaries, so forced-aligner output can be scored
# against them (see utils/score_alignment_pilot.py).
#
# The aligner runs in exactly the production configuration used for the 11
# force-aligned languages (--language und, same --chapter-intro), so the error
# distribution characterises the pipeline as actually applied to the languages
# that have no reference boundaries.
#
# Scale: 26 languages x 66 books = 1,716 books / ~30,900 chapters / ~800k verses.
#
# ONE JOB, NOT A JOB ARRAY. All 1,716 books go through a single bounded worker
# pool in this one task. The work is CPU-bound and embarrassingly parallel at
# the book level, so a large --cpus-per-task on one node does the same work as
# an array of small tasks while occupying one scheduler slot instead of 26.
# One global pool also beats per-language batching: the pool stays saturated to
# the end instead of draining at every language boundary, and the runtime floor
# set by the longest serial book (Psalms, 150 chapters) is paid once rather
# than once per language.
#
# DISK: the aligner re-cuts every verse to WAV as a side effect. At full-corpus
# scale that is ~800k files and several hundred GB, none of which the scoring
# step reads -- it only needs the TextGrids. So verse WAV/TXT are deleted after
# each book completes, leaving ~2-3 GB of TextGrids. Set KEEP_AUDIO=1 to retain
# them, and make sure there is room first.
#
# Submit from THIS directory:
#   cd jobs && sbatch align_full_timing_languages.sh
# Logs land in jobs/logs/ (must already exist -- Slurm opens the output file
# before the script runs and will not create it).
#
# Environment overrides:
#   DRY_RUN=1              resolve and print every path, align nothing
#   LANGUAGES="A,B"        restrict to a comma-separated subset
#   BOOKS="PSA,GEN"        restrict to specific book codes
#   KEEP_AUDIO=1           keep the re-cut verse WAV/TXT (needs ~300 GB)
#   JOBS=8                 worker-pool size (default: --cpus-per-task)
#   OUT_ROOT=/some/path    output location
# ---------------------------------------------------------------------------

set -uo pipefail

START_TIME=$(date +%s)
echo "Job ${SLURM_JOB_ID:-local} starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: ${SLURM_NODELIST:-n/a}"

# Repo root is the parent of the submit directory (jobs/). SLURM_SUBMIT_DIR is
# the reliable anchor: Slurm may run the script from a spool copy, so $0 and
# BASH_SOURCE cannot be trusted. Falls back to the checkout's usual location.
REPO="$(cd "${SLURM_SUBMIT_DIR:-$PWD}/.." 2>/dev/null && pwd)"
if [[ ! -f "$REPO/utils/force_align_book.py" ]]; then
  REPO="/home/mila/g/guzmand/scratch/Repositories/open-bible-resources"
fi

OUT_ROOT="${OUT_ROOT:-$REPO/openbibletts_alignment_full}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_AUDIO="${KEEP_AUDIO:-0}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

# Production forced-alignment settings -- keep in sync with
# jobs/align_with_forced_alignment.sh or the validation stops being comparable.
G2P_LANG="und"
CHAPTER_INTRO="chapter introduction"

# The 26 languages segmented from timing files. Gamo and Luganda are NOT here:
# both were force-aligned and have no reference boundaries to score against.
LANGS=(
  "Assamese" "Bengali" "Central Kurdish" "Chhattisgarhi" "Dholuo" "Ewe"
  "Gujarati" "Hausa" "Hiligaynon" "Hindi" "Igbo" "Kannada" "Lingala"
  "Malayalam" "Marathi" "Ndebele" "Nepali" "Oromo" "Punjabi" "Tamil"
  "Telugu" "Twi (Akuapem)" "Twi (Asante)" "Urdu" "Vietnamese" "Yoruba"
)

if [[ -n "${LANGUAGES:-}" ]]; then
  IFS=',' read -r -a LANGS <<< "$LANGUAGES"
  echo "Restricted to ${#LANGS[@]} language(s) from \$LANGUAGES"
fi

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
if [[ ! -f utils/force_align_book.py ]]; then
  echo "ERROR: $REPO does not look like the repo root." >&2
  echo "       Submit from jobs/: cd jobs && sbatch align_full_timing_languages.sh" >&2
  exit 2
fi
echo "Repo root: $REPO"

##################################################################
# Path resolution
#
# Layouts are not uniform, so locate files rather than assuming a shape:
#   - The USX directory is USX_1 for most languages, USX_2 for Igbo and
#     Ndebele. Hausa ships BOTH: USX_1 holds only the 27 NT books while USX_2
#     holds all 66 (byte-identical text for the books they share), so pick the
#     directory with the MOST books -- picking the first would silently drop
#     the whole Old Testament for Hausa.
#   - Chapter audio sits in "<Testament> - mp3/<Book Name>/" for 25 languages,
#     but Nepali nests it under .../p_01-book-<code>/release/audio_1/...
#   - Chhattisgarhi ships 24-bit WAV under source/source/<CODE>/, not MP3.
##################################################################
resolve_usx_dir() {    # $1=language
  local best="" n best_n=0 d
  while IFS= read -r d; do
    n=$(find "$d" -maxdepth 1 -name '*.usx' 2>/dev/null | wc -l)
    (( n > best_n )) && { best_n=$n; best="$d"; }
  done < <(find "data/texts/$1/USX/release" -mindepth 1 -maxdepth 1 -type d -name 'USX_*' 2>/dev/null | sort)
  echo "$best"
}

# Punjabi names Obadiah's audio and timing files OBD_* while its USX file is
# OBA.usx; every other language uses OBA throughout. Only 1 of 1,716
# language-book pairs needs this, so an explicit alias beats a general rule.
declare -A CODE_ALIASES=( [OBA]="OBD" )

resolve_audio_dir() {  # $1=language  $2=book code
  local hit
  for c in "$2" "${CODE_ALIASES[$2]:-}"; do
    [[ -z "$c" ]] && continue
    hit=$(find "data/audios/$1" -type f \( -name "${c}_001.mp3" -o -name "${c}_001.wav" \) \
          -printf '%h\n' 2>/dev/null | sort | head -1)
    [[ -n "$hit" ]] && { echo "$hit"; return 0; }
  done
  return 1
}

##################################################################
# Phase 1: build the global work queue (fast, serial)
##################################################################
echo "=========================================================="
echo "Languages : ${#LANGS[@]}"
echo "Output    : $OUT_ROOT"
echo "G2P       : $G2P_LANG    Chapter intro: '$CHAPTER_INTRO'"
echo "Workers   : $JOBS    Keep audio: $KEEP_AUDIO    Dry run: $DRY_RUN"
echo "=========================================================="

declare -a W_LANG=() W_CODE=() W_AUDIO=() W_USX=() W_OUT=()
unresolved=0

for lang in "${LANGS[@]}"; do
  usx_dir="$(resolve_usx_dir "$lang")"
  if [[ -z "$usx_dir" ]]; then
    echo "[$lang] ERROR: no USX directory found -- skipping language"
    unresolved=$((unresolved+1))
    continue
  fi

  mapfile -t codes < <(find "$usx_dir" -maxdepth 1 -name '*.usx' -printf '%f\n' | sed 's/\.usx$//' | sort)
  [[ -n "${BOOKS:-}" ]] && IFS=',' read -r -a codes <<< "$BOOKS"

  lang_out="$OUT_ROOT/$lang"
  mkdir -p "$lang_out"
  echo "language,book_code,audio_dir,usx_file,output_dir,status" > "$lang_out/manifest.csv"
  echo "[$lang] $(basename "$usx_dir"), ${#codes[@]} books"

  for code in "${codes[@]}"; do
    audio_dir="$(resolve_audio_dir "$lang" "$code")"
    usx_file="$usx_dir/$code.usx"
    book_out="$lang_out/$code"

    if [[ -z "$audio_dir" || ! -f "$usx_file" ]]; then
      echo "  [$lang/$code] UNRESOLVED (audio='${audio_dir:-NONE}')"
      echo "\"$lang\",$code,\"${audio_dir:-}\",\"$usx_file\",\"$book_out\",unresolved" >> "$lang_out/manifest.csv"
      unresolved=$((unresolved+1))
      continue
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "\"$lang\",$code,\"$audio_dir\",\"$usx_file\",\"$book_out\",dry-run" >> "$lang_out/manifest.csv"
      continue
    fi

    W_LANG+=("$lang"); W_CODE+=("$code"); W_AUDIO+=("$audio_dir")
    W_USX+=("$usx_file"); W_OUT+=("$book_out")
  done
done

echo "----------------------------------------------------------"
echo "Resolved ${#W_LANG[@]} book(s) to align; $unresolved unresolved."

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 -- nothing aligned. Manifests written under $OUT_ROOT."
  exit $(( unresolved > 0 ? 1 : 0 ))
fi

##################################################################
# Phase 2: one bounded worker pool over every book of every language
##################################################################
(( JOBS > ${#W_LANG[@]} )) && JOBS=${#W_LANG[@]}
echo "Aligning ${#W_LANG[@]} books with up to $JOBS concurrent workers..."
echo "(runtime floor is the longest single book: Psalms = 150 chapters, serial)"

for i in "${!W_LANG[@]}"; do
  while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n 2>/dev/null || break; done
  (
    book_out="${W_OUT[$i]}"
    mkdir -p "$book_out"
    if python utils/force_align_book.py \
          --audio_folder "${W_AUDIO[$i]}" \
          --book_usx "${W_USX[$i]}" \
          --output "$book_out" \
          --language "$G2P_LANG" \
          --chapter-intro "$CHAPTER_INTRO" \
          --save-temps > "$book_out/align.log" 2>&1; then
      echo ok > "$book_out/.status"
    else
      echo failed > "$book_out/.status"
    fi
    # Drop the re-cut verse audio as soon as the book is done: the scorer reads
    # only TextGrids, and at full-corpus scale this is the difference between
    # ~3 GB and several hundred GB.
    if [[ "$KEEP_AUDIO" != "1" ]]; then
      find "$book_out" -maxdepth 1 \( -name '*_Verse_*.wav' -o -name '*_Verse_*.txt' \) -delete
    fi
  ) &
done
wait

##################################################################
# Phase 3: collect results
##################################################################
fail=0
declare -A LANG_VERSES=() LANG_FAILED=()
for i in "${!W_LANG[@]}"; do
  lang="${W_LANG[$i]}"; code="${W_CODE[$i]}"; book_out="${W_OUT[$i]}"
  status="$(cat "$book_out/.status" 2>/dev/null || echo failed)"
  if [[ "$status" != "ok" ]]; then
    fail=1
    LANG_FAILED[$lang]=$(( ${LANG_FAILED[$lang]:-0} + 1 ))
    echo "[$lang/$code] FAILED - see $book_out/align.log"
  fi
  n=$(grep -hoE 'Extracted [0-9]+ verse' "$book_out/align.log" 2>/dev/null | awk '{s+=$2} END{print s+0}')
  LANG_VERSES[$lang]=$(( ${LANG_VERSES[$lang]:-0} + n ))
  echo "\"$lang\",$code,\"${W_AUDIO[$i]}\",\"${W_USX[$i]}\",\"$book_out\",$status" \
    >> "$OUT_ROOT/$lang/manifest.csv"
done

echo "=========================================================="
printf "%-18s %10s %10s %9s\n" "Language" "verses" "TextGrids" "booksFail"
total_v=0; total_tg=0
for lang in "${LANGS[@]}"; do
  tg=$(find "$OUT_ROOT/$lang" -name '*.TextGrid' 2>/dev/null | wc -l)
  v=${LANG_VERSES[$lang]:-0}
  printf "%-18s %10s %10s %9s\n" "$lang" "$v" "$tg" "${LANG_FAILED[$lang]:-0}"
  total_v=$((total_v+v)); total_tg=$((total_tg+tg))
done
printf "%-18s %10s %10s\n" "TOTAL" "$total_v" "$total_tg"
echo "Unresolved books : $unresolved"
echo "Output size      : $(du -sh "$OUT_ROOT" 2>/dev/null | cut -f1)"

##################################################################
# Phase 4: score the alignment against the reference timing markers
#
# Compares every predicted verse boundary in the TextGrids against the
# Open.Bible timing marker for that verse, and writes the per-language
# boundary-error table. Runs in the same job so the run produces a
# measurement, not just alignment output. Scoring is cheap (minutes) and is
# not allowed to fail the job -- the TextGrids are the expensive artefact and
# scoring can always be re-run by hand with the same command.
##################################################################
echo "=========================================================="
echo "Scoring alignment against reference timing markers..."
SCORE_TXT="$OUT_ROOT/alignment_scores.txt"
SCORE_CSV="$OUT_ROOT/alignment_scores_per_chapter.csv"

if python utils/score_alignment_pilot.py \
      --pilot-root "$OUT_ROOT" \
      --repo-root "$REPO" \
      --per-chapter-csv "$SCORE_CSV" > "$SCORE_TXT" 2>&1; then
  cat "$SCORE_TXT"
  echo "Scores written to $SCORE_TXT"
  echo "Per-chapter detail in $SCORE_CSV"
else
  echo "WARNING: scoring failed; see $SCORE_TXT"
  tail -20 "$SCORE_TXT" 2>/dev/null
  echo "Re-run by hand with:"
  echo "  python utils/score_alignment_pilot.py --pilot-root $OUT_ROOT --repo-root $REPO"
fi

echo "=========================================================="
echo "Elapsed          : $(( ($(date +%s) - START_TIME) / 60 )) min"
echo "Job ${SLURM_JOB_ID:-local} finished at $(date) with fail=$fail"
exit $fail
