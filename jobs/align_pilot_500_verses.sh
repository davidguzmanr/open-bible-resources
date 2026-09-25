#!/usr/bin/env bash
#SBATCH --job-name=align_pilot
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=david.guzman@mila.quebec

# ---------------------------------------------------------------------------
# Alignment-validation pilot.
#
# Runs ReadAlongs forced alignment over a FIXED set of 6 New Testament books
# (501 verses / 23 chapters) for each of the 26 languages that DO have
# Open.Bible timing metadata. Those timing files provide reference verse
# boundaries, so the forced-aligner output can be scored against them.
#
# The aligner runs in exactly the production configuration used for the 11
# forced-aligned languages (--language und, same --chapter-intro), so the
# resulting error distribution characterises the pipeline as actually applied
# to the languages that have no reference boundaries.
#
# --save-temps retains the Praat TextGrids, which carry the word and verse
# timings the analysis step compares against the timing markers.
#
# ONE JOB, ONE GPU. All 26 languages x 6 books = 156 alignments are processed
# in this single task through one bounded worker pool. This is NOT a job array:
# the workload is entirely CPU-bound (force_align_book.py shells out to
# `readalongs align`, which decodes with SoundSwallower -- no torch, no CUDA
# anywhere in the alignment path), so the GPU is only a ticket onto the `long`
# partition while long-cpu is down for its OS update. Requesting more than one
# would mean more idle GPUs, not more throughput; the real parallelism comes
# from --cpus-per-task.
#
# Once long-cpu is back: set --partition=long-cpu and delete the --gres line.
#
# Submit from THIS directory:
#   cd jobs && sbatch align_pilot_500_verses.sh
# Logs land in jobs/logs/ (must already exist -- Slurm opens the output file
# before the script runs and will not create it).
#
# Environment overrides:
#   DRY_RUN=1              resolve and print every path, align nothing
#   LANGUAGES="A,B"        restrict to a comma-separated subset
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

# Defaults to openbibletts_alignment_pilot/ inside the repo (git-ignored).
# $REPO is resolved above, so this is correct regardless of the submit directory.
OUT_ROOT="${OUT_ROOT:-$REPO/openbibletts_alignment_pilot}"
DRY_RUN="${DRY_RUN:-0}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

# Production forced-alignment settings -- keep in sync with
# jobs/align_with_forced_alignment.sh or the validation stops being comparable.
G2P_LANG="und"
CHAPTER_INTRO="chapter introduction"

# Books: 501 verses, 23 chapters. Same books for every language.
#   PHP 104v/4ch  COL 95v/4ch  1TH 89v/5ch  2TI 83v/4ch  1PE 105v/5ch  JUD 25v/1ch
# BOOK_DIRS is index-parallel to BOOKS and names the audio directory, which is
# only needed to locate a per-book zip when chapter audio is not extracted.
BOOKS=(PHP COL 1TH 2TI 1PE JUD)
BOOK_DIRS=("Philippians" "Colossians" "1 Thessalonians" "2 Timothy" "1 Peter" "Jude")

# The 26 languages segmented from timing files (see Table 1 correction:
# Gamo and Luganda are force-aligned, so they are NOT in this list).
LANGS=(
  "Assamese" "Bengali" "Central Kurdish" "Chhattisgarhi" "Dholuo" "Ewe"
  "Gujarati" "Hausa" "Hiligaynon" "Hindi" "Igbo" "Kannada" "Lingala"
  "Malayalam" "Marathi" "Ndebele" "Nepali" "Oromo" "Punjabi" "Tamil"
  "Telugu" "Twi (Akuapem)" "Twi (Asante)" "Urdu" "Vietnamese" "Yoruba"
)

# Optional subset, e.g. LANGUAGES="Hausa,Yoruba"
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

# Reported only to confirm the allocation; the aligner never touches the GPU.
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd "$REPO" || { echo "ERROR: cannot cd to $REPO" >&2; exit 2; }
if [[ ! -f utils/force_align_book.py ]]; then
  echo "ERROR: $REPO does not look like the repo root (utils/force_align_book.py missing)." >&2
  echo "       Submit from the jobs/ directory: cd jobs && sbatch align_pilot_500_verses.sh" >&2
  exit 2
fi
echo "Repo root: $REPO"

##################################################################
# Path resolution
#
# Layouts are not uniform across languages, so resolve by locating the files
# themselves rather than assuming a fixed directory shape:
#   - USX lives under USX_1 for most languages but USX_2 for Igbo and Ndebele;
#     Hausa ships both (sort picks USX_1, matching production).
#   - Chapter audio sits in "New Testament - mp3/<Book Name>/" for 25
#     languages, but Nepali nests it under
#     "New Testament - mp3/p_01-book-<code>/release/audio_1/p_01-book-<code>/".
#   - Chhattisgarhi ships 24-bit WAV under source/source/<CODE>/, not MP3.
##################################################################
resolve_audio_dir() {  # $1=language  $2=book code
  find "data/audios/$1" -type f \( -name "$2_001.mp3" -o -name "$2_001.wav" \) \
       -printf '%h\n' 2>/dev/null | sort | head -1
}

resolve_usx() {        # $1=language  $2=book code
  find "data/texts/$1/USX/release" -type f -name "$2.usx" 2>/dev/null | sort | head -1
}

# Safety net: extract a book's zip in place if its chapter audio is missing.
# Currently a no-op for all 26 languages, but kept so a stale or partial unpack
# cannot silently reduce the pilot's coverage. Skipped entirely under DRY_RUN.
ensure_extracted() {   # $1=language  $2=book code  $3=audio directory name
  [[ -n "$(resolve_audio_dir "$1" "$2")" ]] && return 0
  local book_dir="data/audios/$1/New Testament - mp3/$3"
  local zip
  zip="$(find "$book_dir" -maxdepth 1 -type f -name '*.zip' 2>/dev/null | sort | head -1)"
  [[ -z "$zip" ]] && return 1
  echo "[$1/$2] chapter audio missing; extracting $(basename "$zip")"
  unzip -n -q "$zip" -d "$book_dir" || return 1
  [[ -n "$(resolve_audio_dir "$1" "$2")" ]]
}

##################################################################
# Phase 1: resolve the full work queue (fast, serial)
##################################################################
echo "=========================================================="
echo "Languages : ${#LANGS[@]}     Books: ${BOOKS[*]}"
echo "Output    : $OUT_ROOT"
echo "G2P       : $G2P_LANG    Chapter intro: '$CHAPTER_INTRO'"
echo "Workers   : $JOBS        Dry run: $DRY_RUN"
echo "=========================================================="

declare -a W_LANG=() W_CODE=() W_AUDIO=() W_USX=() W_OUT=()
unresolved=0

for lang in "${LANGS[@]}"; do
  lang_out="$OUT_ROOT/$lang"
  mkdir -p "$lang_out"
  manifest="$lang_out/manifest.csv"
  echo "language,book_code,audio_dir,usx_file,output_dir,status" > "$manifest"

  for i in "${!BOOKS[@]}"; do
    code="${BOOKS[$i]}"
    [[ "$DRY_RUN" != "1" ]] && { ensure_extracted "$lang" "$code" "${BOOK_DIRS[$i]}" || true; }
    audio_dir="$(resolve_audio_dir "$lang" "$code")"
    usx_file="$(resolve_usx "$lang" "$code")"
    book_out="$lang_out/$code"

    if [[ -z "$audio_dir" || -z "$usx_file" ]]; then
      echo "[$lang/$code] UNRESOLVED (audio='${audio_dir:-NONE}' usx='${usx_file:-NONE}')"
      echo "\"$lang\",$code,\"${audio_dir:-}\",\"${usx_file:-}\",\"$book_out\",unresolved" >> "$manifest"
      unresolved=$((unresolved+1))
      continue
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[$lang/$code] OK  audio=$audio_dir"
      echo "\"$lang\",$code,\"$audio_dir\",\"$usx_file\",\"$book_out\",dry-run" >> "$manifest"
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
# Phase 2: align the queue with a single bounded worker pool
#
# One pool across all languages (not per-language batches) keeps every CPU
# busy to the end instead of draining at each language boundary. Books are
# independent -- separate output directories -- so this is safe. Each
# readalongs decode is single-threaded, so JOBS is the real parallelism.
##################################################################
(( JOBS > ${#W_LANG[@]} )) && JOBS=${#W_LANG[@]}
echo "Aligning ${#W_LANG[@]} books with up to $JOBS concurrent workers..."

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
  ) &
done
wait

##################################################################
# Phase 3: collect results
##################################################################
fail=0
for i in "${!W_LANG[@]}"; do
  lang="${W_LANG[$i]}"; code="${W_CODE[$i]}"; book_out="${W_OUT[$i]}"
  status="$(cat "$book_out/.status" 2>/dev/null || echo failed)"
  [[ "$status" != "ok" ]] && { fail=1; echo "[$lang/$code] FAILED - see $book_out/align.log"; }
  echo "\"$lang\",$code,\"${W_AUDIO[$i]}\",\"${W_USX[$i]}\",\"$book_out\",$status" \
    >> "$OUT_ROOT/$lang/manifest.csv"
done

echo "=========================================================="
echo "Per-language totals (expected 501 verses / 23 TextGrids each):"
for lang in "${LANGS[@]}"; do
  printf "  %-18s verses=%-5s textgrids=%s\n" "$lang" \
    "$(find "$OUT_ROOT/$lang" -name '*.wav' 2>/dev/null | wc -l)" \
    "$(find "$OUT_ROOT/$lang" -name '*.TextGrid' 2>/dev/null | wc -l)"
done
echo "Unresolved books: $unresolved"
echo "Elapsed: $(( ($(date +%s) - START_TIME) / 60 )) min"
echo "Job ${SLURM_JOB_ID:-local} finished at $(date) with fail=$fail"
exit $(( fail || unresolved > 0 ))
