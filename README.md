# Bible TTS Resources

A toolkit for creating Text-to-Speech (TTS) datasets from Bible audio recordings. This project downloads, aligns, and processes Bible audio/text pairs from [Open.Bible](https://open.bible/) into verse-level segments suitable for TTS training.

The full list of available languages is in [Bible audio resources clean](https://docs.google.com/spreadsheets/d/1P4xk-MgjP7nxWTuo8-pTmVTeICUw1dQOT7GXmjgjsSc/edit?gid=830487007#gid=830487007).

## Features

- **Download** Bible audio files and text transcripts from Open.Bible
- **Align** audio with text using two methods:
  - **Timing files**: When timing files are available from Biblica
  - **Forced alignment**: Using [ReadAlongs Studio](https://github.com/ReadAlongs/Studio) when timing files are not available
- **Split** chapter-level audio into individual verse segments
- **Analyze** audio statistics and data quality

## Installation

### Using Conda (Recommended)

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate bible-tts-resources
```

### Using pip

```bash
pip install -r requirements.txt
```

### Additional Dependencies

Some alignment methods require `ffmpeg` and `sox`:

```bash
# Using Conda (recommended)
conda install -c conda-forge ffmpeg sox

# Ubuntu/Debian
sudo apt-get install ffmpeg sox libsox-fmt-mp3
```

## Usage

### 1. Download Data

#### Download Audio Files

Place HTML files containing artifact links in `html_files/audio/`, then use the notebook or run:

```python
from utils import download_audios

# Extract links from HTML file
links = download_audios.extract_artifact_links("html_files/audio/Yoruba.html")

# Download and extract all files
download_audios.download_and_unzip_all(links, "data/audios/Yoruba")
```

#### Download Text Files

```python
from utils import download_texts

# Extract links from HTML file
links = download_texts.extract_artifact_links("html_files/text/Yoruba.html")

# Download and extract all files
download_texts.download_and_unzip_all(links, "data/texts/Yoruba")
```

### 2. Align Audio with Text

#### Method A: Using Timing Files (Preferred)

For languages with timing files from Biblica, alignment is more accurate:

```bash
# Process a single book
python utils/split_verse_with_timing.py \
    -wav_folder "data/audios/Yoruba/New Testament - mp3/Matthew" \
    -timing_folder "data/audios/Yoruba/Timing Files/Timing Files Bundle" \
    -book_sfm "data/texts/Yoruba/Paratext (USFM)/release/USX_1/MAT.usfm" \
    -output "data/audios/Yoruba/Alignment/Matthew"

# Process all books for a language
python utils/process_all_books_with_timing.py \
    --base_path "data/audios/Yoruba" \
    --timing_folder "data/audios/Yoruba/Timing Files/Timing Files Bundle" \
    --usfm_folder "data/texts/Yoruba/Paratext (USFM)/release/USX_1" \
    --workers 8
```

#### Method B: Using Forced Alignment (ReadAlongs)

For languages without timing files, use zero-shot forced alignment:

```bash
# Process a single book
python utils/force_align_book.py \
    -audio_folder "data/audios/Swahili/New Testament - mp3/Matthew" \
    -book_usx "data/texts/Swahili/USX/release/USX_1/MAT.usx" \
    -output "data/audios/Swahili/Alignment/Matthew" \
    -language "und"

# Process all books for a language
python utils/process_all_books_force_align.py \
    --base_path "data/audios/Swahili" \
    --usfm_folder "data/texts/Swahili/USX/release/USX_1" \
    --language "und" \
    --workers 4 \
    --chapter-intro "chapter introduction"
```

**Options:**
- `-language`: Language code for g2p (use `"und"` for undetermined)
- `--chapter-intro`: Placeholder text to absorb speaker's chapter announcements
- `--dry-run`: Preview what would be processed without running

### 3. Analyze Audio Statistics

```python
from utils import audio_stats

# Get all audio files with durations
df = audio_stats.get_all_audio_files(
    audios_dir="data/audios",
    alignment_filter="only"  # "exclude", "only", or "all"
)

# View statistics
print(f"Total duration: {df['duration_seconds'].sum() / 3600:.2f} hours")
print(df.groupby('language')['duration_seconds'].sum())
```

### 4. Check Data Quality

Use the data checks module to identify outliers and validate TTS datasets:



```python
from utils import data_checks

# Remove outliers from alignment DataFrame
# Returns DataFrame with a 'label' column classifying each sample
alignment_df = data_checks.remove_outliers(alignment_df, num_std_devs=2.0)

# View label distribution
print(alignment_df.label.value_counts())
```

The checker validates and labels samples as:
- `BEST`: Clean samples suitable for TTS training
- `TOO_LONG`: Audio clips over 30 seconds
- `TOO_SHORT_TRANS`: Transcripts under 10 characters
- `OFFENDING_DATA`: Pairs with more text than audio (bad for CTC)
- `NON_NORMAL`: Pairs outside the specified standard deviations from mean ratio

### 5. Speaker Diarization and Upload

Several languages contain recordings from more than one speaker. Speaker labels are assigned using [pyannote/speaker-diarization-precision-2](https://huggingface.co/pyannote/speaker-diarization-precision-2) before uploading to the Hub. The first verse of every Bible book is concatenated into a 10–15 minute reference file, diarization is run on it, and the dominant speaker for each book is propagated to all its verses as a `speaker_id` metadata column.

To run diarization and upload to Hugging Face, set the `PYANNOTE_TOKEN` environment variable to a valid token with access to the pyannote model:

```bash
export PYANNOTE_TOKEN="hf_..."
python upload_to_hf.py
```

If the token is not set, diarization is skipped and `speaker_id` is not included.

## Output Format

After alignment, each verse is saved as:
- `{BOOK}_{CHAPTER:03d}_Verse_{VERSE:03d}.wav` - Audio segment (22050 Hz, mono, 16-bit)
- `{BOOK}_{CHAPTER:03d}_Verse_{VERSE:03d}.txt` - Text transcript

Example:
```
MAT_001_Verse_001.wav
MAT_001_Verse_001.txt
MAT_001_Verse_002.wav
MAT_001_Verse_002.txt
...
```

## Supported Languages

Languages with **timing files** (26 languages):
Assamese, Bengali, Central Kurdish, Chhattisgarhi, Dholuo, Ewe, Gujarati, Hausa, Hiligaynon, Hindi, Igbo, Kannada, Lingala, Malayalam, Marathi, Ndebele, Nepali, Oromo, Punjabi, Tamil, Telugu, Twi (Akuapem), Twi (Asante), Urdu, Vietnamese, Yoruba

Languages using **forced alignment** (11 languages):
Arabic Standard, Chichewa, Dawro, Gamo, Gofa, Haitian Creole, Kikuyu, Luganda, Shona, Swahili, Turkish

## Alignment Quality

For the 26 languages that ship timing files, the forced aligner can be scored against
those human-authored verse markers. `jobs/align_full_timing_languages.sh` aligns
**every chapter of all 66 books** per language in the production configuration
(`--language und`), then compares each predicted verse boundary against its marker:

```bash
cd jobs && sbatch align_full_timing_languages.sh   # ~18 h, scores on completion
# or score an existing run:
python utils/score_alignment_pilot.py \
    --pilot-root openbibletts_alignment_full --repo-root .
```

**803,158 verse boundaries.** All values in milliseconds, `Δ = predicted − reference`:
`med`/`mean` are median and mean of |Δ|; `bias` is the mean *signed* Δ (positive =
aligner lands late); `gross%` is the share beyond 1000 ms; `fine` is the median of the
rest; `<N` is the share within N ms. `n` totals in the bottom row, the rest are medians.

| Language | Script | n | med | mean | bias | gross% | fine | <100 | <250 | <500 | <1000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Central Kurdish | Arabic | 31064 | 53 | 701 | +147 | 9.8 | 47 | 75 | 84 | 87 | 90 |
| Twi (Asante) | Latin | 31053 | 58 | 452 | +260 | 6.2 | 57 | 88 | 92 | 93 | 94 |
| Twi (Akuapem) | Latin | 30713 | 59 | 480 | +290 | 7.0 | 58 | 87 | 91 | 92 | 93 |
| Igbo | Latin | 31060 | 59 | 470 | +23 | 6.7 | 58 | 87 | 93 | 93 | 93 |
| Vietnamese | Latin | 30981 | 60 | 610 | -88 | 10.6 | 50 | 76 | 87 | 88 | 89 |
| Ewe | Latin | 30740 | 67 | 991 | +669 | 8.1 | 65 | 72 | 88 | 91 | 92 |
| Hausa | Latin | 31081 | 102 | 404 | -98 | 5.7 | 99 | 48 | 93 | 94 | 94 |
| Yoruba | Latin | 31046 | 130 | 513 | +41 | 7.9 | 126 | 20 | 87 | 90 | 92 |
| Hiligaynon | Latin | 29844 | 155 | 3485 | +2449 | 22.5 | 137 | 11 | 71 | 77 | 78 |
| Lingala | Latin | 31159 | 166 | 650 | -139 | 9.1 | 156 | 15 | 78 | 90 | 91 |
| Ndebele | Latin | 30854 | 250 | 1816 | +1199 | 15.2 | 220 | 6 | 51 | 79 | 85 |
| Kannada | Kannada | 30955 | 265 | 518 | +167 | 4.6 | 260 | 4 | 44 | 94 | 95 |
| Marathi | Devanagari | 31012 | 266 | 490 | +101 | 5.0 | 261 | 1 | 41 | 92 | 95 |
| Tamil | Tamil | 31100 | 285 | 442 | +264 | 3.2 | 282 | 1 | 32 | 94 | 97 |
| Urdu | Arabic | 31031 | 320 | 593 | +376 | 4.3 | 315 | 0 | 18 | 91 | 96 |
| Telugu | Telugu | 29841 | 341 | 827 | +51 | 7.3 | 329 | 1 | 23 | 80 | 93 |
| Dholuo | Latin | 30985 | 345 | 865 | +507 | 7.1 | 335 | 0 | 12 | 80 | 93 |
| Punjabi | Gurmukhi | 31079 | 369 | 468 | +278 | 3.7 | 366 | 0 | 4 | 87 | 96 |
| Bengali | Bengali | 31063 | 374 | 721 | +260 | 6.3 | 361 | 0 | 18 | 79 | 94 |
| Malayalam | Malayalam | 30289 | 399 | 1163 | -113 | 11.0 | 373 | 4 | 21 | 68 | 89 |
| Oromo | Latin | 31070 | 410 | 1002 | +184 | 9.1 | 390 | 0 | 14 | 66 | 91 |
| Gujarati | Gujarati | 31087 | 415 | 1230 | -268 | 7.8 | 405 | 0 | 4 | 75 | 92 |
| Hindi | Devanagari | 31104 | 430 | 836 | +67 | 6.3 | 421 | 0 | 5 | 74 | 94 |
| Assamese | Bengali | 31012 | 443 | 672 | +249 | 5.3 | 438 | 0 | 3 | 77 | 95 |
| Nepali | Devanagari | 30898 | 556 | 732 | +261 | 5.7 | 548 | 0 | 1 | 29 | 94 |
| Chhattisgarhi | Devanagari | 31037 | 560 | 1851 | +1123 | 13.4 | 529 | 0 | 3 | 37 | 87 |
| **Median of languages** | | **803158** | **275** | **686** | **+216** | **7.1** | **272** | **3** | **36** | **87** | **93** |

Notes:

- Agreement splits along **script**: Latin 116 ms median, Arabic 186 ms, Indic 386 ms.
  Source audio is a uniform 48 kHz mono, so bandwidth does not explain it — the `und`
  g2p fallback estimates Brahmic grapheme-to-phone timing poorly. Indic is looser but
  has *fewer* gross failures (6.0% vs 8.0%).
- Hiligaynon, Ndebele and Chhattisgarhi show a large positive bias with a high gross
  rate — cumulative drift within chapters. Treat them separately rather than pooling.
- The markers are production artefacts for audio-Bible navigation, not phonetic
  annotation, so this measures **agreement between two segmentations**, not accuracy.
  Tolerances are looser than the 10–25 ms used for phone alignment because verse
  boundaries sit inside inter-verse pauses.
- 7 chapters of ~30,900 (0.023%) failed with `"only noise or silence segments"`. Four
  are the longest chapter in their book (incl. Psalm 119, 176 verses), so this is an
  aligner capacity limit above roughly 60 verses per utterance, not a memory problem.

## References

- [coqui-ai/open-bible-scripts](https://github.com/coqui-ai/open-bible-scripts) - Template for MFA-based alignment
- [ReadAlongs Studio](https://github.com/ReadAlongs/Studio) - Zero-shot text-speech alignment
- [coqui-ai/data-checker](https://github.com/coqui-ai/data-checker) - Data quality validation for TTS
- [Open.Bible](https://open.bible/) - Source of Bible audio recordings

## License

See [LICENSE](LICENSE) for details.
