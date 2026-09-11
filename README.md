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
those human-authored verse markers. The pilot below aligns a fixed set of 6 New
Testament books (PHP, COL, 1TH, 2TI, 1PE, JUD — 490–501 verses per language, 23
chapters) in the same configuration used in production (`--language und`), then
compares each predicted verse boundary against the corresponding marker.

Reproduce with `jobs/align_pilot_500_verses.sh`, then:

```bash
python utils/score_alignment_pilot.py \
    --pilot-root openbibletts_alignment_pilot --repo-root .
```

| Language | n | med | mean | bias | gross% | fine | <100 | <250 | <500 | <1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vietnamese | 500 | 40 | 710 | +91 | 14.0 | 30 | 81 | 84 | 84 | 86 |
| Igbo | 501 | 59 | 353 | -255 | 8.0 | 58 | 90 | 92 | 92 | 92 |
| Twi (Asante) | 501 | 59 | 179 | +8 | 3.2 | 59 | 94 | 96 | 96 | 97 |
| Twi (Akuapem) | 500 | 62 | 543 | +308 | 6.2 | 61 | 87 | 93 | 93 | 94 |
| Central Kurdish | 501 | 63 | 486 | -321 | 9.4 | 58 | 71 | 83 | 87 | 91 |
| Ewe | 500 | 65 | 341 | -55 | 5.4 | 63 | 80 | 91 | 94 | 95 |
| Hausa | 501 | 79 | 190 | -39 | 5.8 | 76 | 68 | 94 | 94 | 94 |
| Hiligaynon | 494 | 109 | 1476 | +793 | 14.8 | 104 | 37 | 85 | 85 | 85 |
| Yoruba | 501 | 114 | 415 | -213 | 10.6 | 108 | 36 | 84 | 85 | 89 |
| Ndebele | 496 | 213 | 915 | +198 | 11.3 | 180 | 15 | 54 | 79 | 89 |
| Lingala | 501 | 217 | 827 | -465 | 13.2 | 206 | 2 | 64 | 86 | 87 |
| Tamil | 501 | 222 | 272 | +192 | 1.6 | 222 | 2 | 66 | 97 | 98 |
| Kannada | 501 | 249 | 315 | +150 | 3.4 | 242 | 7 | 51 | 95 | 97 |
| Marathi | 501 | 259 | 411 | +76 | 4.2 | 256 | 1 | 44 | 93 | 96 |
| Bengali | 473 | 284 | 487 | +41 | 5.5 | 279 | 1 | 29 | 89 | 95 |
| Assamese | 501 | 291 | 421 | +215 | 4.4 | 288 | 0 | 28 | 90 | 96 |
| Dholuo | 501 | 298 | 509 | +40 | 6.2 | 292 | 1 | 23 | 93 | 94 |
| Telugu | 412 | 307 | 1856 | -1405 | 19.2 | 262 | 2 | 37 | 78 | 81 |
| Punjabi | 501 | 326 | 427 | +211 | 5.2 | 322 | 0 | 9 | 89 | 95 |
| Hindi | 501 | 358 | 529 | +145 | 6.4 | 350 | 0 | 10 | 84 | 94 |
| Urdu | 501 | 359 | 476 | +243 | 3.8 | 356 | 0 | 6 | 89 | 96 |
| Oromo | 501 | 403 | 868 | -202 | 12.2 | 368 | 1 | 21 | 66 | 88 |
| Gujarati | 501 | 432 | 608 | +166 | 8.4 | 421 | 0 | 3 | 73 | 92 |
| Malayalam | 454 | 486 | 1624 | -922 | 15.9 | 449 | 0 | 4 | 52 | 84 |
| Chhattisgarhi | 501 | 494 | 778 | -57 | 11.2 | 458 | 0 | 8 | 51 | 89 |
| Nepali | 501 | 533 | 667 | +280 | 6.4 | 523 | 0 | 0 | 38 | 94 |
| **Median of languages** | **12847** | **254** | **498** | **+58** | **6.4** | **249** | **2** | **47** | **88** | **94** |

All values are milliseconds. For each verse, `Δ = predicted start − reference marker`:

| Column | Meaning |
|---|---|
| `n` | Verse boundaries compared (the bottom row is the **total**, not a median) |
| `med` / `mean` | Median and mean of \|Δ\| |
| `bias` | Mean of **signed** Δ — positive means the aligner lands late, negative early |
| `gross%` | Share of \|Δ\| > 1000 ms (catastrophic misalignments) |
| `fine` | Median \|Δ\| excluding those gross cases |
| `<100` … `<1000` | Share of \|Δ\| within that many ms |


## References

- [coqui-ai/open-bible-scripts](https://github.com/coqui-ai/open-bible-scripts) - Template for MFA-based alignment
- [ReadAlongs Studio](https://github.com/ReadAlongs/Studio) - Zero-shot text-speech alignment
- [coqui-ai/data-checker](https://github.com/coqui-ai/data-checker) - Data quality validation for TTS
- [Open.Bible](https://open.bible/) - Source of Bible audio recordings

## License

See [LICENSE](LICENSE) for details.
