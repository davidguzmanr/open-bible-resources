import os

import torch
from tqdm import tqdm
from pyannote.audio import Pipeline

from utils.hf_preprocessing import (
    get_alignment_dataframe,
    prepare_alignment_dataset,
    upload_alignment_to_hf,
)
from utils.diarization import add_speaker_ids

LANGUAGES = [
    # 'Apali',
    # 'Arabic Standard',
    # 'Assamese',
    # 'Bengali',
    # 'Central Kurdish',
    # 'Chhattisgarhi',
    # 'Chichewa',
    # 'Dawro',
    # 'Dholuo',
    # 'Ewe',
    # 'Gamo',
    # 'Gofa',
    # 'Gujarati',
    # 'Haitian Creole',
    # 'Hausa',
    # 'Hiligaynon',
    # 'Hindi',
    # 'Igbo',
    # 'Kannada',
    # 'Kikuyu',
    # 'Lingala',
    # 'Luganda',
    # 'Malayalam',
    # 'Maori',
    # 'Matengo',
    # 'Marathi',
    # 'Ndebele',
    # 'Nepali',
    # 'Oromo',
    'Polish',
    # 'Punjabi',
    # 'Shona',
    # 'Swahili',
    # 'Tamil',
    # 'Telugu',
    # 'Turkish',
    # 'Twi (Akuapem)',
    # 'Twi (Asante)',
    # 'Ukrainian',
    # 'Urdu',
    # 'Vietnamese',
    # 'Yoruba',
]

BASE_DIR = "data/audios"
REPO_ID = "davidguzmanr/open-bible-resources"
PYANNOTE_TOKEN = os.environ.get("PYANNOTE_TOKEN", "")

# Load diarization pipeline once (expensive operation)
pipeline = None
if not PYANNOTE_TOKEN:
    print("Warning: PYANNOTE_TOKEN is not set. Skipping diarization (step 3).\n")
else:
    try:
        print("Loading pyannote diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-precision-2",
            token=PYANNOTE_TOKEN,
        )
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Pipeline loaded.\n")
    except Exception as e:
        print(f"Warning: Failed to load pyannote pipeline ({e}). Skipping diarization (step 3).\n")

for language in tqdm(LANGUAGES, desc="Uploading languages"):
    print(f"\n--- Processing '{language}' ---")
    try:
        # Step 1: Build a DataFrame of (audio_file, text, book, chapter, verse,
        # testament, duration_seconds) by scanning the local Alignment folder.
        alignment_df = get_alignment_dataframe(language, base_dir=BASE_DIR)

        # Step 2: Convert the DataFrame to an HF Dataset, casting the audio
        # column to Audio() and removing statistical outliers.
        ds = prepare_alignment_dataset(alignment_df)

        # Step 3: Concatenate the first verse of each book into a single audio
        # file, run pyannote diarization on it, and map each book to the
        # speaker that dominates its time window.  The result is the same
        # dataset with an extra `speaker_id` column.
        if pipeline is not None:
            ds = add_speaker_ids(
                ds=ds,
                pipeline=pipeline,
                language=language,
                output_dir="diarization",
            )

        # Step 4: Split into train/test and push both splits to the Hub under
        # a config named after the language.
        upload_alignment_to_hf(
            dataset=ds,
            language=language,
            repo_id=REPO_ID,
            private=False,
        )
    except Exception as e:
        print(f"✗ Error uploading '{language}': {e}")
    else:
        print(f"✓ Finished uploading '{language}'")
