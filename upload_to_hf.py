import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from IPython.display import Audio as IPythonAudio, display
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from utils import download_audios, download_texts
from utils import usx_parser
from utils import audio_stats
from utils import data_checks
from utils import speaker_identifier
from utils.hf_preprocessing import get_alignment_dataframe, upload_alignment_to_hf
# from utils import force_align_book

LANGUAGES = [
    'Apali',
    'Arabic Standard',
    'Assamese',
    'Bengali',
    'Central Kurdish',
    'Chhattisgarhi',
    'Chichewa',
    'Dawro',
    'Dholuo',
    'Ewe',
    'Gamo',
    'Gofa',
    'Gujarati',
    'Haitian Creole',
    'Hausa',
    'Hiligaynon',
    'Hindi',
    'Igbo',
    'Kannada',
    'Kikuyu',
    'Lingala',
    'Luganda',
    'Malayalam',
    'Maori',
    'Matengo',
    'Marathi',
    'Ndebele',
    'Nepali',
    'Oromo',
    'Punjabi',
    'Shona',
    'Swahili',
    'Tamil',
    'Telugu',
    'Turkish',
    'Twi (Akuapem)',
    'Twi (Asante)',
    'Ukrainian',
    'Urdu',
    'Vietnamese',
    'Yoruba'
]
base_dir = "data/audios"

for language in tqdm(LANGUAGES, desc="Uploading languages"):
    print(f"\n--- Processing '{language}' ---")
    try:
        alignment_df = get_alignment_dataframe(language, base_dir)

        repo_id = "davidguzmanr/open-bible-resources" 

        upload_alignment_to_hf(
            alignment_df=alignment_df,
            language=language,
            repo_id=repo_id,
            private=False,
        )
    except Exception as e:
        print(f"✗ Error uploading '{language}': {e}")
    else:
        print(f"✓ Finished uploading '{language}'")