import os
import time
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from IPython.display import Audio as IPythonAudio, display
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from datasets import Dataset, Audio

from utils import download_audios, download_texts
from utils import usx_parser
from utils import audio_stats
from utils import data_checks
from utils import speaker_identifier
# from utils import force_align_book

# Book to testament mapping (constant, defined outside function)
BOOK_TO_TESTAMENT = {
    # Old Testament
    "Genesis": "Old Testament",
    "Exodus": "Old Testament",
    "Leviticus": "Old Testament",
    "Numbers": "Old Testament",
    "Deuteronomy": "Old Testament",
    "Joshua": "Old Testament",
    "Judges": "Old Testament",
    "Ruth": "Old Testament",
    "1 Samuel": "Old Testament",
    "2 Samuel": "Old Testament",
    "1 Kings": "Old Testament",
    "2 Kings": "Old Testament",
    "1 Chronicles": "Old Testament",
    "2 Chronicles": "Old Testament",
    "Ezra": "Old Testament",
    "Nehemiah": "Old Testament",
    "Esther": "Old Testament",
    "Job": "Old Testament",
    "Psalms": "Old Testament",
    "Proverbs": "Old Testament",
    "Ecclesiastes": "Old Testament",
    "Song of Songs": "Old Testament",
    "Isaiah": "Old Testament",
    "Jeremiah": "Old Testament",
    "Lamentations": "Old Testament",
    "Ezekiel": "Old Testament",
    "Daniel": "Old Testament",
    "Hosea": "Old Testament",
    "Joel": "Old Testament",
    "Amos": "Old Testament",
    "Obadiah": "Old Testament",
    "Jonah": "Old Testament",
    "Micah": "Old Testament",
    "Nahum": "Old Testament",
    "Habakkuk": "Old Testament",
    "Zephaniah": "Old Testament",
    "Haggai": "Old Testament",
    "Zechariah": "Old Testament",
    "Malachi": "Old Testament",
    # New Testament
    "Matthew": "New Testament",
    "Mark": "New Testament",
    "Luke": "New Testament",
    "John": "New Testament",
    "Acts": "New Testament",
    "Romans": "New Testament",
    "1 Corinthians": "New Testament",
    "2 Corinthians": "New Testament",
    "Galatians": "New Testament",
    "Ephesians": "New Testament",
    "Philippians": "New Testament",
    "Colossians": "New Testament",
    "1 Thessalonians": "New Testament",
    "2 Thessalonians": "New Testament",
    "1 Timothy": "New Testament",
    "2 Timothy": "New Testament",
    "Titus": "New Testament",
    "Philemon": "New Testament",
    "Hebrews": "New Testament",
    "James": "New Testament",
    "1 Peter": "New Testament",
    "2 Peter": "New Testament",
    "1 John": "New Testament",
    "2 John": "New Testament",
    "3 John": "New Testament",
    "Jude": "New Testament",
    "Revelation": "New Testament",
}


def get_alignment_dataframe(language: str, base_dir: str = "data/audios") -> pd.DataFrame:
    """
    Load alignment data for a given language and return a DataFrame.
    
    Args:
        language: The language name (e.g., "Yoruba")
        base_dir: Base directory for audio files (default: "data/audios")
    
    Returns:
        DataFrame with columns: audio_file, text_file, text, book, chapter, 
                               verse, testament, duration_seconds
    """
    alignment_dir = os.path.join(base_dir, language, "Alignment")
    
    # Collect all audio files recursively (.wav and .mp3)
    audio_files = []
    for root, dirs, files in os.walk(alignment_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):
                audio_files.append(os.path.join(root, file))
    
    # Each audio file has a corresponding .txt file with the same name
    text_files = [os.path.splitext(x)[0] + ".txt" for x in audio_files]
    
    # Build initial dataframe with file paths
    df = pd.DataFrame({
        "audio_file": audio_files,
        "text_file": text_files,
    })
    
    # Helper to safely read text file contents
    def read_text_file(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    
    # Read transcript text from each text file
    df["text"] = df["text_file"].apply(read_text_file)
    
    # Extract metadata from file path structure:
    # Format: .../Alignment/{Book}/{BOOK_CHAPTER_Verse_VERSE}.txt
    df["book"] = df["text_file"].apply(lambda x: x.split("/")[-2])
    df["chapter"] = df["text_file"].apply(lambda x: x.replace(".txt", "").split("/")[-1].split("_")[1])
    df["verse"] = df["text_file"].apply(lambda x: x.replace(".txt", "").split("/")[-1].split("_")[-1])
    
    # Map book name to testament (Old/New)
    df["testament"] = df["book"].map(BOOK_TO_TESTAMENT)
    
    # Get audio duration in seconds
    df["duration_seconds"] = df["audio_file"].apply(audio_stats.get_audio_duration)

    # Reorder columns
    df = df[["audio_file", "text", "testament", "book", "chapter", "verse", "duration_seconds"]]
    
    return df

def upload_alignment_to_hf(
    alignment_df: pd.DataFrame,
    language: str,
    repo_id: str,
    private: bool = False,
    max_shard_size: str = "1GB",
    max_retries: int = 3,
):
    """
    Upload alignment data as a TTS dataset to Hugging Face Hub.
    
    Args:
        alignment_df: DataFrame with audio_file, text, and metadata columns
        language: Language name to use as the split/config name
        repo_id: Hugging Face repository ID (e.g., "username/bible-tts")
        private: Whether the dataset should be private
        max_shard_size: Maximum shard size for upload (smaller = more reliable)
        max_retries: Number of retries on timeout errors
    """
    # Create a copy to avoid modifying the original
    df = alignment_df.copy()

    # Simple outlier removal
    df = data_checks.remove_outliers(df, num_std_devs=3.0)
    df = df[df["label"] == "BEST"]
    df = df[['audio_file', 'text', 'testament', 'book', 'chapter', 'verse', 'duration_seconds']]
    
    # Rename audio_file to audio for HF convention
    df = df.rename(columns={"audio_file": "audio"})
    
    # Create HF Dataset
    dataset = Dataset.from_pandas(df, preserve_index=False)
    
    # Cast the audio column to Audio feature (this handles loading the actual audio files)
    dataset = dataset.cast_column("audio", Audio())
    
    # Push to hub with retry logic for timeout errors
    for attempt in range(max_retries):
        try:
            dataset.push_to_hub(
                repo_id=repo_id,
                config_name=language,  # Use language as the config name
                split="train",  # HF requires a split name, but you can ignore it when loading
                private=private,
                max_shard_size=max_shard_size,  # Smaller shards for more reliable uploads
                commit_message=f"Upload {language} dataset",
            )
            print(f"✓ Uploaded {len(dataset)} samples for '{language}' to {repo_id}")
            return
        except Exception as e:
            if "timeout" in str(e).lower() or "ReadTimeout" in str(type(e).__name__):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                    print(f"⚠ Timeout on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"✗ Failed after {max_retries} attempts. The data may have uploaded - check the repo.")
                    raise
            else:
                raise

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
    # 'Haryanvi',
    'Hausa',
    'Hiligaynon',
    'Hindi',
    'Igbo',
    'Kannada',
    'Kikuyu',
    'Lingala',
    'Luganda',
    'Malayalam',
    'Marathi',
    'Ndebele',
    'Oromo',
    'Punjabi',
    'Shona',
    'Swahili',
    'Tamil',
    'Telugu',
    # 'Toma',
    'Turkish',
    'Twi (Akuapem)',
    'Twi (Asante)',
    'Ukrainian',
    'Urdu',
    'Vietnamese',
    'Yoruba'
][::-1]
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