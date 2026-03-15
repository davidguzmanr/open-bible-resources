import os
import time

import pandas as pd
from datasets import Dataset, Audio

from utils import audio_stats
from utils import data_checks

BOOK_CODE_TO_NAME = {
    # Old Testament
    "GEN": "Genesis", "EXO": "Exodus", "LEV": "Leviticus", "NUM": "Numbers",
    "DEU": "Deuteronomy", "JOS": "Joshua", "JDG": "Judges", "RUT": "Ruth",
    "1SA": "1 Samuel", "2SA": "2 Samuel", "1KI": "1 Kings", "2KI": "2 Kings",
    "1CH": "1 Chronicles", "2CH": "2 Chronicles", "EZR": "Ezra", "NEH": "Nehemiah",
    "EST": "Esther", "JOB": "Job", "PSA": "Psalms", "PRO": "Proverbs",
    "ECC": "Ecclesiastes", "SNG": "Song of Songs", "ISA": "Isaiah", "JER": "Jeremiah",
    "LAM": "Lamentations", "EZK": "Ezekiel", "DAN": "Daniel", "HOS": "Hosea",
    "JOL": "Joel", "AMO": "Amos", "OBA": "Obadiah", "JON": "Jonah",
    "MIC": "Micah", "NAM": "Nahum", "HAB": "Habakkuk", "ZEP": "Zephaniah",
    "HAG": "Haggai", "ZEC": "Zechariah", "MAL": "Malachi",
    # New Testament
    "MAT": "Matthew", "MRK": "Mark", "LUK": "Luke", "JHN": "John",
    "ACT": "Acts", "ROM": "Romans", "1CO": "1 Corinthians", "2CO": "2 Corinthians",
    "GAL": "Galatians", "EPH": "Ephesians", "PHP": "Philippians", "COL": "Colossians",
    "1TH": "1 Thessalonians", "2TH": "2 Thessalonians", "1TI": "1 Timothy",
    "2TI": "2 Timothy", "TIT": "Titus", "PHM": "Philemon", "HEB": "Hebrews",
    "JAS": "James", "1PE": "1 Peter", "2PE": "2 Peter", "1JN": "1 John",
    "2JN": "2 John", "3JN": "3 John", "JUD": "Jude", "REV": "Revelation",
}

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
    # Format: .../Alignment/{book_folder}/{BOOKCODE_CHAPTER_Verse_VERSE}.wav
    filename_stem = df["audio_file"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df["book_code"] = filename_stem.apply(lambda x: x.split("_")[0].upper())
    df["chapter"] = filename_stem.apply(lambda x: x.split("_")[1])
    df["verse"] = filename_stem.apply(lambda x: x.split("_")[-1])

    # Resolve full book name: use code->name mapping, fall back to folder name
    # (folder name is already a full name for previously processed languages like Marathi)
    book_folder = df["audio_file"].apply(lambda x: x.split("/")[-2])
    df["book"] = df["book_code"].map(BOOK_CODE_TO_NAME).fillna(book_folder)

    # Map book name to testament (Old/New)
    df["testament"] = df["book"].map(BOOK_TO_TESTAMENT)

    # Get audio duration in seconds
    df["duration_seconds"] = df["audio_file"].apply(audio_stats.get_audio_duration)

    # Reorder columns (drop book_code, it was only needed for name resolution)
    df = df[["audio_file", "text", "testament", "book", "chapter", "verse", "duration_seconds"]]

    return df


def prepare_alignment_dataset(
    alignment_df: pd.DataFrame,
    num_std_devs: float = 3.0,
    test_size: float = 0.05,
    seed: int = 42,
):
    """
    Preprocess an alignment DataFrame and return a train/test split HF DatasetDict.

    Args:
        alignment_df: DataFrame with audio_file, text, and metadata columns
        num_std_devs: Number of standard deviations for outlier removal
        test_size: Fraction of data to use for the test split
        seed: Random seed for the train/test split

    Returns:
        DatasetDict with "train" and "test" splits
    """
    df = alignment_df.copy()

    # Simple outlier removal
    df = data_checks.remove_outliers(df, num_std_devs=num_std_devs)
    df = df[df["label"] == "BEST"]
    df = df[['audio_file', 'text', 'testament', 'book', 'chapter', 'verse', 'duration_seconds']]

    # Rename audio_file to audio for HF convention
    df = df.rename(columns={"audio_file": "audio"})

    # Create HF Dataset and cast audio column to Audio()
    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset = dataset.cast_column("audio", Audio())

    # Split into train and test
    return dataset.train_test_split(test_size=test_size, seed=seed)


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
    split_dataset = prepare_alignment_dataset(alignment_df)

    # Push to hub with retry logic for timeout errors
    for split_name in ["train", "test"]:
        split_data = split_dataset[split_name]
        for attempt in range(max_retries):
            try:
                split_data.push_to_hub(
                    repo_id=repo_id,
                    config_name=language,  # Use language as the config name
                    split=split_name,
                    private=private,
                    max_shard_size=max_shard_size,  # Smaller shards for more reliable uploads
                    commit_message=f"Upload {language} {split_name} split",
                )
                print(f"✓ Uploaded {len(split_data)} samples for '{language}' ({split_name}) to {repo_id}")
                break
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
