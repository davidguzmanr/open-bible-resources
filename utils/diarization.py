"""
Pyannote-based speaker diarization utilities.

Typical usage
-------------
from pyannote.audio import Pipeline
from utils.hf_preprocessing import get_alignment_dataframe, prepare_alignment_dataset
from utils.diarization import add_speaker_ids

alignment_df = get_alignment_dataframe(LANGUAGE, base_dir="data/audios")
ds = prepare_alignment_dataset(alignment_df)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-precision-2",
    token=<token>,
)
pipeline.to(torch.device("cuda"))

ds = add_speaker_ids(ds, pipeline, language=LANGUAGE)
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm
from datasets import Dataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_first_samples(ds: Dataset) -> dict:
    """Return the first dataset example for each book (preserves insertion order)."""
    first_samples = {}
    for sample in tqdm(ds, desc="Finding first sample per book"):
        book = sample["book"]
        if book not in first_samples:
            first_samples[book] = sample
    return first_samples


def _build_timing_dataframe(
    samples: list,
    silence_duration: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Concatenate audio arrays (with silence gaps) and build a DataFrame that
    records the [start, stop] window of each book inside the concatenated signal.

    Returns
    -------
    concatenated_audio : np.ndarray
    df : pd.DataFrame  columns: book, testament, chapter, verse, text,
                                duration, start, stop  (times as floats in seconds)
    """
    sampling_rate = samples[0]["audio"]["sampling_rate"]
    for sample in samples:
        assert sample["audio"]["sampling_rate"] == sampling_rate, (
            f"Sampling rate mismatch: expected {sampling_rate}, "
            f"got {sample['audio']['sampling_rate']} for {sample['book']}"
        )

    silence = np.zeros(int(silence_duration * sampling_rate), dtype=np.float32)

    audio_pieces = []
    for i, sample in enumerate(tqdm(samples, desc="Concatenating audio")):
        audio_pieces.append(sample["audio"]["array"])
        if i < len(samples) - 1:
            audio_pieces.append(silence)

    concatenated_audio = np.concatenate(audio_pieces)

    rows = []
    cumulative = 0.0
    for i, sample in enumerate(samples):
        duration = sample["duration_seconds"]
        start = cumulative
        gap = silence_duration if i < len(samples) - 1 else 0.0
        stop = cumulative + duration + gap
        rows.append({
            "book": sample["book"],
            "testament": sample["testament"],
            "chapter": sample["chapter"],
            "verse": sample["verse"],
            "text": sample["text"],
            "duration": duration,
            "start": start,
            "stop": stop,
        })
        cumulative = stop

    return concatenated_audio, sampling_rate, pd.DataFrame(rows)


def _assign_speaker_ids(df: pd.DataFrame, diarization_output) -> list:
    """
    For each book's [start, stop] window, find the speaker with the most
    accumulated speaking time in the diarization output.
    """
    segments = [
        (turn.start, turn.end, speaker)
        for turn, speaker in diarization_output.speaker_diarization
    ]

    speaker_ids = []
    for _, row in df.iterrows():
        book_start, book_stop = row["start"], row["stop"]
        speaker_time: dict[str, float] = defaultdict(float)

        for seg_start, seg_end, speaker in segments:
            overlap = max(0.0, min(seg_end, book_stop) - max(seg_start, book_start))
            if overlap > 0:
                speaker_time[speaker] += overlap

        dominant = max(speaker_time, key=speaker_time.get) if speaker_time else None
        speaker_ids.append(dominant)

    return speaker_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_book_to_speaker_id(
    ds: Dataset,
    pipeline,
    language: str,
    silence_duration: float = 1.0,
    output_dir: str = "diarization",
) -> dict:
    """
    Run pyannote speaker diarization on a concatenated clip of the first verse
    from each book and return a ``{book: speaker_id}`` mapping.

    Parameters
    ----------
    ds : datasets.Dataset
        Dataset produced by ``prepare_alignment_dataset``.  Must have columns:
        ``audio``, ``book``, ``testament``, ``chapter``, ``verse``,
        ``text``, ``duration_seconds``.
    pipeline : pyannote.audio.Pipeline
        Loaded pyannote speaker-diarization pipeline (already sent to device).
    language : str
        Language name used when naming output files.
    silence_duration : float
        Seconds of silence inserted between book clips (default 1.0).
    output_dir : str
        Directory where the concatenated WAV and timing CSV are saved.

    Returns
    -------
    dict  mapping book name → speaker_id string (e.g. ``"SPEAKER_03"``).
    """
    os.makedirs(output_dir, exist_ok=True)

    first_samples = _get_first_samples(ds)
    samples = list(first_samples.values())
    print(f"\nFound {len(samples)} books")

    concatenated_audio, sampling_rate, df = _build_timing_dataframe(
        samples, silence_duration
    )

    wav_path = os.path.join(
        output_dir, f"concatenated_bible_books_{language}_with_silence.wav"
    )
    print(f"Saving concatenated audio to {wav_path} ...")
    wavfile.write(wav_path, sampling_rate, concatenated_audio.astype(np.float32))
    print(f"Total duration: {len(concatenated_audio) / sampling_rate:.2f}s")

    csv_path = os.path.join(output_dir, f"samples_{language}_with_silence.csv")
    df.to_csv(csv_path, index=False)

    print("Running diarization pipeline ...")
    diarization_output = pipeline(wav_path)

    df["speaker_id"] = _assign_speaker_ids(df, diarization_output)
    print("\nBook → speaker mapping:")
    print(df[["book", "testament", "speaker_id"]].to_string(index=False))

    return dict(zip(df["book"], df["speaker_id"]))


def add_speaker_ids(
    ds: Dataset,
    pipeline,
    language: str,
    silence_duration: float = 1.0,
    output_dir: str = "diarization",
) -> Dataset:
    """
    Add a ``speaker_id`` column to *ds* using pyannote speaker diarization.

    Internally calls :func:`build_book_to_speaker_id` to obtain a
    ``{book: speaker_id}`` mapping, then maps it over every example.

    Parameters
    ----------
    ds : datasets.Dataset
    pipeline : pyannote.audio.Pipeline
    language : str
    silence_duration : float
    output_dir : str

    Returns
    -------
    datasets.Dataset  with an additional ``speaker_id`` column.
    """
    book_to_speaker_id = build_book_to_speaker_id(
        ds=ds,
        pipeline=pipeline,
        language=language,
        silence_duration=silence_duration,
        output_dir=output_dir,
    )
    ds = ds.map(lambda example: {"speaker_id": book_to_speaker_id[example["book"]]})
    return ds
