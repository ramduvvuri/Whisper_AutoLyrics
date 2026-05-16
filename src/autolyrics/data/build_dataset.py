# File: build_dataset.py
"""End-to-end: read raw dataset, produce processed chunks + manifests."""

from pathlib import Path
from langdetect import detect
from collections import Counter

from autolyrics.data.preprocess import (
    load_and_normalize,
    chunk_uniform,            # kept for reference / fallback — not called by default
    chunk_with_alignments,
)
from autolyrics.data.alignment import generate_word_alignments
from autolyrics.data.manifest import build_manifest_from_records

BASE       = Path("src/autolyrics/data/data")
AUDIO_DIR  = Path("src/autolyrics/data/data/vocals")
LYRICS_DIR = BASE / "raw/jamendo/annotations/lyrics"
PROCESSED  = BASE / "processed"
MANIFESTS  = BASE / "manifests"


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"


def main():

    records = []

    audio_files = (
        list(AUDIO_DIR.rglob("*.wav"))
        + list(AUDIO_DIR.rglob("*.mp3"))
    )

    print(f"Found {len(audio_files)} audio files")

    for audio in audio_files:

        txt = LYRICS_DIR / f"{audio.stem}.txt"

        if not txt.exists():
            print(f"Missing lyrics for: {audio.stem}")
            continue

        try:
            text = txt.read_text(encoding="utf-8").strip()

            if not text:
                print(f"Empty lyrics file: {txt.name}")
                continue

            language = detect_language(text)
            print(f"[ALIGN] {audio.stem} -> {language}")

            wav, sr = load_and_normalize(audio)

            # --- timestamp-based alignment (replaces chunk_uniform) ---
            alignments = generate_word_alignments(
                str(audio),
                language=language,
            )

            if not alignments:
                print(f"  [WARN] No alignments for {audio.stem} — skipping")
                continue

            print(f"  [ALIGN] {len(alignments)} alignment segments")

            chunks = chunk_with_alignments(
                wav,
                sr,
                alignments,
                PROCESSED / audio.parent.name,
                stem=audio.stem,
                language=language,
            )

            records.extend(chunks)
            print(f"  [CHUNKS] {len(chunks)} final chunks")

        except Exception as e:
            print(f"Failed processing {audio.name}: {e}")

    print(f"\nTotal chunks: {len(records)}")

    if len(records) == 0:
        print("No valid audio-text pairs found.")
        return

    MANIFESTS.mkdir(parents=True, exist_ok=True)

    build_manifest_from_records(records, MANIFESTS)

    print(f"Manifest files saved to: {MANIFESTS}")


if __name__ == "__main__":
    main()