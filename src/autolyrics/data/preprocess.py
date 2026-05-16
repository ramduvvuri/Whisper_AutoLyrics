# File: preprocess.py
"""Resample, mono-fy, normalize, chunk."""
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm

TARGET_SR = 16000
MAX_CHUNK_SEC = 30  # whisper hard limit
MIN_CHUNK_SEC = 2   # too-short chunks are unstable


def load_and_normalize(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)        # mono
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    # peak normalize
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.95
    return wav, TARGET_SR


def chunk_with_alignments(
    wav: torch.Tensor,
    sr: int,
    alignments: list[tuple[float, float, str]],
    out_dir: Path,
    stem: str,
    language: str,
) -> list[dict]:
    """
    alignments: list of (start_sec, end_sec, text_for_that_segment).
    Greedy-merge consecutive segments until ~25s, write a wav and a record.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    buf_start = None
    buf_end = None
    buf_text = []

    def flush(idx):
        nonlocal buf_start, buf_end, buf_text
        if buf_start is None:
            return
        dur = buf_end - buf_start
        if dur < MIN_CHUNK_SEC:
            buf_start = buf_end = None; buf_text = []; return
        s, e = int(buf_start * sr), int(buf_end * sr)
        clip = wav[:, s:e]
        out_path = out_dir / f"{stem}_{idx:04d}.wav"
        torchaudio.save(str(out_path), clip, sr)
        records.append({
            "audio_path": str(out_path),
            "text": " ".join(buf_text).strip(),
            "duration": dur,
            "source": stem,
            "language": language,
        })
        buf_start = buf_end = None; buf_text = []

    idx = 0
    for (s, e, t) in alignments:
        if buf_start is None:
            buf_start, buf_end, buf_text = s, e, [t]
            continue
        if e - buf_start <= MAX_CHUNK_SEC:
            buf_end = e
            buf_text.append(t)
        else:
            flush(idx); idx += 1
            buf_start, buf_end, buf_text = s, e, [t]
    flush(idx)
    return records


def chunk_uniform(
    wav: torch.Tensor,
    sr: int,
    full_text: str,
    out_dir: Path,
    stem: str,
    language: str,
    chunk_sec: int = 25,
) -> list[dict]:
    """When you have full_text but no alignments: split text proportionally.
    Lossy but workable as a fallback."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_sec = wav.shape[-1] / sr
    n_chunks = max(1, int(total_sec // chunk_sec))
    words = full_text.split()
    words_per_chunk = max(1, len(words) // n_chunks)

    records = []
    for i in range(n_chunks):
        s = int(i * chunk_sec * sr)
        e = int(min((i + 1) * chunk_sec * sr, wav.shape[-1]))
        clip = wav[:, s:e]
        text = " ".join(words[i * words_per_chunk : (i + 1) * words_per_chunk])
        out_path = out_dir / f"{stem}_{i:04d}.wav"
        torchaudio.save(str(out_path), clip, sr)
        records.append({
            "audio_path": str(out_path),
            "text": text,
            "duration": (e - s) / sr,
            "source": stem,
            "language": language,
        })
    return records