"""Generate timestamp-based word alignments from audio using Whisper.

Returns phrase-level segments (start_sec, end_sec, text) suitable for
chunk_with_alignments() in preprocess.py.
"""

from __future__ import annotations

import torch
from transformers import pipeline

# Languages supported by whisper-small for forced decoding.
# Everything else falls back to None (auto-detect).
_SUPPORTED_LANGS = {
    "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr", "cs",
    "da", "nl", "en", "et", "fi", "fr", "gl", "de", "el", "he", "hi",
    "hu", "is", "id", "it", "ja", "kn", "kk", "ko", "lv", "lt", "mk",
    "ms", "mr", "mi", "ne", "no", "fa", "pl", "pt", "ro", "ru", "sr",
    "sk", "sl", "es", "sw", "sv", "tl", "ta", "th", "tr", "uk", "ur",
    "vi", "cy",
}

# Quality thresholds for alignment segments
MIN_SEG_SEC   = 2.0   # seconds  — shorter segments are too unstable
MAX_WPS       = 5.0   # words/s  — denser than this is a timestamp error
MIN_WPS       = 0.3   # words/s  — sparser than this is likely silence/instr.
MIN_WORDS     = 3     # words    — single-word segments are useless
FLUSH_SEC     = 6.0   # target segment length when accumulating words


def _safe_language(language: str | None) -> str | None:
    """Return the language code if Whisper supports forced decoding for it."""
    if language is None:
        return None
    lang = language.lower().strip()
    return lang if lang in _SUPPORTED_LANGS else None


def generate_word_alignments(
    audio_path: str,
    language: str = "de",
    model_id: str = "openai/whisper-small",
    device: str | None = None,
) -> list[tuple[float, float, str]]:
    """
    Run Whisper with word-level timestamps on *audio_path* and return
    merged phrase segments.

    Returns
    -------
    list of (start_sec, end_sec, text)  — sorted, non-overlapping.
    Empty list on any failure.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    safe_lang = _safe_language(language)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        return_timestamps="word",
        generate_kwargs={
            "task": "transcribe",
            "language": safe_lang,
        },
    )

    try:
        result = asr(str(audio_path), return_timestamps="word")
    except Exception as exc:
        print(f"  [WARN] Whisper failed on {audio_path}: {exc}")
        return []

    raw_words: list[dict] = result.get("chunks", [])
    if not raw_words:
        print(f"  [WARN] No word chunks returned for {audio_path}")
        return []

    print(f"  [ALIGN] {len(raw_words)} word segments")

    # --- merge words into phrase segments of ~FLUSH_SEC seconds ---
    segments: list[tuple[float, float, str]] = []
    buf_words: list[str] = []
    buf_start: float | None = None
    buf_end:   float | None = None

    def _flush():
        nonlocal buf_start, buf_end, buf_words
        if not buf_words or buf_start is None or buf_end is None:
            buf_words = []; buf_start = buf_end = None
            return
        text = " ".join(buf_words).strip()
        dur  = buf_end - buf_start
        wps  = len(buf_words) / max(dur, 1e-6)

        if dur < MIN_SEG_SEC:
            pass  # silently drop — too short
        elif len(buf_words) < MIN_WORDS:
            print(f"  [WARN] sparse alignment ({len(buf_words)} words / {dur:.1f}s) — skipped")
        elif wps > MAX_WPS:
            print(f"  [WARN] dense alignment ({wps:.1f} WPS) — skipped")
        elif wps < MIN_WPS:
            print(f"  [WARN] sparse alignment ({wps:.2f} WPS) — likely silence/instr — skipped")
        else:
            segments.append((buf_start, buf_end, text))

        buf_words = []; buf_start = buf_end = None

    for chunk in raw_words:
        word = chunk.get("text", "").strip()
        ts   = chunk.get("timestamp")     # (start, end) tuple or None

        # Skip words with missing timestamps
        if not word or ts is None:
            continue
        t_start, t_end = ts
        if t_start is None or t_end is None:
            continue

        # Start a new buffer or accumulate
        if buf_start is None:
            buf_start = t_start
        buf_end = t_end
        buf_words.append(word)

        # Flush when we reach the target segment length
        if (buf_end - buf_start) >= FLUSH_SEC:
            _flush()

    _flush()  # handle remaining words

    return segments
