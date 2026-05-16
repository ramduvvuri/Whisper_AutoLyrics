# Author: ramduvvuri
# File: metrics.py
"""WER / CER and domain-specific metrics."""
"""WER/CER with consistent text normalization."""
import re
import jiwer

# Canonical normalization for scoring:
# lowercase, strip punctuation except apostrophes, collapse whitespace.
_PUNC = re.compile(r"[^a-z0-9äöüß'\s]")
_WS = re.compile(r"\s+")

def normalize_for_scoring(t: str) -> str:
    t = t.lower()
    t = _PUNC.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t

def compute_wer_cer(refs, preds, normalize=True):
    if normalize:
        refs = [normalize_for_scoring(r) for r in refs]
        preds = [normalize_for_scoring(p) for p in preds]
    # jiwer wants non-empty strings; guard
    refs  = [r if r else "<empty>" for r in refs]
    preds = [p if p else "<empty>" for p in preds]
    return {
        "wer": float(jiwer.wer(refs, preds)),
        "cer": float(jiwer.cer(refs, preds)),
    }