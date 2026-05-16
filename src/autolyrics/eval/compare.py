# File: compare.py
"""End-to-end eval: baseline vs fine-tuned, full breakdown."""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import time
import torch
import pandas as pd
import numpy as np
from autolyrics.models.inference import WhisperTranscriber
from autolyrics.eval.metrics import normalize_for_scoring
import jiwer


def evaluate_model(
    transcriber: WhisperTranscriber,
    df: pd.DataFrame,
    batch_size: int = 4,
    num_beams: int = 3,
) -> pd.DataFrame:
    """Returns df with columns: prediction, inference_sec, rtf, wer_clip, cer_clip."""
    paths = df["audio_path"].tolist()
    refs = df["text"].tolist()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    results = transcriber.transcribe_batch(paths, batch_size=batch_size,
                                            num_beams=num_beams)
    wall = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    preds = [r.text for r in results]
    refs_n  = [normalize_for_scoring(r) or "<empty>" for r in refs]
    preds_n = [normalize_for_scoring(p) or "<empty>" for p in preds]

    # per-clip WER/CER (jiwer accepts pairs)
    wer_clip = [jiwer.wer([r], [p]) for r, p in zip(refs_n, preds_n)]
    cer_clip = [jiwer.cer([r], [p]) for r, p in zip(refs_n, preds_n)]

    out = df.copy()
    out["prediction"] = preds
    out["inference_sec"] = [r.inference_sec for r in results]
    out["rtf"] = [r.rtf for r in results]
    out["wer_clip"] = wer_clip
    out["cer_clip"] = cer_clip
    out.attrs["wall_sec"] = wall
    out.attrs["peak_vram_gb"] = peak_mem_gb
    out.attrs["wer_corpus"] = float(jiwer.wer(refs_n, preds_n))
    out.attrs["cer_corpus"] = float(jiwer.cer(refs_n, preds_n))
    return out


def categorize_failures(df_with_preds: pd.DataFrame) -> pd.Series:
    """Bucket each clip into a failure category for error analysis."""
    cats = []
    for _, row in df_with_preds.iterrows():
        ref = normalize_for_scoring(row["text"])
        pred = normalize_for_scoring(row["prediction"])
        ref_words = ref.split()
        pred_words = pred.split()
        wer = row["wer_clip"]
        if pred.strip() == "":
            cats.append("empty_output")
        elif wer < 0.10:
            cats.append("near_perfect")
        elif wer < 0.30:
            cats.append("good")
        elif len(pred_words) > 1.5 * len(ref_words):
            cats.append("hallucination_long")
        elif len(pred_words) < 0.5 * len(ref_words):
            cats.append("undergeneration")
        elif len(set(pred_words)) < 0.5 * len(pred_words):
            cats.append("repetition_loop")
        elif wer > 0.80:
            cats.append("severe_failure")
        else:
            cats.append("partial")
    return pd.Series(cats, index=df_with_preds.index, name="failure_category")


def main(baseline_id: str, ft_adapter: str, manifest: str, out_dir: str,
         num_beams: int):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest)
    print(f"Test set: {len(df)} clips, {df['duration'].sum()/60:.1f} min")

    print("\n[1/2] Baseline eval…")
    base_tx = WhisperTranscriber(model_id=baseline_id)
    base_df = evaluate_model(base_tx, df, num_beams=num_beams)
    base_df["failure_category"] = categorize_failures(base_df)
    base_df.to_csv(out_dir / "baseline_predictions.csv", index=False)
    del base_tx; torch.cuda.empty_cache()

    print("\n[2/2] Fine-tuned eval…")
    ft_tx = WhisperTranscriber(model_id=baseline_id, adapter_path=ft_adapter)
    ft_df = evaluate_model(ft_tx, df, num_beams=num_beams)
    ft_df["failure_category"] = categorize_failures(ft_df)
    ft_df.to_csv(out_dir / "ft_predictions.csv", index=False)

    # Headline comparison
    summary = {
        "baseline": {
            "model": baseline_id,
            "wer": base_df.attrs["wer_corpus"],
            "cer": base_df.attrs["cer_corpus"],
            "mean_rtf": float(base_df["rtf"].mean()),
            "peak_vram_gb": base_df.attrs["peak_vram_gb"],
            "wall_sec": base_df.attrs["wall_sec"],
        },
        "fine_tuned": {
            "model": f"{baseline_id}+{ft_adapter}",
            "wer": ft_df.attrs["wer_corpus"],
            "cer": ft_df.attrs["cer_corpus"],
            "mean_rtf": float(ft_df["rtf"].mean()),
            "peak_vram_gb": ft_df.attrs["peak_vram_gb"],
            "wall_sec": ft_df.attrs["wall_sec"],
        },
        "delta": {
            "wer_abs": ft_df.attrs["wer_corpus"] - base_df.attrs["wer_corpus"],
            "wer_rel_pct": 100 * (1 - ft_df.attrs["wer_corpus"] / base_df.attrs["wer_corpus"]),
            "cer_abs": ft_df.attrs["cer_corpus"] - base_df.attrs["cer_corpus"],
        },
        "n_clips": len(df),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Failure category breakdown
    cat_table = pd.DataFrame({
        "baseline": base_df["failure_category"].value_counts(normalize=True),
        "fine_tuned": ft_df["failure_category"].value_counts(normalize=True),
    }).fillna(0).round(3)
    cat_table.to_csv(out_dir / "failure_categories.csv")
    print("\nFailure categories (fraction):")
    print(cat_table)

    # Per-source breakdown if "source" column exists
    if "source" in df.columns:
        per_source = pd.DataFrame({
            "baseline_wer": base_df.groupby("source")["wer_clip"].mean(),
            "ft_wer": ft_df.groupby("source")["wer_clip"].mean(),
        }).round(3)
        per_source["delta"] = (per_source["ft_wer"] - per_source["baseline_wer"]).round(3)
        per_source.to_csv(out_dir / "per_source_wer.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_id", default="openai/whisper-small")
    ap.add_argument("--ft_adapter",  required=True)
    ap.add_argument("--manifest",    default="data/manifests/test.csv")
    ap.add_argument("--out_dir",     default="outputs/eval")
    ap.add_argument("--num_beams",   type=int, default=3)
    main(**vars(ap.parse_args()))