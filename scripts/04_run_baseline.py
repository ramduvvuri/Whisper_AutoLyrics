# Author: ramduvvuri
# File: 04_run_baseline.py
"""Run baseline Whisper-small on the test split, dump predictions, compute WER/CER."""
from pathlib import Path
import argparse
import json
import pandas as pd
from tqdm import tqdm
from autolyrics.models.inference import WhisperTranscriber
from autolyrics.eval.metrics import compute_wer_cer, normalize_for_scoring


def main(model_id: str, manifest: str, out_dir: str, batch_size: int, num_beams: int,
         adapter_path: str = None):
    df = pd.read_csv(manifest)
    paths = df["audio_path"].tolist()
    refs = df["text"].tolist()

    if adapter_path:
        print(f"[LORA] using adapter: {adapter_path}")
    tx = WhisperTranscriber(model_id=model_id, adapter_path=adapter_path)
    print(f"Transcribing {len(paths)} clips with {model_id} (beams={num_beams})…")
    results = tx.transcribe_batch(paths, batch_size=batch_size, num_beams=num_beams)

    preds = [r.text for r in results]

    # raw + normalized scores
    raw = compute_wer_cer(refs, preds, normalize=False)
    norm = compute_wer_cer(refs, preds, normalize=True)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df_out = df.assign(
        prediction=preds,
        inference_sec=[r.inference_sec for r in results],
        rtf=[r.rtf for r in results],
    )
    df_out.to_csv(out_dir / "predictions.csv", index=False)

    summary = {
        "model_id": model_id,
        "n_clips": len(paths),
        "total_audio_sec": float(df["duration"].sum()),
        "wer_raw": raw["wer"], "cer_raw": raw["cer"],
        "wer_norm": norm["wer"], "cer_norm": norm["cer"],
        "mean_rtf": float(pd.Series([r.rtf for r in results]).mean()),
        "num_beams": num_beams,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="openai/whisper-small")
    ap.add_argument("--manifest", default="data/manifests/test.csv")
    ap.add_argument("--out_dir",  default="outputs/baseline_whisper_small")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_beams", type=int, default=3)
    ap.add_argument("--adapter_path", default=None)
    args = ap.parse_args()
    main(**vars(args))
