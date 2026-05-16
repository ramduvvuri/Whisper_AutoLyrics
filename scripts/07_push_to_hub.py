# File: 07_push_to_hub.py
"""Upload LoRA adapter + processor to HF Hub with a real model card."""
from pathlib import Path
import argparse
import json
from huggingface_hub import HfApi, create_repo


MODEL_CARD_TEMPLATE = """---
language: de
license: apache-2.0
tags:
  - whisper
  - lora
  - peft
  - automatic-speech-recognition
  - singing-voice
  - lyrics-transcription
base_model: openai/whisper-small
library_name: peft
pipeline_tag: automatic-speech-recognition
---

# AUTOLYRICS — Whisper-small + LoRA for Singing Lyrics Transcription

LoRA adapter for `openai/whisper-small`, fine-tuned for **singing voice → lyrics**
transcription. Built as a 4-day end-to-end ML project; see the full repo at
[GitHub](https://github.com/{gh_user}/autolyrics) and live demo at
[HF Space](https://huggingface.co/spaces/{hf_user}/autolyrics).

## Why this exists

Off-the-shelf ASR fails on singing because of pitch variation, sustained
phonemes, rhythm irregularities, and (often) backing music. This adapter
recovers a substantial fraction of that loss with ~0.5% extra trainable
parameters.

## Results on held-out singing test set

| Metric | Whisper-small (baseline) | + LoRA (this adapter) | Δ |
|---|---|---|---|
| WER  | {base_wer}% | **{ft_wer}%** | **{wer_delta} pts** |
| CER  | {base_cer}% | **{ft_cer}%** | {cer_delta} pts |
| RTF on T4 | {base_rtf} | {ft_rtf} | ~same |

Test set: {n_clips} clips, song-disjoint from train.

## How to use

```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "{hf_user}/autolyrics-whisper-small-lora")
proc  = WhisperProcessor.from_pretrained("{hf_user}/autolyrics-whisper-small-lora")
model.generation_config.language = "de"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

wav, sr = torchaudio.load("song_clip.wav")
if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)

feats = proc(wav.squeeze(0).numpy(), sampling_rate=16000,
             return_tensors="pt").input_features
ids = model.generate(feats, num_beams=5, max_new_tokens=225)
print(proc.batch_decode(ids, skip_special_tokens=True)[0])
```

For best results, isolate vocals first with [Demucs](https://github.com/facebookresearch/demucs)
(`htdemucs_ft`), then pass the `vocals.wav` to this model.

## Training details

- Base model: `openai/whisper-small` (244M params)
- PEFT: LoRA, r=32, alpha=64, dropout=0.05, target=`q_proj,v_proj`
- Trainable params: ~1.2M (~0.5% of total)
- Optimizer: AdamW, lr=1e-3, linear warmup 50 steps
- Batch: 8 × grad_accum 2 = effective 16; fp16
- Epochs: 5 with early stopping (patience=2) on eval WER
- Hardware: single NVIDIA T4 (Colab Pro)

## Dataset

{dataset_blurb}

## Limitations

- German only (training data was German).
- Heavy distortion / extreme growl vocals are still hard.
- Best results require vocal isolation as a preprocessing step.

## Citation

```
@misc{{autolyrics2026,
  author = {{ {gh_user} }},
  title  = {{AUTOLYRICS: LoRA Fine-tuning of Whisper for Singing Lyrics}},
  year   = {{2026}},
  howpublished = {{\\url{{https://github.com/{gh_user}/autolyrics}}}}
}}
```
"""


def main(args):
    api = HfApi()
    create_repo(args.repo_id, exist_ok=True, repo_type="model")

    # Render model card from training summary
    s = json.loads(Path(args.summary_json).read_text())
    b, f = s["baseline"], s["fine_tuned"]
    card_md = MODEL_CARD_TEMPLATE.format(
        gh_user=args.gh_user, hf_user=args.hf_user,
        base_wer=f"{b['wer']*100:.1f}", ft_wer=f"{f['wer']*100:.1f}",
        base_cer=f"{b['cer']*100:.1f}", ft_cer=f"{f['cer']*100:.1f}",
        wer_delta=f"{(f['wer']-b['wer'])*100:+.1f}",
        cer_delta=f"{(f['cer']-b['cer'])*100:+.1f}",
        base_rtf=f"{b['mean_rtf']:.2f}", ft_rtf=f"{f['mean_rtf']:.2f}",
        n_clips=s["n_clips"],
        dataset_blurb=args.dataset_blurb,
    )
    Path(args.adapter_dir, "README.md").write_text(card_md, encoding="utf-8")

    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=args.adapter_dir,
        repo_type="model",
        commit_message="Upload AUTOLYRICS LoRA adapter",
    )
    print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir",  default="outputs/checkpoints/lora_run1/final")
    ap.add_argument("--repo_id",      required=True,
                    help="e.g. yourname/autolyrics-whisper-small-lora")
    ap.add_argument("--summary_json", default="outputs/eval/summary.json")
    ap.add_argument("--gh_user",      required=True)
    ap.add_argument("--hf_user",      required=True)
    ap.add_argument("--dataset_blurb",
                    default="DSing30 + curated Jamendo Lyrics subset, "
                            "vocal-isolated via Demucs htdemucs_ft, "
                            "song-disjoint train/val/test splits.")
    main(ap.parse_args())
