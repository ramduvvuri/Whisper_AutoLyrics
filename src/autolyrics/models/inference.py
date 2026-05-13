from __future__ import annotations

"""Whisper inference (baseline AND fine-tuned)."""
#infernce.py
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import time
import torch
import torchaudio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)


@dataclass
class TranscriptionResult:
    text: str
    audio_path: str
    duration_sec: float
    inference_sec: float
    rtf: float                    # real-time factor: inference / duration
    model_id: str
    num_beams: int


class WhisperTranscriber:
    """Thin wrapper around Whisper for batched and single-clip inference."""

    def __init__(
        self,
        model_id: str = "openai/whisper-small",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        adapter_path: Optional[str] = None,    # set for fine-tuned PEFT
    ):
        self.device = device
        self.model_id = model_id
        self.processor = WhisperProcessor.from_pretrained(model_id)

        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype
        )

        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model_id = f"{model_id}+{adapter_path}"
            print(f"[LORA] loaded adapter: {adapter_path}")

        # Move to device AFTER adapter load, then set eval + inference config
        self.model.to(device).eval()
        self.model.config.forced_decoder_ids = None
        self.model.config.use_cache = True

        # Lock generation behavior
        gc = self.model.generation_config
        gc.language = "de"
        gc.task = "transcribe"
        gc.forced_decoder_ids = None
        gc.suppress_tokens = []
        gc.no_repeat_ngram_size = 3
        self.processor.tokenizer.forced_decoder_ids = None  # silence tokenizer-side warning

    @torch.inference_mode()
    def transcribe_file(
        self,
        path: str | Path,
        num_beams: int = 5,
        max_new_tokens: int = 225,
    ) -> TranscriptionResult:
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        duration = wav.shape[-1] / 16000

        # inputs = self.processor(
        #     wav.squeeze(0).numpy(),
        #     sampling_rate=16000,
        #     return_tensors="pt",
        # )
        # feats = inputs.input_features.to(self.device, dtype=self.model.dtype)
        inputs = self.processor(
            wav.squeeze(0).numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        input_features = inputs.input_features

        if input_features.shape[-1] < 3000:
            pad = 3000 - input_features.shape[-1]

            input_features = torch.nn.functional.pad(
                input_features,
                (0, pad),
            )

        elif input_features.shape[-1] > 3000:
            input_features = input_features[..., :3000]

        feats = input_features.to(
            self.device,
            dtype=self.model.dtype,
        )

        t0 = time.perf_counter()
        ids = self.model.generate(
            feats,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            condition_on_prev_tokens=False,
        )
        t1 = time.perf_counter()

        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        inf = t1 - t0
        return TranscriptionResult(
            text=text.strip(),
            audio_path=str(path),
            duration_sec=duration,
            inference_sec=inf,
            rtf=inf / max(duration, 1e-6),
            model_id=self.model_id,
            num_beams=num_beams,
        )

    @torch.inference_mode()
    def transcribe_batch(
        self,
        paths: list[str | Path],
        batch_size: int = 8,
        num_beams: int = 5,
        max_new_tokens: int = 225,
    ) -> list[TranscriptionResult]:
        """Batched for throughput. Pads variable-length to the longest in batch."""
        results: list[TranscriptionResult] = []

        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            wavs = []
            durations = []
            for p in chunk:
                wav, sr = torchaudio.load(str(p))
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                wavs.append(wav.squeeze(0).numpy())
                durations.append(wav.shape[-1] / 16000)

            # inputs = self.processor(
            #     wavs,
            #     sampling_rate=16000,
            #     return_tensors="pt",
            #     padding=True,
            # )
            # feats = inputs.input_features.to(self.device, dtype=self.model.dtype)
            inputs = self.processor(
                wavs,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )

            input_features = inputs.input_features

            if input_features.shape[-1] < 3000:
                pad = 3000 - input_features.shape[-1]

                input_features = torch.nn.functional.pad(
                    input_features,
                    (0, pad),
                )

            elif input_features.shape[-1] > 3000:
                input_features = input_features[..., :3000]

            feats = input_features.to(
                self.device,
                dtype=self.model.dtype,
            )
            t0 = time.perf_counter()
            ids = self.model.generate(
                feats, num_beams=num_beams, max_new_tokens=max_new_tokens,
                condition_on_prev_tokens=False,
            )
            t1 = time.perf_counter()
            texts = self.processor.batch_decode(ids, skip_special_tokens=True)

            inf_per = (t1 - t0) / len(chunk)
            for p, txt, d in zip(chunk, texts, durations):
                results.append(TranscriptionResult(
                    text=txt.strip(),
                    audio_path=str(p),
                    duration_sec=d,
                    inference_sec=inf_per,
                    rtf=inf_per / max(d, 1e-6),
                    model_id=self.model_id,
                    num_beams=num_beams,
                ))
        return results