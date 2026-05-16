# Author: ramduvvuri
# File: collator.py
"""Data collator for Whisper seq2seq with padding + label -100 trick."""
from dataclasses import dataclass
from typing import Any
import torch
from transformers import WhisperProcessor

#collator.py
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads audio features to the same length within a batch, pads labels with
    -100 (ignored by CE loss), strips the BOS decoder-start token from labels
    if present.
    """
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Audio path
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Labels path
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip BOS (decoder_start_token_id) if collated tokenizer added it.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch