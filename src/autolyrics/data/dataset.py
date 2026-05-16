# Author: ramduvvuri
# File: dataset.py
"""HF Dataset wrapper that maps audio_path/text → input_features/labels."""
from datasets import Dataset, Audio
import pandas as pd


def load_split_as_hf(csv_path: str, audio_col: str = "audio_path",
                    text_col: str = "text") -> Dataset:
    df = pd.read_csv(csv_path)
    df = df[[audio_col, text_col]].rename(
        columns={audio_col: "audio", text_col: "sentence"})
    ds = Dataset.from_pandas(df, preserve_index=False)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds


def make_prepare_fn(processor):
    """Returns a function .map() can apply to one example at a time."""
    def prepare(example):
        audio = example["audio"]
        # log-mel features (input_features), shape (80, 3000) for 30s @ 16k
        feat = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        # tokenize text → input_ids (these become labels)
        ids = processor.tokenizer(example["sentence"]).input_ids
        return {"input_features": feat, "labels": ids}
    return prepare