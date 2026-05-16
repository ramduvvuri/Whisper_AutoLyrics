"""Build train/val/test CSVs from preprocessed clips."""
from pathlib import Path
import json
import re
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

CLEAN_RE = re.compile(r"[^\w'\s]", re.UNICODE)


def normalize_text(t: str) -> str:
    t = t.lower()
    t = CLEAN_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_manifest_from_records(
    records: list[dict],
    out_dir: str | Path,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    min_dur: float = 2.0,
    max_dur: float = 30.0,
    min_words: int = 3,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)

    lang_counts = Counter(df["language"])

    print("\nLanguage distribution:")
    for lang, count in lang_counts.items():
        print(f"{lang}: {count}")

    main_language = lang_counts.most_common(1)[0][0]

    print(f"\nUsing dominant language: {main_language}")

    df = df[df["language"] == main_language]

    df["text"] = df["text"].map(normalize_text)
    df = df[df["duration"].between(min_dur, max_dur)]
    df = df[df["text"].str.split().str.len() >= min_words]
    df = df.drop_duplicates(subset=["audio_path"])
    print(f"After filter: {len(df)} clips, "
          f"total {df['duration'].sum()/3600:.2f} h")

    # split by source so songs don't leak across splits
    sources = df["source"].unique()
    train_src, test_src = train_test_split(
        sources, test_size=test_frac, random_state=seed)
    train_src, val_src = train_test_split(
        train_src, test_size=val_frac/(1-test_frac), random_state=seed)

    splits = {
        "train": df[df.source.isin(train_src)],
        "val":   df[df.source.isin(val_src)],
        "test":  df[df.source.isin(test_src)],
    }
    for name, d in splits.items():
        path = out_dir / f"{name}.csv"
        d.to_csv(path, index=False)
        print(f"  {name}: {len(d)} clips → {path}")

    # also write a stats blob for the report
    (out_dir / "stats.json").write_text(json.dumps({
        "total_clips": int(len(df)),
        "total_hours": float(df["duration"].sum() / 3600),
        "splits": {k: int(len(v)) for k, v in splits.items()},
        "sources": int(len(sources)),
    }, indent=2))
    return {k: out_dir / f"{k}.csv" for k in splits}