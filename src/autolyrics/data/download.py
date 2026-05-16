# File: download.py
"""Dataset download utilities."""
from pathlib import Path
import subprocess
import shutil
from datasets import load_dataset


def download_dsing(out_dir: str | Path = "data/raw/dsing") -> Path:
    """
    DSing via HuggingFace mirror if available.
    Falls back to manual instruction if not.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset("ProgramComputer/dsing", cache_dir=str(out_dir))
        print(f"Loaded DSing: {ds}")
        return out_dir
    except Exception as e:
        print(f"HF mirror failed: {e}")
        print("Manual fallback:")
        print("  1) Visit https://github.com/groadabike/Kaldi-Dsing-task")
        print("  2) Follow the 'preparing the dataset' steps")
        print(f"  3) Place the resulting wavs + transcripts in {out_dir}")
        raise SystemExit(1)


def download_jamendo_lyrics(out_dir: str | Path = "data/raw/jamendo") -> Path:
    """Clone the Jamendo Lyrics annotation repo + download referenced audio."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = out_dir / "annotations"
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/f90/jamendolyrics.git", str(repo)],
            check=True,
        )

    # Audio download per-song instructions are in the repo's README;
    # they require the Jamendo API or direct CC links.
    print(f"Annotations cloned to {repo}.")
    print("Now run scripts/01b_jamendo_audio.py to fetch audio per the repo README.")
    return out_dir


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["dsing", "jamendo"], required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.source == "dsing":
        download_dsing(args.out or "data/raw/dsing")
    else:
        download_jamendo_lyrics(args.out or "data/raw/jamendo")