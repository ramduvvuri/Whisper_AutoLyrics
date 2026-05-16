"""Vocal isolation via demucs htdemucs_ft."""
from pathlib import Path
import subprocess
from tqdm import tqdm
import sys

def isolate_vocals(
    input_dir: str | Path,
    output_dir: str | Path,
    model: str = "htdemucs",
    device: str = "cuda",
    pattern: str = "*.wav",
) -> None:
    """
    Run demucs on every audio file in input_dir; keep only the vocals stem
    in output_dir/<stem>.wav.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.rglob(pattern)) + sorted(input_dir.rglob("*.mp3"))
    print(f"Isolating {len(files)} files with {model}…")

    tmp = output_dir / "_demucs_tmp"
    for f in tqdm(files):
        out_path = output_dir / f"{f.stem}.wav"
        if out_path.exists():
            continue
        subprocess.run([
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-n", model,
            "-d", device,
            "-o", str(tmp),
            str(f),
        ], check=True)
        # demucs writes to tmp/<model>/<stem>/vocals.wav
        produced = next(tmp.rglob(f"{f.stem}/vocals.wav"), None)
        if produced is not None:
            produced.rename(out_path)

    # cleanup
    import shutil
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    isolate_vocals(args.inp, args.out, device=args.device)