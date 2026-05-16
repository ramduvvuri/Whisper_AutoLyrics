# File: 00_verify_env.py
"""Run this first. If any check fails, fix before proceeding."""
import sys
import importlib

REQUIRED = {
    "torch": "2.4",
    "torchaudio": "2.4",
    "transformers": "4.46",
    "peft": "0.13",
    "datasets": "3.1",
    "accelerate": "1.1",
    "evaluate": "0.4",
    "jiwer": "3.0",
    "soundfile": "0.12",
    "librosa": "0.10",
    "gradio": "5.",
    "huggingface_hub": "0.26",
}

def check():
    failed = []
    for pkg, want in REQUIRED.items():
        try:
            m = importlib.import_module(pkg)
            v = getattr(m, "__version__", "?")
            ok = v.startswith(want)
            print(f"[{'OK ' if ok else 'BAD'}] {pkg:>20s}  {v}  (want {want}.x)")
            if not ok:
                failed.append(pkg)
        except ImportError:
            print(f"[MISS] {pkg:>20s}  not installed")
            failed.append(pkg)

    import torch
    if not torch.cuda.is_available():
        print("[BAD] CUDA unavailable")
        failed.append("cuda")
    else:
        print(f"[OK ] CUDA  {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")

    if failed:
        print(f"\nFAILED: {failed}")
        sys.exit(1)
    print("\nAll checks passed.")

if __name__ == "__main__":
    check()