# Author: ramduvvuri
# File: plots.py
"""Two plots that go in the report and the website."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_wer_distribution(base_csv: str, ft_csv: str, out_path: str):
    base = pd.read_csv(base_csv)["wer_clip"]
    ft = pd.read_csv(ft_csv)["wer_clip"]

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    bins = np.linspace(0, 2.0, 41)
    ax.hist(base, bins=bins, alpha=0.55, label="Baseline", color="#888")
    ax.hist(ft,   bins=bins, alpha=0.75, label="Fine-tuned (LoRA)", color="#111")
    ax.set_xlabel("Per-clip WER")
    ax.set_ylabel("Number of clips")
    ax.set_title("WER distribution: baseline vs fine-tuned")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_failure_categories(base_csv: str, ft_csv: str, out_path: str):
    base = pd.read_csv(base_csv)["failure_category"].value_counts(normalize=True)
    ft   = pd.read_csv(ft_csv)["failure_category"].value_counts(normalize=True)
    cats = ["near_perfect", "good", "partial", "undergeneration",
            "hallucination_long", "repetition_loop", "severe_failure",
            "empty_output"]
    base = base.reindex(cats, fill_value=0)
    ft   = ft.reindex(cats, fill_value=0)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=140)
    x = np.arange(len(cats)); w = 0.4
    ax.bar(x - w/2, base.values, w, label="Baseline", color="#888")
    ax.bar(x + w/2, ft.values,   w, label="Fine-tuned", color="#111")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel("Fraction of clips")
    ax.set_title("Failure category shift after fine-tuning")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plot_wer_distribution(
        "outputs/eval/baseline_predictions.csv",
        "outputs/eval/ft_predictions.csv",
        "reports/figures/wer_distribution.png",
    )
    plot_failure_categories(
        "outputs/eval/baseline_predictions.csv",
        "outputs/eval/ft_predictions.csv",
        "reports/figures/failure_categories.png",
    )