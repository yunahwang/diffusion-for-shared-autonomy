#!/usr/bin/env python3
"""
viz_losses.py — Plot the 'loss' column from one or two CSV files.

Usage:
    python viz_losses.py <file1.csv>
    python viz_losses.py <file1.csv> <file2.csv>
"""

import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_loss(path: str) -> pd.Series:
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"Error: file not found — {path}")
    df = pd.read_csv(p)
    if "loss" not in df.columns:
        sys.exit(
            f"Error: no 'loss' column in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    
    print(type(df["loss"]))
    return df["loss"].reset_index(drop=True)


def style_ax(ax, loss: pd.Series, title: str, color: str) -> None:
    """Apply a clean, consistent style to a single axes."""
    ax.plot(loss, color=color, linewidth=1.8, alpha=0.9)

    # Subtle fill under the curve
    ax.fill_between(range(len(loss)), loss, alpha=0.12, color=color)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Step", fontsize=10, labelpad=6)
    ax.set_ylabel("Loss", fontsize=10, labelpad=6)

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", labelsize=9)
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.grid(which="minor", linestyle=":", linewidth=0.3, alpha=0.3)
    ax.set_xlim(left=0)

    # Annotate final loss value
    final = loss.dropna().iloc[-1]
    ax.annotate(
        f"final: {final:.4f}",
        xy=(len(loss) - 1, final),
        xytext=(-60, 14),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
        arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
    )


def main() -> None:
    args = sys.argv[1:]

    if len(args) == 0 or len(args) > 2:
        print(__doc__)
        sys.exit(1)

    COLORS = ["#2563eb", "#dc2626"]   # blue, red

    plt.rcParams.update(
        {
            "figure.facecolor": "#f9fafb",
            "axes.facecolor": "#ffffff",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )

    if len(args) == 1:
        loss = load_loss(args[0])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("#f9fafb")
        label = pathlib.Path(args[0]).stem
        style_ax(ax, loss, f"Loss — {label}", COLORS[0])
        fig.suptitle("Training Loss", fontsize=15, fontweight="bold", y=1.01)

    else:  # two files — side by side
        loss1 = load_loss(args[0])
        loss2 = load_loss(args[1])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), sharey=False)
        fig.patch.set_facecolor("#f9fafb")

        label1 = pathlib.Path(args[0]).stem
        label2 = pathlib.Path(args[1]).stem
        style_ax(ax1, loss1, label1, COLORS[0])
        style_ax(ax2, loss2, label2, COLORS[1])

        fig.suptitle("Training Loss Comparison", fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()