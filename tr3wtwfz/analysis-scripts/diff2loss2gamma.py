
"""
reads analysis csvs, plots diff as x axis and loss on the y axis. plot user-left and user-right with different colored-lines in a single plot. optionally have a second plot on the right horizontal side that interactively maps diff-losses to gammas
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys

# def plot_lines(ax, side):
#     gammas = sorted(set(left.keys()) | set(right.keys()), key=float)

#     left_losses = []; left_diffs = []; right_losses = []; right_diffs = []

#     for g in gammas:
#         if g in left:
#             left_losses.extend(left[g]["loss"])
#             left_diffs.extend(left[g]["diff"])
#         elif g in right:
#             right_losses.extend(right[g]["loss"])
#             right_diffs.extend(right[g]["diff"])
#     print(len(left_losses), len(left_diffs), len(right_losses), len(right_diffs))

#     ax.plot(left_diffs, left_losses, "o-", color = "coral", markersize = 2, label = "(Expert - raw user actions) on User Trajectories Diverging Training Data (Going Left)")
#     ax.plot(right_diffs, right_losses, "o-", color = "steelblue", markersize = 2, label = "(Expert - raw user actions) on User Trajectories Matching Training Data (Going Right)")

#     ax.set_xlabel("(Expert - raw user actions)")
#     ax.set_ylabel("Diffusion Loss")
#     ax.set_title("Relationship Between Diffusion Loss and Expert - User Action Gap")
#     ax.legend()

def plot_lines(ax, side, color, label):
    gammas = sorted(side.keys(), key=float)
 
    losses = []; diffs = []
    for g in gammas:
        losses.extend(side[g]["loss"])
        diffs.extend(side[g]["diff"])
    print(f"{label}: n_diffs={len(diffs)}, n_losses={len(losses)}")
 
    #ax.plot(diffs, losses, "o-", color=color, markersize=2, label=label)
    ax.scatter(diffs, losses, color=color, s=4, alpha=0.5, label=label)
    ax.set_xlabel("(Expert - raw user actions)")
    ax.set_ylabel("Diffusion Loss")
    ax.set_title("Relationship Between Diffusion Loss and Expert - User Action Gap")
    ax.legend()
 
def plot_right_by_gamma(ax, right):
    plot_by_gamma(ax, right, None, "Diffusion Loss vs Expert-User Gap per Gamma (Going Right)")

def plot_left_by_gamma(ax, left):
    plot_by_gamma(ax, left, None, "Diffusion Loss vs Expert-User Gap per Gamma (Going Left)")


def plot_by_gamma(ax, data, color_scatter, title):
    """Generic per-gamma scatter plot (works for left or right)."""
    gammas = sorted(data.keys(), key=float)
    half_width = 0.4

    for i, g in enumerate(gammas):
        diffs = np.array(data[g]["diff"])
        losses = np.array(data[g]["loss"])

        if diffs.max() != diffs.min():
            diffs_scaled = (diffs - diffs.min()) / (diffs.max() - diffs.min())
        else:
            diffs_scaled = np.zeros_like(diffs)

        x_positions = i + (diffs_scaled - 0.5) * half_width * 2
        ax.scatter(x_positions, losses, s=5, alpha=0.5, color=color_scatter, label=f'gamma={g}')

        y_bottom = ax.get_ylim()[0] if ax.get_ylim()[0] != 0.0 else 0
        ax.annotate('', xy=(i + half_width, y_bottom), xytext=(i - half_width, y_bottom),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=1))
        ax.annotate(f'{diffs.min():.2f}', xy=(i - half_width, y_bottom),
                    ha='center', va='top', fontsize=6, color='gray')
        ax.annotate(f'{diffs.max():.2f}', xy=(i + half_width, y_bottom),
                    ha='center', va='top', fontsize=6, color='gray')
        mid_diff = (diffs.min() + diffs.max()) / 2
        ax.annotate(f'{mid_diff:.2f}', xy=(i, y_bottom),
                    ha='center', va='top', fontsize=6, color='gray')

    ax.set_xticks(np.arange(len(gammas)))
    ax.set_xticklabels(gammas)
    ax.set_xlabel("Gamma (spread within = Expert - Raw User Actions)")
    ax.set_ylabel("Diffusion Loss")
    ax.set_title(title)
    ax.legend(markerscale=3)


def plot_diff_vs_loss_with_correlation(left, right):
    """Standalone plot: top row = scatter+correlation for left & right;
       bottom row = 1x3 per-gamma subplots (left only, right only, combined label)."""

    fig = plt.figure(figsize=(18, 14))

    # --- Top row: combined scatter with regression lines (spans all 3 cols) ---
    ax_top = fig.add_subplot(2, 1, 1)

    all_left_diffs, all_left_losses = [], []
    all_right_diffs, all_right_losses = [], []

    for g in left:
        all_left_diffs.extend(left[g]["diff"])
        all_left_losses.extend(left[g]["loss"])
    for g in right:
        all_right_diffs.extend(right[g]["diff"])
        all_right_losses.extend(right[g]["loss"])

    ax_top.scatter(all_left_diffs, all_left_losses, color="coral", s=10, alpha=0.4,
                   label="Going Left (Diverging Training)")
    ax_top.scatter(all_right_diffs, all_right_losses, color="steelblue", s=10, alpha=0.4,
                   label="Going Right (Matching Training)")

    def add_regression(ax, diffs, losses, color, label_prefix):
        if len(diffs) < 2:
            return
        diffs_arr = np.array(diffs)
        losses_arr = np.array(losses)
        slope, intercept, r, p, se = stats.linregress(diffs_arr, losses_arr)
        x_line = np.linspace(diffs_arr.min(), diffs_arr.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linewidth=2,
                label=f"{label_prefix} | r={r:.3f}, p={p:.3e}, slope={slope:.4f}")
        print(f"\n--- Correlation Analysis: {label_prefix} ---")
        print(f"  Pearson r     : {r:.4f}")
        print(f"  p-value       : {p:.4e}")
        print(f"  Slope         : {slope:.4f}")
        print(f"  Intercept     : {intercept:.4f}")
        print(f"  Std Error     : {se:.4f}")
        print(f"  n             : {len(diffs_arr)}")

    add_regression(ax_top, all_left_diffs, all_left_losses, "tomato", "Left (Diverging)")
    add_regression(ax_top, all_right_diffs, all_right_losses, "royalblue", "Right (Matching)")

    ax_top.set_xlabel("Expert - Raw User Actions (diff)")
    ax_top.set_ylabel("Diffusion Loss")
    ax_top.set_title("Diff vs Diffusion Loss — Scatter + Correlation")
    ax_top.legend(fontsize=9)

    # --- Bottom row: 1x3 per-gamma plots ---
    ax_left  = fig.add_subplot(2, 3, 4)
    ax_right = fig.add_subplot(2, 3, 5)
    ax_both  = fig.add_subplot(2, 3, 6)

    plot_by_gamma(ax_left,  left,  "coral",     "Diffusion Loss per Gamma — Going Left (Diverging)")
    plot_by_gamma(ax_right, right, "steelblue", "Diffusion Loss per Gamma — Going Right (Matching)")

    # Combined: overlay left + right on same axes
    plot_by_gamma(ax_both, left,  "coral",     "Combined (Left + Right) per Gamma")
    plot_by_gamma(ax_both, right, "steelblue", "Combined (Left + Right) per Gamma")
    ax_both.set_title("Combined per Gamma")

    plt.tight_layout()
    return fig


def read_and_plot(path):

    left = dict()
    right = dict()

    numeric_dirs = [d for d in path.iterdir() if d.is_dir() and (d.name.startswith("0") or d.name.startswith("1"))]

    for numeric_dir in numeric_dirs:
        numeric_dir_full_path = path / numeric_dir

        left_right = [d for d in numeric_dir_full_path.iterdir() if d.name == "left" or d.name == "right"]

        # FIX: process both left and right inside the loop
        for sub_dir in left_right:
            sub_dir_full_path = numeric_dir_full_path / sub_dir

            if "left" in sub_dir_full_path.name:
                target = left
            elif "right" in sub_dir_full_path.name:
                target = right
            else:
                continue

            csv = [csv for csv in sub_dir_full_path.iterdir() if csv.name.endswith(".csv")][0]
            csv_full_path = sub_dir_full_path / csv
            print(csv_full_path)
            df = pd.read_csv(csv_full_path)

            gamma = sub_dir_full_path.parts[3]
            print(gamma)

            if gamma not in target:
                if gamma == "0.0" and target == left:
                    break
                loss = df["loss"].to_numpy()
                diff = df["diff"].to_numpy()
                target[gamma] = {"diff": diff, "loss": loss}

            print("left, ", left)
            print("right, ", right)

    # Original subplots
    fig, (ax_top, ax2, ax_middle, ax_bottom) = plt.subplots(4, 1, figsize=(12, 12))
    plot_lines(ax_top, right, "steelblue", "Going Right")
    plot_lines(ax2, left, "coral", "Going Left")
    plot_right_by_gamma(ax_middle, right)
    plot_left_by_gamma(ax_bottom, left)
    plt.tight_layout()

    save_path = root_dir / "diff2loss2gamma.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"saved to {save_path}")

    # New correlation plot
    corr_fig = plot_diff_vs_loss_with_correlation(left, right)
    corr_save_path = root_dir / "diff_vs_loss_correlation.png"
    corr_fig.savefig(corr_save_path, dpi=150)
    corr_fig.show()
    print(f"saved correlation plot to {corr_save_path}")


if __name__ == "__main__":
    root_dir = Path(sys.argv[1])
    read_and_plot(root_dir)