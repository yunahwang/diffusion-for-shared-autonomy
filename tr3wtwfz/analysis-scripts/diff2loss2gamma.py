"""
reads analysis csvs, plots diff as x axis and loss on the y axis. plot user-left and user-right with different colored-lines in a single plot. optionally have a second plot on the right horizontal side that interactively maps diff-losses to gammas
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_lines(ax, left, right):
    # gammas_left = sorted(left.keys(), key=float)
    # gammas_right = sorted(right.keys(), key=float)
    gammas = sorted(set(left.keys()) | set(right.keys()), key=float)

    # left_losses = left["loss"]
    # left_diffs = left["diff"]
    # right_losses = right["loss"]
    # right_diffs = right["diff"]

    left_losses = []; left_diffs = []; right_losses = []; right_diffs = []

    for g in gammas:
        if g in left:
            left_losses.extend(left[g]["loss"])
            left_diffs.extend(left[g]["diff"])
        elif g in right:
            right_losses.extend(right[g]["loss"])
            right_diffs.extend(right[g]["diff"])
    print(len(left_losses), len(left_diffs), len(right_losses), len(right_diffs))

    # left_losses = [left[g]["loss"] for g in gammas if g in left]
    # left_diffs = [left[g]["diff"] for g in gammas if g in left]
    # right_losses = [right[g]["loss"] for g in gammas if g in right]
    # right_diffs = [right[g]["diff"] for g in gammas if g in right]

    ax.plot(left_diffs, left_losses, "o-", color = "coral", markersize = 2, label = "(Expert - raw user actions) on User Trajectories Diverging Training Data (Going Left)")
    ax.plot(right_diffs, right_losses, "o-", color = "steelblue", markersize = 2, label = "(Expert - raw user actions) on User Trajectories Matching Training Data (Going Right)")

    ax.set_xlabel("(Expert - raw user actions)")
    ax.set_ylabel("Diffusion Loss")

    ax.set_title("Relationship Between Diffusion Loss and Expert - User Action Gap")
    ax.legend()

# def plot_right_by_gamma(ax, right):
#     gammas = sorted(right.keys(), key=float)
#     x = np.arange(len(gammas))

#     for i, g in enumerate(gammas):
#         diffs = right[g]["diff"]
#         losses = right[g]["loss"]
#         ax.scatter([i] * len(losses), losses, s=5, alpha=0.5, label=f'gamma={g}')

#     ax.set_xticks(x)
#     ax.set_xticklabels(gammas)
#     ax.set_xlabel("Gamma")
#     ax.set_ylabel("Diffusion Loss")
#     ax.set_title("Diffusion Loss per Gamma (User Trajectories Matching Training Data / Going Right)")
#     ax.legend(markerscale=3)

# def plot_right_by_gamma(ax, right):
#     gammas = sorted(right.keys(), key=float)

#     for i, g in enumerate(gammas):
#         diffs = np.array(right[g]["diff"])
#         losses = np.array(right[g]["loss"])

#         # normalize diffs to a small range around the gamma's x position
#         if diffs.max() != diffs.min():
#             diffs_scaled = (diffs - diffs.min()) / (diffs.max() - diffs.min())  # 0 to 1
#         else:
#             diffs_scaled = np.zeros_like(diffs)
        
#         x_positions = i + (diffs_scaled - 0.5) * 0.8  # centered around i, within ±0.4

#         ax.scatter(x_positions, losses, s=5, alpha=0.5, label=f'gamma={g}')

#     ax.set_xticks(np.arange(len(gammas)))
#     ax.set_xticklabels(gammas)
#     ax.set_xlabel("Gamma (spread within each = Expert - Raw User Actions)")
#     ax.set_ylabel("Diffusion Loss")
#     ax.set_title("Diffusion Loss vs Expert-User Gap per Gamma (Going Right)")
#     ax.legend(markerscale=3)

def plot_right_by_gamma(ax, right):
    gammas = sorted(right.keys(), key=float)
    half_width = 0.4

    for i, g in enumerate(gammas):
        diffs = np.array(right[g]["diff"])
        losses = np.array(right[g]["loss"])

        if diffs.max() != diffs.min():
            diffs_scaled = (diffs - diffs.min()) / (diffs.max() - diffs.min())
        else:
            diffs_scaled = np.zeros_like(diffs)

        x_positions = i + (diffs_scaled - 0.5) * half_width * 2

        ax.scatter(x_positions, losses, s=5, alpha=0.5, label=f'gamma={g}')

        # draw sub x-axis line at the bottom of the plot
        y_bottom = ax.get_ylim()[0] if ax.get_ylim()[0] != 0.0 else 0
        ax.annotate('', xy=(i + half_width, y_bottom), xytext=(i - half_width, y_bottom),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=1))

        # draw min and max diff ticks + labels
        ax.annotate(f'{diffs.min():.2f}', xy=(i - half_width, y_bottom),
                    ha='center', va='top', fontsize=6, color='gray')
        ax.annotate(f'{diffs.max():.2f}', xy=(i + half_width, y_bottom),
                    ha='center', va='top', fontsize=6, color='gray')

        # draw mid tick
        mid_diff = (diffs.min() + diffs.max()) / 2
        ax.annotate(f'{mid_diff:.2f}', xy=(i, y_bottom),
                    ha='center', va='top', fontsize=6, color='gray')

    ax.set_xticks(np.arange(len(gammas)))
    ax.set_xticklabels(gammas)
    ax.set_xlabel("Gamma (spread within = Expert - Raw User Actions)")
    ax.set_ylabel("Diffusion Loss")
    ax.set_title("Diffusion Loss vs Expert-User Gap per Gamma (Going Right)")
    ax.legend(markerscale=3)

def read_and_plot(path):

    left = dict()
    right = dict()


    numeric_dirs = [d for d in path.iterdir() if d.is_dir() and (d.name.startswith("0") or d.name.startswith("1"))]

    for numeric_dir in numeric_dirs:
        numeric_dir_full_path = path / numeric_dir

        left_right = [d for d in numeric_dir_full_path.iterdir() if d.name == "left" or d.name == "right"]

        for sub_dir in left_right:
            sub_dir_full_path = numeric_dir_full_path / sub_dir

        if "left" in sub_dir_full_path.name:
            target = left
        elif "right" in sub_dir_full_path.name:
            target = right

        csv = [csv for csv in sub_dir_full_path.iterdir() if csv.name.endswith(".csv")][0]
        csv_full_path = sub_dir_full_path / csv
        print(csv_full_path)
        df = pd.read_csv(csv_full_path)

        gamma = sub_dir_full_path.parts[3]
        print(gamma)

        # left -> {0.6: {"diff": [], "loss": []}, 1.0: {"diff": [], "loss": []}}

        if gamma not in target:
            loss = df["loss"].to_numpy()
            diff = df["diff"].to_numpy()

            target[gamma] = {"diff": diff, "loss": loss}
        
        print("left, ", left)
        print("right, ", right)

    # START PLOTTING
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 12))
    plot_lines(ax_top, left, right)
    plot_right_by_gamma(ax_bottom, right)
    plt.tight_layout()

    save_path = root_dir / "diff2loss2gamma.png"
    #plt.savefig(save_path, dpi = 150)
    plt.show()
    print(f"saved to {save_path}")

if __name__ == "__main__":
    root_dir = Path(sys.argv[1]) # python diff2loss2gamma /code/tr3wtwfz/
    read_and_plot(root_dir)