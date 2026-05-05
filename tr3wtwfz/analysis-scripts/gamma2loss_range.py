"""
reads analysis csvs, plots gamma as discrete bins on the x axis and losses plots with median ~ full range as y axis
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_loss(ax, data_dict, title):
    gammas = sorted(data_dict.keys(), key=float)
    mins, maxs, medians = [], [], []

    for gamma in gammas:
        combined = np.array(data_dict[gamma])  # only this gamma's data
        mins.append(combined.min())
        maxs.append(combined.max())
        medians.append(np.median(combined))

    mins = np.array(mins)
    maxs = np.array(maxs)
    medians = np.array(medians)
    x = np.arange(len(gammas))

    print("x, mins, maxs, medians, ", x, mins, maxs, medians)

    # min/max vertical bars
    ax.vlines(x, mins, maxs, color='steelblue', linewidth=2, label='min-max range')
    # horizontal caps at min and max
    ax.hlines(mins, x - 0.1, x + 0.1, color='steelblue', linewidth=2)
    ax.hlines(maxs, x - 0.1, x + 0.1, color='steelblue', linewidth=2)
    # median dot
    ax.plot(x, medians, 'o', color='tomato', zorder=5, markersize=7, label='median')
    # connect medians
    ax.plot(x, medians, '-', color='tomato', linewidth=1.5, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(gammas)
    ax.set_xlabel('gamma')
    ax.set_ylabel('loss')
    ax.set_title(title)
    ax.legend()

def plot_combined(ax, left, right):
    gammas = sorted(set(left.keys()) | set(right.keys()), key=float)
    x = np.arange(len(gammas))
    cap = 0.1

    all_values = []

    for i, gamma in enumerate(gammas):
        for data_dict, color, label in [(right, 'steelblue', 'known'), (left, 'coral', 'unknown')]:
            if gamma not in data_dict:
                continue
            combined = np.array(data_dict[gamma]).flatten()
            median = np.median(combined)
            mn, mx = combined.min(), combined.max()
            all_values.append(combined)

            offset = -0.1 if color == 'steelblue' else 0.1  # side by side per gamma
            ax.vlines(x[i] + offset, mn, mx, color=color, linewidth=2)
            ax.hlines(mn, x[i] + offset - cap/2, x[i] + offset + cap/2, color=color, linewidth=2)
            ax.hlines(mx, x[i] + offset - cap/2, x[i] + offset + cap/2, color=color, linewidth=2)
            ax.plot(x[i] + offset, median, 'o', color=color, zorder=5, markersize=7)

    # global min/max across both dicts
    all_combined = np.concatenate(all_values)
    global_min = all_combined.min()
    global_max = all_combined.max()

    ax.axhline(global_min, color='gray', linestyle='--', linewidth=1.2, label=f'global min ({global_min:.2f})')
    ax.axhline(global_max, color='black', linestyle='--', linewidth=1.2, label=f'global max ({global_max:.2f})')

    # dummy handles for legend
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color='steelblue', linewidth=2, label='Matching Training Data (Going Right)'),
        Line2D([0], [0], color='coral',     linewidth=2, label='Diverging Training Data (Going Left)'),
    ]
    ax.legend(handles=handles)

    ax.set_xticks(x)
    ax.set_xticklabels(gammas)
    ax.set_xlabel('gamma')
    ax.set_ylabel('loss')
    ax.set_title('Loss on User Trajectories - Combined (each n = 50)')

def read_and_plot(path):

    left = dict()
    right = dict()

    numeric_dirs = [d for d in path.iterdir() if d.is_dir() and (d.name.startswith("0") or d.name.startswith("1"))] # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    #print(numeric_dirs)
    for numeric_dir in numeric_dirs:
        numeric_dir_full_path = path / numeric_dir

        left_right = [d for d in numeric_dir_full_path.iterdir() if d.name == "left" or d.name == "right"]

        for sub_dir in left_right:
            sub_dir_full_path = numeric_dir_full_path / sub_dir # tr3.../0.0/left

            print(sub_dir_full_path)

            # we want to collect 
            # 0.0/left, 0.2/left, 0.4/left, 0.6/left, 0.8/left, 1.0/left together later
            # right now it's going to show up as 0.0/left and 0.0/right together


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
            if gamma not in target:
                target[gamma] = df["loss"].to_numpy()
            
            # after first run -- 
            # 0.0/left 0.0/right
            # and then 0.0/left, 0.0/right, 0.2/left, 0.2/right
            # ...
            print("left, ", left)
            print("right, ",right)

    fig = plt.figure(figsize=(14, 10))
    ax_known   = fig.add_subplot(2, 2, 1)
    ax_unknown = fig.add_subplot(2, 2, 2)
    ax_combined = fig.add_subplot(2, 1, 2)  # spans full bottom row

    # known = user also trying to go to known space which is right goalpost post
    # unknown = user trying to go to unknown space which is left goalpost post
    plot_loss(ax_known, right, title = "Loss on User Trajectories Matching Training Data (Going Right); n = 50")
    plot_loss(ax_unknown, left, title = "Loss on User Trajectories Diverging Training Data (Going Left); n = 50")
    plot_combined(ax_combined, left, right)

    plt.tight_layout()

    save_path = root_dir / "gamma2loss_range.png"
    #plt.savefig(save_path, dpi = 150)
    plt.show()
    print(f"saved to {save_path}")


if __name__ == '__main__':
    root_dir = Path(sys.argv[1]) # python gamma2loss_range.py /code/tr3wtwfz/
    read_and_plot(root_dir)
