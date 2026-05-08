"""
reads analysis csvs, plots gamma as discrete bins on the x axis and the number of occurrences on either going to the right goal post or to the left goal post on the y axis as bar plots
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_count(ax, data_dict, title):
    gammas = sorted(data_dict.keys(), key=float)

    print("gammas, ", gammas)
    x = np.arange(len(gammas))
    width = 0.35

    left_counts = [data_dict[g].get('left_goal(not trained)', 0) for g in gammas]
    right_counts = [data_dict[g].get('right_goal(trained)', 0) for g in gammas]

    b1 = ax.bar(x - width/2, left_counts, width, label='Left Goalpost (Untrained Goal)', color='coral')
    b2 = ax.bar(x + width/2, right_counts, width, label='Right Goalpost (Trained Goal)', color='steelblue')

    ax.bar_label(b1, fontsize=10, fontweight='bold', color='black')
    ax.bar_label(b2, fontsize=10, fontweight='bold', color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(gammas)
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Visit Count')
    ax.set_title(title)
    ax.legend()


def plot_combined(ax, left, right):
    gammas = sorted(set(left.keys()) | set(right.keys()), key=float)
    x = np.arange(len(gammas))
    width = 0.2

    left_untrained = [left.get(g, {}).get('left_goal(not trained)', 0) for g in gammas]
    left_trained = [left.get(g, {}).get('right_goal(trained)', 0) for g in gammas]
    right_untrained = [right.get(g, {}).get('left_goal(not trained)', 0) for g in gammas]
    right_trained = [right.get(g, {}).get('right_goal(trained)', 0) for g in gammas]

    b1 = ax.bar(x - 1.5*width, left_untrained, width, label='User Going Left + Arriving at Left Goalpost (Untrained Goal)', color='coral')
    b2 = ax.bar(x - 0.5*width, left_trained, width, label='User Going Left + Arriving at Right Goalpost (Trained Goal)', color='tomato')
    b3 = ax.bar(x + 0.5*width, right_untrained, width, label='User Going Right + Arriving at Left Goalpost (Untrained Goal)', color='steelblue')
    b4 = ax.bar(x + 1.5*width, right_trained, width, label='User Going Right + Arriving at Right Goalpost (Trained Goal)', color='royalblue')

    ax.bar_label(b1, fontsize=10, fontweight='bold', color='black')
    ax.bar_label(b2, fontsize=10, fontweight='bold', color='black')
    ax.bar_label(b3, fontsize=10, fontweight='bold', color='black')
    ax.bar_label(b4, fontsize=10, fontweight='bold', color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(gammas)
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Visit Count')
    ax.set_title('Combined Goal Visits by Gamma (each n = 50)')
    ax.legend()
    
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

            left_count = 0
            right_count = 0

            if gamma not in target:
                which_goal = df["which_goal"].to_numpy()
                
                for goal in which_goal:
                    if goal == 0.0:
                        left_count += 1
                    elif goal == 1.0:
                        right_count += 1
                    
                    target[gamma] = {"left_goal(not trained)": left_count, "right_goal(trained)": right_count}

        
        print("left_count, right_count, ", left_count, right_count)
        print("left, ", left)
        print("right, ", right)


    # START PLOTTING
    fig = plt.figure(figsize=(16, 8))
    ax_known   = fig.add_subplot(2, 2, 1)
    ax_unknown = fig.add_subplot(2, 2, 2)
    ax_combined = fig.add_subplot(2, 1, 2)  # spans full bottom row

    plot_count(ax_known, right, title = "Number of Goal Visits For User Trajectories Matching Training Data (Going Right); n = 50")
    plot_count(ax_unknown, left, title = "Number of Goal Visits For User Trajectories Diverging Training Data (Going Left); n = 50")
    plot_combined(ax_combined, left, right)

    plt.tight_layout()

    save_path = root_dir / "gamma2which_side.png"
    plt.savefig(save_path, dpi = 150)
    plt.show()
    print(f"saved to {save_path}")

if __name__ == "__main__":
    root_dir = Path(sys.argv[1]) # root@67762e0bc982:/code# python tr3wtwfz/analysis-scripts/gamma2which_side.py /code/tr3wtwfz/
    read_and_plot(root_dir)