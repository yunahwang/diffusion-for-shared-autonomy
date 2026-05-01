import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_two_csvs(file1, file2):
    # Read WITHOUT headers
    df1 = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)

    # ---- Extract from CSV 1 ----
    block1_x = df1.iloc[:, 0]
    block1_y = df1.iloc[:, 1]
    ee1_x = df1.iloc[:, 3]
    ee1_y = df1.iloc[:, 4]

    # ---- Extract from CSV 2 ----
    block2_x = df2.iloc[:, 0]
    block2_y = df2.iloc[:, 1]
    ee2_x = df2.iloc[:, 3]
    ee2_y = df2.iloc[:, 4]

    # ---- Plot ----
    plt.figure()

    # CSV 1 (blue)
    #plt.plot(block1_x, block1_y, color='lightblue', lw = 0.5, label='Block 1')
    plt.plot(ee1_x, ee1_y, color='blue', lw = 2, label='tgt')

    # CSV 2 (red)
    #plt.plot(block2_x, block2_y, color='lightcoral', lw = 0.5, label='Block 2')
    plt.plot(ee2_x, ee2_y, color='darkred', lw = 2, label='tgtf')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    plt.savefig("trajectory_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved plot to trajectory_comparison.png")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot.py file1.csv file2.csv")
    else:
        plot_two_csvs(sys.argv[1], sys.argv[2])