from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_path(path):
    # for entry in path.iterdir():
    #     print(entry.name)
    csvs = [entry.name for entry in path.iterdir() if entry.name.endswith(".csv")] # this is 2023 orig data

    plt.figure()

    for i, csv in enumerate(csvs):
        print("i, ",i)
        full_csv_path = path / csv
        df = pd.read_csv(full_csv_path, header=  None) # NOTE: selectively put header = None
        ee_x = df.iloc[:, 3]
        ee_y = df.iloc[:, 4]
        # print("ee_x, ", ee_x)
        # print("ee_y, ", ee_y)
        start_x = df.iloc[0, 3]; start_y = df.iloc[0, 4]
        end_x = df.iloc[len(df)-1, 3]; end_y = df.iloc[len(df)-1, 4]

        label_traj = "ee_pos" if i == 0 else None
        label_start = "ee_start_pos" if i == 0 else None
        label_end = "ee_end_pos" if i == 0 else None
        plt.plot(ee_x, ee_y, color = "blue", lw = 0.01, label=label_traj)
        plt.scatter(start_x, start_y, c = "green", s = 300, label=label_start)
        plt.scatter(end_x, end_y, c = "red", s = 300, label=label_end)

        # if i == 50:
        #     break

    print("plot stuck here?")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    print("2")

    plt.show()
    #plt.savefig("2023_ee_traj.png", dpi=300)
    # plt.show()
    print("4")
    print("Saved plot to 2023_ee_traj.png")


if __name__ == '__main__':
    # input - folder path where csvs are located
    path = Path(sys.argv[1])
    plot_path(path)