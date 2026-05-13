# from pathlib import Path
# import sys
# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_path(path):
#     # for entry in path.iterdir():
#     #     print(entry.name)
#     csvs = [entry.name for entry in path.iterdir() if entry.name.endswith(".csv")] # this is 2023 orig data

#     plt.figure()

#     for i, csv in enumerate(csvs):
#         print("i, ",i)
#         full_csv_path = path / csv
#         df = pd.read_csv(full_csv_path, header=  None) # NOTE: selectively put header = None
#         ee_x = df.iloc[:, 3]
#         ee_y = df.iloc[:, 4]
#         # print("ee_x, ", ee_x)
#         # print("ee_y, ", ee_y)
#         start_x = df.iloc[0, 3]; start_y = df.iloc[0, 4]
#         end_x = df.iloc[len(df)-1, 3]; end_y = df.iloc[len(df)-1, 4]

#         label_traj = "ee_pos" if i == 0 else None
#         label_start = "ee_start_pos" if i == 0 else None
#         label_end = "ee_end_pos" if i == 0 else None
#         plt.plot(ee_x, ee_y, color = "blue", lw = 0.01, label=label_traj)
#         plt.scatter(start_x, start_y, c = "green", s = 30, label=label_start)
#         plt.scatter(end_x, end_y, c = "red", s = 30, label=label_end)

#         # if i == 50:
#         #     break

#     print("plot stuck here?")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("Trajectory Comparison")
#     plt.legend()
#     plt.axis("equal")
#     plt.grid(True)

#     print("2")

#     plt.show()
#     #plt.savefig("2023_ee_traj.png", dpi=300)
#     # plt.show()
#     print("4")
#     print("Saved plot to 2023_ee_traj.png")


# if __name__ == '__main__':
#     # input - folder path where csvs are located
#     path = Path(sys.argv[1])
#     opt = Path(sys.argv[3]) # how to set None as default
#     path2 = Path(sys.argv[2])
#     if opt:
#         opt_plot_path(path, path2)
#     plot_path(path)

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


def plot_path(path, zoom_endpoints_only = False):
    csvs = [entry.name for entry in path.iterdir() if entry.name.endswith(".csv")]
    plt.figure()
    for i, csv in enumerate(csvs):
        print("i, ", i)
        full_csv_path = path / csv
        df = pd.read_csv(full_csv_path, header=None)
        ee_x = df.iloc[:, 3]
        ee_y = df.iloc[:, 4]
        start_x = df.iloc[0, 3];  start_y = df.iloc[0, 4]
        end_x = df.iloc[-1, 3];   end_y   = df.iloc[-1, 4]
        label_traj  = "ee_pos"       if i == 0 else None
        label_start = "ee_start_pos" if i == 0 else None
        label_end   = "ee_end_pos"   if i == 0 else None
        if not zoom_endpoints_only:
            plt.plot(ee_x, ee_y, color="blue", lw=0.01, label=label_traj)
        plt.scatter(start_x, start_y, c="green", s=30, label=label_start)
        plt.scatter(end_x,   end_y,   c="red",   s=120, marker = "*", edgecolors = "black", linewidths=0.5, label=label_end)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
    print("Saved plot to 2023_ee_traj.png")


# def plot_two_paths(path1, path2):
#     """
#     Plots both folders on the same figure.
#     Folder 1: blue trajectory, green start (circle), red end (X)
#     Folder 2: orange trajectory, lime start (circle), darkred end (X)
#     Markers use different shapes so start/end are distinguishable even in grayscale.
#     """
#     folders = [
#         (path1, "blue",   "green", "red",     "target1"),
#         (path2, "orange", "lime",  "darkred",  "target2"),
#     ]

#     plt.figure(figsize=(16, 8))

#     for folder_path, traj_color, start_color, end_color, folder_name in folders:
#         csvs = [entry.name for entry in folder_path.iterdir() if entry.name.endswith(".csv")]

#         for i, csv in enumerate(csvs):
#             full_csv_path = folder_path / csv
#             df = pd.read_csv(full_csv_path, header=None)

#             ee_x    = df.iloc[:, 3]
#             ee_y    = df.iloc[:, 4]
#             start_x = df.iloc[0, 3];  start_y = df.iloc[0, 4]
#             end_x   = df.iloc[-1, 3]; end_y   = df.iloc[-1, 4]

#             # Only label once per folder to keep the legend clean
#             label_traj  = f"{folder_name} traj"  if i == 0 else None
#             label_start = f"{folder_name} start"  if i == 0 else None
#             label_end   = f"{folder_name} end"    if i == 0 else None

#             plt.plot(ee_x, ee_y, color=traj_color, lw=0.1, label=label_traj)
#             plt.scatter(start_x, start_y, c=start_color, s=40, marker="o",
#                         edgecolors="black", linewidths=0.5, label=label_start)
#             plt.scatter(end_x,   end_y,   c=end_color,   s=40, marker="X",
#                         edgecolors="black", linewidths=0.5, label=label_end)

#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("Trajectory Comparison — Two Folders")
#     plt.legend()
#     plt.axis("equal")
#     plt.grid(True)
#     plt.show()

# def plot_two_paths(path1, path2):
#     folders = [
#         (path1, "blue",   "green", "red",     "target"),
#         (path2, "orange", "lime",  "black",  "target2"),
#     ]

#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

#     for ax, title_suffix, xlim, ylim in [
#         (ax1, "",           None,         None),
#         (ax2, " (zoomed)",  (-0.5, 0.5),  (-1, 1)),
#     ]:
#         for folder_path, traj_color, start_color, end_color, folder_name in folders:
#             csvs = [entry.name for entry in folder_path.iterdir() if entry.name.endswith(".csv")]

#             for i, csv in enumerate(csvs):
#                 full_csv_path = folder_path / csv
#                 df = pd.read_csv(full_csv_path, header=None)

#                 ee_x    = df.iloc[:, 3]
#                 ee_y    = df.iloc[:, 4]
#                 start_x = df.iloc[0, 3];  start_y = df.iloc[0, 4]
#                 end_x   = df.iloc[-1, 3]; end_y   = df.iloc[-1, 4]

#                 label_traj  = f"{folder_name} traj"  if i == 0 else None
#                 label_start = f"{folder_name} start"  if i == 0 else None
#                 label_end   = f"{folder_name} end"    if i == 0 else None

#                 ax.plot(ee_x, ee_y, color=traj_color, lw=0.01, alpha=0.2, label=label_traj)
#                 ax.scatter(start_x, start_y, c=start_color, s=40, marker="o",
#                            edgecolors="black", linewidths=0.5, label=label_start)
#                 ax.scatter(end_x,   end_y,   c=end_color,   s=120, marker="*",
#                            edgecolors="black", linewidths=0.5, label=label_end)

#         if xlim: ax.set_xlim(xlim)
#         if ylim: ax.set_ylim(ylim)
#         ax.set_xlabel("X")
#         ax.set_ylabel("Y")
#         ax.set_title(f"Trajectory Comparison — Two Folders{title_suffix}")
#         ax.legend()
#         ax.grid(True)

#     plt.tight_layout()
#     plt.show()

def plot_two_paths(path1, path2, zoom_endpoints_only=False):
    folders = [
        (path1, "blue",   "green", "red",      "o", "*", "target"),
        (path2, "orange", "lime",  "black", "o", "*", "target2"),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

    for ax, title_suffix, xlim, ylim, is_zoom in [
        (ax1, "",          None,         None,   False),
        (ax2, " (zoomed)", (-0.5, 0.5),  (-1, 1), True),
    ]:
        for folder_path, traj_color, start_color, end_color, start_marker, end_marker, folder_name in folders:
            csvs = [entry.name for entry in folder_path.iterdir() if entry.name.endswith(".csv")]

            for i, csv in enumerate(csvs):
                full_csv_path = folder_path / csv
                df = pd.read_csv(full_csv_path, header=None)

                ee_x    = df.iloc[:, 3]
                ee_y    = df.iloc[:, 4]
                start_x = df.iloc[0, 3];  start_y = df.iloc[0, 4]
                end_x   = df.iloc[-1, 3]; end_y   = df.iloc[-1, 4]

                label_traj  = f"{folder_name} traj"  if i == 0 else None
                label_start = f"{folder_name} start"  if i == 0 else None
                label_end   = f"{folder_name} end"    if i == 0 else None

                # Skip trajectory lines in zoomed plot if zoom_endpoints_only is set
                if not (is_zoom and zoom_endpoints_only):
                    ax.plot(ee_x, ee_y, color=traj_color, lw=0.01, alpha=0.2, label=label_traj)

                ax.scatter(start_x, start_y, c=start_color, s=40,  marker=start_marker,
                           edgecolors="black", linewidths=0.5, label=label_start)
                ax.scatter(end_x,   end_y,   c=end_color,   s=120, marker=end_marker,
                           edgecolors="black", linewidths=0.5, label=label_end)

        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Trajectory Comparison — Two Folders{title_suffix}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if path2 is not None:
        plot_two_paths(path1, path2, zoom_endpoints_only= True)
    else:
        plot_path(path1, zoom_endpoints_only = True)