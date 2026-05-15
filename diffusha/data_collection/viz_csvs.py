from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# def plot_path(path, zoom_endpoints_only = False, no_plot_ee_end = False):
#     csvs = [entry.name for entry in path.iterdir() if entry.name.endswith(".csv")]
#     plt.figure()
#     for i, csv in enumerate(csvs):
#         print("i, ", i)
#         full_csv_path = path / csv
#         df = pd.read_csv(full_csv_path, header=None)
#         ee_x = df.iloc[:, 3]
#         ee_y = df.iloc[:, 4]
#         start_x = df.iloc[0, 3];  start_y = df.iloc[0, 4]
#         end_x = df.iloc[-1, 3];   end_y   = df.iloc[-1, 4]
#         label_traj  = "ee_pos"       if i == 0 else None
#         label_start = "ee_start_pos" if i == 0 else None
#         label_end   = "ee_end_pos"   if i == 0 else None
#         if not zoom_endpoints_only:
#             plt.plot(ee_x, ee_y, color="blue", lw=0.01, label=label_traj)
#         plt.scatter(start_x, start_y, c="green", s=30, label=label_start)
        
#         if not no_plot_ee_end:
#             plt.scatter(end_x,   end_y,   c="red",   s=120, marker = "*", edgecolors = "black", linewidths=0.5, label=label_end)

#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("Trajectory Comparison")
#     plt.legend()
#     plt.axis("equal")
#     plt.grid(True)
#     plt.show()
#     print("Saved plot to 2023_ee_traj.png")

def plot_path(path, zoom_endpoints_only=False, no_plot_ee_end=False):
    csvs = sorted([entry.name for entry in path.iterdir() if entry.name.endswith(".csv")])

    coord_offset = np.array([0.4, 0.35])

    # world_targets = {
    #     "green/right target": np.array([0.3, 0.35]),
    #     "red/left target": np.array([0.5, 0.35]),
    # }

    plt.figure()

    for i, csv in enumerate(csvs):
        print("i, ", i)

        full_csv_path = path / csv
        df = pd.read_csv(full_csv_path, header=None)

        ee_centered = df.iloc[:, [3, 4]].to_numpy(dtype=float)

        # Same mapping as PyBullet replay:
        # world_x = -centered_x + 0.4
        # world_y =  centered_y + 0.35
        ee_world = np.column_stack([
            -ee_centered[:, 0],
             ee_centered[:, 1],
        ]) + coord_offset

        ee_x = ee_world[:, 0]
        ee_y = ee_world[:, 1]

        start_x, start_y = ee_world[0]
        end_x, end_y = ee_world[-1]

        label_traj = "ee_pos" if i == 0 else None
        label_start = "ee_start_pos" if i == 0 else None
        label_end = "ee_end_pos" if i == 0 else None

        if not zoom_endpoints_only:
            plt.plot(ee_x, ee_y, color="blue", lw=0.01, alpha=0.25, label=label_traj)

        plt.scatter(start_x, start_y, c="green", s=30, label=label_start)

        if not no_plot_ee_end:
            plt.scatter(
                end_x,
                end_y,
                c="peachpuff",
                s=120,
                marker="v",
                edgecolors="black",
                linewidths=0.5,
                label=label_end,
            )

    # Plot fixed world target locations
    # plt.scatter(
    #     world_targets["green/right target"][0],
    #     world_targets["green/right target"][1],
    #     c="green",
    #     s=150,
    #     marker="s",
    #     edgecolors="black",
    #     linewidths=0.5,
    #     label="green/right target",
    # )

    # plt.scatter(
    #     world_targets["red/left target"][0],
    #     world_targets["red/left target"][1],
    #     c="red",
    #     s=150,
    #     marker="s",
    #     edgecolors="black",
    #     linewidths=0.5,
    #     label="red/left target",
    # )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    print("Saved plot to 2023_ee_traj.png")



def plot_two_paths(path1, path2, zoom_endpoints_only=False, no_plot_ee_end = False):
    # TODO - I think I can draw two boxes, with how I did for workspace centering logic
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
                if not no_plot_ee_end:
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