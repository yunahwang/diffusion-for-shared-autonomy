from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.set_loglevel("critical")
import matplotlib.pyplot as plt


# ee version - one plot per csv (ignored multiple episodes inside csv)
# def plot_path(path, zoom_endpoints_only=False, no_plot_ee_end=False):
#     csvs = sorted([entry.name for entry in path.iterdir() if entry.name.endswith(".csv")])

#     coord_offset = np.array([0.4, 0.35])

#     plt.figure()

#     for i, csv in enumerate(csvs):
#         print("i, ", i)

#         full_csv_path = path / csv
#         df = pd.read_csv(full_csv_path, header=None)

#         ee_centered = df.iloc[:, [3, 4]].to_numpy(dtype=float)

#         # Same mapping as PyBullet replay:
#         # world_x = -centered_x + 0.4
#         # world_y =  centered_y + 0.35
#         ee_world = np.column_stack([
#             -ee_centered[:, 0],
#              ee_centered[:, 1],
#         ]) + coord_offset

#         ee_x = ee_world[:, 0]
#         ee_y = ee_world[:, 1]

#         start_x, start_y = ee_world[0]
#         end_x, end_y = ee_world[-1]

#         label_traj = "ee_pos" if i == 0 else None
#         label_start = "ee_start_pos" if i == 0 else None
#         label_end = "ee_end_pos" if i == 0 else None

#         if not zoom_endpoints_only:
#             plt.plot(ee_x, ee_y, color="blue", lw=0.01, alpha=0.25, label=label_traj)

#         plt.scatter(start_x, start_y, c="green", s=30, label=label_start)

#         if not no_plot_ee_end:
#             plt.scatter(
#                 end_x,
#                 end_y,
#                 c="peachpuff",
#                 s=120,
#                 marker="v",
#                 edgecolors="black",
#                 linewidths=0.5,
#                 label=label_end,
#             )


#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("Trajectory Comparison")
#     plt.legend()
#     plt.axis("equal")
#     plt.grid(True)
#     plt.show()

#     print("Saved plot to 2023_ee_traj.png")

def plot_path(path, zoom_endpoints_only=False, no_plot_ee_end=False, use_saved_png=True, return_option=False):
    csvs = sorted([entry.name for entry in path.iterdir() if entry.name.endswith(".csv")])
    coord_offset = np.array([0.4, 0.35])
    
    fig, ax = plt.subplots()

    for i, csv in enumerate(csvs):
        if i % 5 == 0:
            print(f"going through {i}th file out of {len(csvs)} in total")
        print("csv, ", csv)
        full_csv_path = path / csv
        df = pd.read_csv(full_csv_path, header=0)

        #plt.figure()  # ← ONE figure per csv, outside the episode loop

        for ep_id, ep_df in df.groupby("episode"):
            block_centered = ep_df[["block_x", "block_y"]].to_numpy(dtype=float)
            block_world = np.column_stack([
                -block_centered[:, 0],
                block_centered[:, 1],
            ]) + coord_offset

            blk_x = block_world[:, 0]
            blk_y = block_world[:, 1]
            start_x, start_y = block_world[0]
            end_x,   end_y   = block_world[-1]

            if not zoom_endpoints_only:
                ax.plot(blk_x, blk_y, color="blue", lw=0.01, alpha=0.7)
            ax.scatter(start_x, start_y, c="green", s=0.02)
            if not no_plot_ee_end:
                ax.scatter(end_x, end_y, c="peachpuff", s=0.02, marker="v",
                            edgecolors="black", linewidths=0.5)

        # labels for legend (just once, after the loop)
        ax.scatter([], [], c="green", s=0.02, label="block_start")
        ax.scatter([], [], c="peachpuff", s=0.02, marker="v", label="block_end")

        ax = plt.gca()
        # ax.set_xlim(0.4, 0.6)
        # ax.set_ylim(-0.3, 0.3)
        # ax.xaxis.set_major_locator(plt.MultipleLocator(0.01))
        # ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
        # ax.tick_params(axis='both', labelsize=4)
        # plt.xticks(rotation=90)
        # plt.legend()
        # out_path = Path(__file__).parents[1] / "data_collection" / "train_data_traj.png"
        # plt.savefig(out_path, dpi=150)
        if not return_option:
            plt.close()

    return fig, ax


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
        plot_path(path1)