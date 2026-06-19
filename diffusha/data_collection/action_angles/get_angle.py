import pandas as pd
import numpy as np

from pathlib import Path
import sys

def compute_theta_dir(dir_full_path):

    sample = dict()
    csv_paths = sorted(Path(dir_full_path).glob("*.csv"))

    for j, csv in enumerate(csv_paths):
        #print("j, ", j)

        csv_dict_key_name = "csv" + str(j)
        sample[csv_dict_key_name] = dict()

        df = pd.read_csv(csv)

        prev_x, prev_y = None, None

        end_x, end_y = None, None

        for i, (_, ep_df) in enumerate(df.groupby("episode", sort = True)):
            # iterate through ep_df and get each row

            ep_dict_key_name = "ep" + str(i)
            ep_df = ep_df.reset_index(drop=True)

            start_x, start_y = ep_df.iloc[0, 3], ep_df.iloc[0, 4]
            end_x, end_y = ep_df.iloc[-1, 3], ep_df.iloc[-1, 4]

            per_point_thetas = []
            start_to_cur_thetas = []
            prev_x, prev_y = None, None

            for row in ep_df.itertuples(index = False):
                ee_x = row[3]
                ee_y = row[4]

                if prev_x is not None:
                    dx = ee_x - prev_x
                    dy = ee_y - prev_y
                    per_point_theta = np.arctan2(dx, dy)

                    # per_point, incremental
                    print("***PER POINT***")
                    print("x, y, dx, dy, per_point_theta, ", ee_x, ee_y, dx, dy, per_point_theta)
                    per_point_theta = np.arctan2(dx, dy)
                    per_point_thetas.append(per_point_theta)
                    # print("**************")

                    # start to current point
                    print("***START TO CURRENT***")
                    dx = ee_x - start_x
                    dy = ee_y - start_y
                    start_to_cur_theta = np.arctan2(dx, dy)
                    print("x, y, dx, dy, start_to_cur_theta, ", ee_x, ee_y, dx, dy, start_to_cur_theta)
                    start_to_cur_theta = np.arctan2(dx, dy)
                    start_to_cur_thetas.append(start_to_cur_theta)
                    print("**************")

                prev_x, prev_y = ee_x, ee_y

            dx_total = end_x - start_x
            dy_total = end_y - start_y
            start_to_end_theta = np.arctan2(dx_total, dy_total)
            print("start_to_end_theta, ", start_to_end_theta)

            sample[csv_dict_key_name][ep_dict_key_name] = {
                "per_point_thetas": per_point_thetas,
                "start_to_cur_thetas": start_to_cur_thetas,
                "start_to_end_theta": start_to_end_theta,
            }
            break
        break
        

    return sample
    

def main():
    base = Path(__file__).parents[3] / "data-dir" / "replay" / "blockpush"
    print("base, ", base)
    dir_with_eps_csv = Path(sys.argv[1])

    dir_full_path = base / dir_with_eps_csv

    sample = compute_theta_dir(dir_full_path)

    # first_csv = sample["csv0"]
    # for ep_key in ["ep0", "ep1", "ep2"]:
    #     if ep_key not in first_csv:
    #         print(f"{ep_key} not found in csv0 (only {len(first_csv)} episodes)")
    #         continue

    #     ep_data = first_csv[ep_key]
    #     print(f"=== {ep_key} ===")
    #     print("per_point_thetas:", ep_data["per_point_thetas"])
    #     print("start_to_cur_thetas:", ep_data["start_to_cur_thetas"])
    #     print("start_to_end_theta:", ep_data["start_to_end_theta"])
    #     print()

if __name__ == "__main__":
    main()