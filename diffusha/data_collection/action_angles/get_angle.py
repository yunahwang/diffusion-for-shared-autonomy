import pandas as pd
import numpy as np
import pickle

from pathlib import Path
import sys

def compute_theta_dir(dir_full_path, which_metric):

    sample = dict()
    csv_paths = sorted(Path(dir_full_path).glob("*.csv"))

    first_val_idx, second_val_idx = None, None

    print("which_metric, ",which_metric)

    # NOTE: recall that the eps directory csv columns are as:
    # episode,file,step, (0 ~ 2)
    # block_x,block_y,block_ori, (3 ~ 5)
    # ee_x,ee_y,ee_target_x,ee_target_y, (6 ~ 9)
    # action_x,action_y, (10 ~ 11)


    if which_metric == "ee":
        first_val_idx = 6
        second_val_idx = 7
    else: # which_metric == "action"
        first_val_idx = 10
        second_val_idx = 11

    for j, csv in enumerate(csv_paths):
        if j % 10 == 0:
            print(f"at {j}-th csv out of 100")

        csv_dict_key_name = "csv" + str(j)
        sample[csv_dict_key_name] = dict()

        df = pd.read_csv(csv)

        prev_x, prev_y = None, None

        end_x, end_y = None, None

        for i, (_, ep_df) in enumerate(df.groupby("episode", sort = True)):
            # iterate through ep_df and get each row

            ep_dict_key_name = "ep" + str(i)
            ep_df = ep_df.reset_index(drop=True)

            start_x, start_y = ep_df.iloc[0, first_val_idx], ep_df.iloc[0, second_val_idx]
            end_x, end_y = ep_df.iloc[-1, first_val_idx], ep_df.iloc[-1, second_val_idx]

            per_point_thetas = []
            start_to_cur_thetas = []
            prev_x, prev_y = None, None

            for row in ep_df.itertuples(index = False):

                # can USE ACTION too
                x = row[first_val_idx]
                y = row[second_val_idx]

                if prev_x is not None:
                    dx = x - prev_x
                    dy = y - prev_y
                    # per_point, incremental
                    # print("***PER POINT***")
                    # print("x, y, dx, dy, per_point_theta, ", x, y, dx, dy, per_point_theta)
                    per_point_theta = np.arctan2(dx, dy)
                    per_point_thetas.append(per_point_theta)
                    # print("**************")

                    # start to current point
                    # print("***START TO CURRENT***")
                    dx = x - start_x
                    dy = y - start_y
                    start_to_cur_theta = np.arctan2(dx, dy)
                    # print("x, y, dx, dy, start_to_cur_theta, ", x, y, dx, dy, start_to_cur_theta)
                    start_to_cur_thetas.append(start_to_cur_theta)
                    # print("**************")

                prev_x, prev_y = x, y

            dx_total = end_x - start_x
            dy_total = end_y - start_y
            start_to_end_theta = np.arctan2(dx_total, dy_total)
            # print("start_to_end_theta, ", start_to_end_theta)

            sample[csv_dict_key_name][ep_dict_key_name] = {
                "per_point_thetas": per_point_thetas,
                "start_to_cur_thetas": start_to_cur_thetas,
                "start_to_end_theta": start_to_end_theta,
            }
        #     break
        # break
        

    return sample
    

def main():
    base = Path(__file__).parents[3] / "data-dir" / "replay" / "blockpush"
    print("base, ", base)
    dir_with_eps_csv = Path(sys.argv[1])
    pkl_file_name = Path(sys.argv[2])

    which_metric = Path(sys.argv[3]) # if "ee" then row[3], row[4] / if "action" then row[7], row[8]

    dir_full_path = base / dir_with_eps_csv

    sample = compute_theta_dir(dir_full_path, which_metric)

    with open(pkl_file_name, "wb") as file:
        pickle.dump(sample, file)

    # save in pickle

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