import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def make_arrays(target, target_flipped):
    target_arr = [] # also across all episodes and csvs
    for csv_dict_key_name, ep_dict in target.items():
        for ep_dict_key_name, sample in ep_dict.items():
            start_current_thetas = sample["start_to_cur_thetas"]
            target_arr.extend(start_current_thetas)

    target_flipped_arr = [] # also across all episodes and csvs
    for csv_dict_key_name, ep_dict in target_flipped.items():
        for ep_dict_key_name, sample in ep_dict.items():
            start_current_thetas = sample["start_to_cur_thetas"]
            target_flipped_arr.extend(start_current_thetas)
    return np.array(target_arr), np.array(target_flipped_arr)

def main():
    base = Path(__file__).parent
    with open(base / "action_target_2023_angle.pkl", "rb") as f:
        target = pickle.load(f)
    with open(base / "action_target_flipped_2023_angle.pkl", "rb") as f:
        target_flipped = pickle.load(f)

    target_arr, target_flipped_arr = make_arrays(target, target_flipped)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(target_arr, bins=60, alpha=0.5, density=True, label="target (ID)")
    ax.hist(target_flipped_arr, bins=60, alpha=0.5, density=True, label="flipped (OOD)")
    ax.set_xlabel("start-to-current theta (rad)")
    ax.set_ylabel("density")
    ax.set_title("Action angle distribution: target vs flipped")
    ax.legend()
    plt.tight_layout()
    plt.savefig("theta_histogram.png", dpi=150)
    print("saved theta_histogram.png")
    print(f"target:  n={len(target_arr)}, mean={target_arr.mean():.3f}, std={target_arr.std():.3f}")
    print(f"flipped: n={len(target_flipped_arr)}, mean={target_flipped_arr.mean():.3f}, std={target_flipped_arr.std():.3f}")

if __name__ == "__main__":
    main()