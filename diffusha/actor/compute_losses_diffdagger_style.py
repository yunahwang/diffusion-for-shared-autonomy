"""
train data csv rollout, read each line of each csv, repeat 512 times for the same line (same sample) and 
use the average value as "the loss", save in a csv to be used in loss_dist_ood.py
"""

from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
import os

from diffusha.diffusion.evaluation.helper import prepare_diffusha
from diffusha.diffusion.ddpm import DiffusionModel
from diffusha.actor.loss_dist_ood import noise_estimation_loss_nb_infer

CSV_PATH = "noise_estimation_log.csv"

def _append_to_csv(t, e, output, err, path=CSV_PATH):
    """Append Nb rows (one per sample) to a single CSV file."""
    # Detach and move to CPU
    t_cpu = t.cpu().numpy()           # (Nb,)
    e_cpu = e.detach().cpu().numpy()          # (Nb, 9)
    out_cpu = output.detach().cpu().numpy()   # (Nb, 9)
    err_cpu = err.detach().cpu().numpy()      # (Nb, 9)

    dim = e_cpu.shape[1]  # 9

    # Build header only if file doesn't exist yet
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            e_cols   = [f"e_{i}"      for i in range(dim)]
            out_cols = [f"output_{i}" for i in range(dim)]
            err_cols = [f"err_{i}"    for i in range(dim)]
            writer.writerow(["t"] + e_cols + out_cols + err_cols)
        for i in range(len(t_cpu)):
            row = (
                [t_cpu[i]]
                + e_cpu[i].tolist()
                + out_cpu[i].tolist()
                + err_cpu[i].tolist()
            )
            writer.writerow(row)


def compute_train_loss_512_repeat(diffusion, train_csv_folder, Nb=512):
    import time
    states = []; losses = []

    csv_files = sorted(Path(train_csv_folder).glob("*.csv"))
    n_files = len(csv_files)
    file_times = []

    for i, csv_file in enumerate(csv_files):

        if i == 25:
            break

        file_start = time.time()

        df = pd.read_csv(csv_file)
        per_file_states = []
        for j, row in enumerate(df.itertuples(index=False)):

            if j % 1000 == 0:
                print(f"{j}th row in {i}th file")

            state = list(row)[0:9]
            per_file_states.append(state)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                loss = noise_estimation_loss_nb_infer(diffusion, state_tensor, 7, Nb)
            losses.append(loss)
            #print("loss, ", loss)
        #print("losses, ", losses)

        states.append(per_file_states)

        file_elapsed = time.time() - file_start
        file_times.append(file_elapsed)

        avg_time = np.mean(file_times)
        files_remaining = n_files - (i + 1)
        eta_seconds = avg_time * files_remaining
        eta_min = eta_seconds / 60

        print(f"[{i+1}/{n_files}] {csv_file.name} | "
              f"took {file_elapsed:.1f}s | "
              f"avg {avg_time:.1f}s/file | "
              f"ETA {eta_min:.1f} min")

    return states, losses

def summarize_losses(losses, title="Loss Distribution"):
    losses = np.array(losses)
    
    print(f"\n=== {title} ===")
    print(f"n samples : {len(losses)}")
    print(f"min       : {losses.min():.4f}")
    print(f"p25       : {np.percentile(losses, 25):.4f}")
    print(f"p50       : {np.percentile(losses, 50):.4f}")
    print(f"p75       : {np.percentile(losses, 75):.4f}")
    print(f"p99       : {np.percentile(losses, 99):.4f}")
    print(f"max       : {losses.max():.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(losses, bins=100, color="steelblue", alpha=0.8)
    for q, label in [(25, "p25"), (50, "p50"), (75, "p75"), (99, "p99")]:
        val = np.percentile(losses, q)
        ax.axvline(val, linestyle="--", linewidth=1.2, label=f"{label}={val:.3f}")
    ax.set_xlabel("loss")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    png_path = "2023-100-losses_1909.png"
    plt.savefig(png_path, dpi = 150)
    #plt.show()

if __name__ == '__main__':
    from diffusha.data_collection.env import make_env

    fwd_diff_ratio = 0.0

    env_name =  "BlockPushMultimodal-v1"

    env = make_env(
        env_name,
        seed=1,
        test=False
    )

    obs_space = env.observation_space
    act_space = env.action_space
    print("obs_space, ", obs_space)
    print("act_space, ", act_space)
    
    with open(Path(__file__).parents[1] / "diffusion" / "evaluation" / "configs.json", "r") as f:
        env2config = json.load(f)

    which_ckpt = Path(sys.argv[1])
    model_dir = Path(__file__).parents[2] / which_ckpt
    print("model_dir, ", model_dir)

    laggy_actor_repeat_prob = 0; noisy_actor_eps = 0

    diffusion = prepare_diffusha(
            env, 
            env2config[env_name], 
            model_dir,
            29999,
            env_name,
            fwd_diff_ratio,
            laggy_actor_repeat_prob,
            noisy_actor_eps
        )
    print(diffusion, flush=True)


    # rollout train csvs
    if "2023" in which_ckpt.name:
        train_csv_folder = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "orig_2023_csv_backup"

    states, losses = compute_train_loss_512_repeat(diffusion, train_csv_folder)    

    summarize_losses(losses, title="Train Loss Distribution (Nb=512)")
    


    losses_csv_path = "2023-100-losses_1909.csv" # NOTE - move later
    #pd.DataFrame({"loss": losses}).to_csv(losses_csv_path, index=False)
    print(f"saved to {losses_csv_path}")



