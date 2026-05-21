"""
Plots losses for training data and ood data

1. load loss data from csv
2. load a series of ood state data from target2 pickle files
3. instantiate a diffusion model with gamma = 0.0
4. input ood state from 2. and plot losses on the same plot after accumulating them
"""

import pandas as pd
import json
import numpy as np
import sys
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # must be before importing pyplot
import matplotlib.pyplot as plt

from diffusha.diffusion.evaluation.helper import prepare_diffusha

#from diffusha.actor.compute_losses_diffdagger_style import noise_estimation_loss_nb


def load_train_loss(csv_path):
    df = pd.read_csv(csv_path)
    return df["loss"].reset_index(drop=True)

def load_ood_loss(csv_path):
    df = pd.read_csv(csv_path)
    return df["loss"].reset_index(drop = True)

def load_ood_states(ood_data_path):
    states = []
    for csv_file in sorted(Path(ood_data_path).glob("*.csv")):
        df = pd.read_csv(csv_file)
        per_file_states = []
        for row in df.itertuples(index=False):
            per_file_states.append(list(row)[0:9])
        states.append(per_file_states)

    states = np.array(states)

    return states


def quantile_analysis(losses, all_losses, ood_states):
    quantiles = [25, 50, 75, 99]
    #ood_flat = np.concatenate(all_losses)
    ood_flat = np.array(all_losses)
    ood_states_flat = np.concatenate(ood_states)  

    train_q = np.percentile(losses, quantiles)
    ood_q   = np.percentile(ood_flat, quantiles)

    print(f"{'quantile':<12} {'train':>10} {'ood':>10}")
    print("-" * 34)
    for q, t, o in zip(quantiles, train_q, ood_q):
        print(f"p{q:<11} {t:>10.4f} {o:>10.4f}")

    # grab states closest to each quantile threshold
    representative_states = {}
    for q, threshold in zip(quantiles, ood_q):
        idx = np.argmin(np.abs(ood_flat - threshold))
        representative_states[f"p{q}"] = {
            "loss": ood_flat[idx],
            "state": ood_states_flat[idx]
        }
        print(f"p{q} — loss: {ood_flat[idx]:.4f}, state: {ood_states_flat[idx]}")

    return train_q, ood_q, representative_states


def plot_ood_on_train_traj(rep_states, path1, path2=None):
    from diffusha.data_collection.viz_csvs import plot_path, plot_two_paths
    from matplotlib.patches import Rectangle

    def obs_xy_to_plot_xy(x, y):
        return np.array([
            0.8 + x + 0.4,
             y + 0.35,
        ])

    def obs_action_to_plot_vec(ax_, ay_):
        # Same x flip as positions, no offset for vectors
        return np.array([
            -ax_,
             ay_,
        ])

    def draw_target_box(ax, center, size, color, label):
        ax.add_patch(
            Rectangle(
                (center[0] - size / 2, center[1] - size / 2),
                size,
                size,
                facecolor=color,
                edgecolor="black",
                alpha=0.75,
                label=label,
                zorder=4,
            )
        )

    #fig, ax = plt.subplots(figsize=(8, 6))

    if path2 is not None:
        plot_two_paths(
            path1,
            path2,
            zoom_endpoints_only=False,
            no_plot_ee_end=True,
        )
        ax = plt.gca()
    else:
        fig, ax = plot_path(
            path1,
            zoom_endpoints_only=False,
            no_plot_ee_end=True,
            # use_saved_png = True
            return_option = True
        )
        #ax = plt.gca()

    # ------------------------------------------------------------
    # Draw target boxes
    # ------------------------------------------------------------
    # Start with actual env-inspired positions, but adjust if needed.
    # Green is placed over/near the blue clump.
    # Red is slightly to the +x side from viewer perspective here.
    green_center = np.array([0.50 + 0.8 , 0.3])
    red_center   = np.array([0.30 + 0.8, 0.3])

    target_size = 0.02

    draw_target_box(
        ax,
        green_center,
        target_size,
        "green",
        "green/right target",
    )

    draw_target_box(
        ax,
        red_center,
        target_size,
        "red",
        "red/left target",
    )

    # ------------------------------------------------------------
    # Overlay representative OOD states + action arrows
    # ------------------------------------------------------------
    colors = {
        "p25": "green",
        "p50": "slategray",
        "p75": "orange",
        "p99": "red",
    }

    arrow_scale = 0.1

    for key, val in rep_states.items():
        s = val["state"]

        # end effector and action extraction
        ee_x, ee_y = s[3], s[4]
        act_x, act_y = s[7], s[8]

        star_xy = obs_xy_to_plot_xy(ee_x, ee_y)
        action_vec = obs_action_to_plot_vec(act_x, act_y)

        # ax.scatter(
        #     star_xy[0],
        #     star_xy[1],
        #     c=colors[key],
        #     s=80,
        #     marker="*",
        #     edgecolors="black",
        #     linewidths=0.8,
        #     label=f"{key} of ee pos (loss={val['loss']:.3f})",
        #     zorder=8,
        # )

        # ax.arrow(
        #     star_xy[0],
        #     star_xy[1],
        #     arrow_scale * action_vec[0],
        #     arrow_scale * action_vec[1],
        #     color=colors[key],
        #     width=0.006,
        #     head_width=0.045,
        #     head_length=0.055,
        #     length_includes_head=True,
        #     alpha=0.9,
        #     zorder=9,
        # )

        # block position extraction
        blk_x, blk_y = s[0], s[1]
        block_xy = obs_xy_to_plot_xy(blk_x, blk_y)

        ax.scatter(
            block_xy[0],
            block_xy[1],
            c=colors[key],
            s=100,
            marker="o",
            edgecolors="black",
            linewidths=0.8,
            alpha=0.5,
            label=f"{key} block pos (loss={val['loss']:.3f})",
            zorder=7,
        )

        ax.arrow(
            block_xy[0],
            block_xy[1],
            arrow_scale * action_vec[0],
            arrow_scale * action_vec[1],
            color=colors[key],
            width=0.004,
            head_width=0.025,
            head_length=0.035,
            length_includes_head=True,
            alpha=0.9,
            zorder=9,
        )



    #ax.legend(loc="lower left", bbox_to_anchor=(1,1))
    ax.legend(fontsize=8)
    ax.set_title("Trajectory Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    #ax.grid(True)
    ax.axis("equal")

    plt.tight_layout()

    png_path = (
        Path(__file__).parents[1]
        / "data_collection"
        / "ood_on_train_traj.png"
    )
    plt.savefig(png_path, dpi=150)
    print(f"Saved plot to {png_path}")
    # plt.show()


def histogram_overlap(train_losses, ood_losses):
    fig, ax = plt.subplots(figsize=(10, 4))

    train_arr = np.array(train_losses).flatten()
    ood_arr   = np.array(ood_losses).flatten()

    train_weights = np.ones(len(train_arr)) / len(train_arr) * 100
    ood_weights   = np.ones(len(ood_arr))   / len(ood_arr)   * 100

    ax.hist(train_arr, bins=100, color="lightpink", alpha=1.0, weights=train_weights, label="train")
    ax.hist(ood_arr,   bins=100, color="steelblue", alpha=0.8, weights=ood_weights,   label="ood")
    ax.set_xlabel("loss")
    ax.set_ylabel("% of samples")

    ax.set_title("Loss distribution train vs ood data")
    ax.legend()
    plt.tight_layout()
    png_path = Path(__file__).parents[1] / "data_collection" / "train_vs_ood_losses_1909.png"
    plt.savefig(png_path, dpi = 150)

def histogram_ood(path):
    df = pd.read_csv(path)

    losses = np.array(df["loss"].values)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(losses, bins=100, color="steelblue", alpha=0.8)
    for q, label in [(25, "p25"), (50, "p50"), (75, "p75"), (99, "p99")]:
        val = np.percentile(losses, q)
        ax.axvline(val, linestyle="--", linewidth=1.2, label=f"{label}={val:.3f}")
    
    # also plot train loss for comparison
    ax.axvline(0.0204, linestyle="--", linewidth=1.2, color="red", label="train p25=0.020")
    ax.axvline(0.0378, linestyle="--", linewidth=1.2, color="red", label="train p50=0.038")
    ax.axvline(0.0744, linestyle="--", linewidth=1.2, color="red", label="train p75=0.074")
    ax.axvline(0.3233, linestyle="--", linewidth=1.2, color="red", label="train p99=0.323")

    ax.set_xlabel("loss")
    ax.set_ylabel("count")
    ax.set_title("Loss distribution of ood data")
    ax.legend()
    plt.tight_layout()
    png_path = Path(__file__).parents[1] / "data_collection" / "ood-losses_1909.png"
    plt.savefig(png_path, dpi = 150)
    #plt.show()

def noise_estimation_loss_nb_infer(diffusion, x_0_single, obs_size=7, Nb=512):
    x_0 = x_0_single.repeat(Nb, 1)  # (Nb, 9)
    #print("shape, ", x_0.shape)
    
    obs = x_0[:, :obs_size]      # clean, never noised
    action = x_0[:, obs_size:]   # only this gets noised
    #print("shape, ", action.shape)
    
    t = torch.randint(0, diffusion.num_diffusion_steps, size=(Nb // 2 + 1,))
    t = torch.cat([t, diffusion.num_diffusion_steps - t - 1], dim=0)[:Nb].long()
    t = t.to(diffusion.device)
    
    # noise only action
    x_t_action, e = diffusion.diffusion_core.diffuse(diffusion, action, t, cond_dim=0)

    #print("action, ", action, "x_t_action, ", x_t_action) # action is 512 copies of the same thing, x_t changes all of those 512 copies
    
    # clamp obs back in — exactly like inference does
    x_t = torch.cat([obs, x_t_action], dim=1)

    #print("x_0, ", x_0, "x_t, ", x_t)
    
    with torch.no_grad():
        output = diffusion.model(x_t, t)
    
    # loss only on action dims
    err = e - output[:, obs_size:]
    return err.square().mean().item()

if __name__ == "__main__":

    print("starting,,, ")

    which_ckpt = Path(sys.argv[1]) # folder name only

    # from diffusha.data_collection.env import make_env

    import time

    # fwd_diff_ratio = 0.0 # GAMMA = 0!

    # env_name =  "BlockPushMultimodal-v1"

    # env = make_env(
    #     env_name,
    #     seed=1,
    #     test=False
    # )

    # obs_space = env.observation_space
    # act_space = env.action_space
    # print("obs_space, ", obs_space)
    # print("act_space, ", act_space)
    
    # with open(Path(__file__).parents[1] / "diffusion" / "evaluation" / "configs.json", "r") as f:
    #     env2config = json.load(f)

    # model_dir = Path(__file__).parents[2] / which_ckpt
    # print("model_dir, ", model_dir)

    # NOTE - always use 2023
    if "2023" in which_ckpt.name:
        train_loss_csv_path = Path(__file__).parents[1] / "data_collection" / "2023-100-losses_1909.csv"
        # fyi the "2023-100-loss.csv" is from the 4096 batch-average loss (from wandb directly)

    print("train csv_path, ", train_loss_csv_path)

    # 1. load loss data from csv 
    train_losses = load_train_loss(train_loss_csv_path)

    # 2. load a series of ood state data from target2 pickle files
    ood_data_path = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "target-flipped" / "realsies_2023_flipped_csv_backup_100"
    print("ood_data_path, ", ood_data_path)
    states = load_ood_states(ood_data_path)
    print("shape of states array, ", states.shape)
    print("length of states array, ", len(states))

    # # 3. instantiate a diffusion model with gamma = 0.0
    # laggy_actor_repeat_prob = 0; noisy_actor_eps = 0

    # diffusion = prepare_diffusha(
    #         env, 
    #         env2config[env_name], 
    #         model_dir,
    #         29999,
    #         env_name,
    #         fwd_diff_ratio,
    #         laggy_actor_repeat_prob,
    #         noisy_actor_eps
    #     )
    # print(diffusion, flush=True)

    # 4. input ood state from 2. and plot losses on the same plot after accumulating them
    # 4-1. calculate losses 

    # Nb = 512

    # all_losses = []; file_times = []; all_losses_flattened = []

    # for i in range(len(states)):
    #     if i == 1:
    #         break
    #     losses_for_file = []

    #     states_for_file = states[i]
    #     file_start = time.time()

    #     for j, state in enumerate(states_for_file):

    #         if j % 100 == 0:
    #             print(f"going through {j}-th row in {i}-th file")
    #         state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(0) # this is a single row
            
    #         with torch.no_grad():
    #             loss = noise_estimation_loss_nb_infer(diffusion, state_tensor, obs_size=7, Nb=Nb)
    #         losses_for_file.append(loss)

    #     file_elapsed = time.time() - file_start
    #     file_times.append(file_elapsed)

    #     avg_time = np.mean(file_times)
    #     files_remaining = len(states) - (i + 1)
    #     eta_seconds = avg_time * files_remaining
    #     eta_min = eta_seconds / 60

    #     print(f"[{i+1}/{len(states)}] | "
    #           f"took {file_elapsed:.1f}s | "
    #           f"avg {avg_time:.1f}s/file | "
    #           f"ETA {eta_min:.1f} min")

    #     all_losses.append(losses_for_file)

    #     all_losses_flattened.extend(losses_for_file)

    # df = pd.DataFrame({"loss": all_losses_flattened})
    #df.to_csv("flipped(ood)_vs_2023_100_1824.csv", index = False)
    # print("ood losses saved to csv")
    
    # 4-2. ood losses plot only
    #plot_ood_losses_1(all_losses)
    #plot_ood_losses_2(all_losses)

    #overlap_train_ood_losses(losses, all_losses) # NOTE: this could be possibly useful

    ood_loss_csv_path = Path(__file__).parents[1] / "data_collection" / "flipped(ood)_vs_2023_100_1824.csv"

    # quantile analysis - 25, 50, 75, 99 quantile for train vs ood
    ood_losses = load_ood_loss(ood_loss_csv_path)

    train_q, ood_q, rep_states = quantile_analysis(train_losses, ood_losses, states)

    # grab state for p25 ood, p50 ood, p75 ood, and p99 ood based on quantile analysis results
    # quantile_analysis may return instead of plain print
    # <- use case: rep_states["p99"]["state"] or rep_states["p99"]["state"][3:5] or rep_states["p99"]["loss"]

    """
        quantile          train        ood
    ----------------------------------
    p25              0.0204     0.8222
    p50              0.0378     1.6512
    p75              0.0744     2.4204
    p99              0.3233    10.3180
    p25 — loss: 0.8222, state: [-0.07266843 -0.3394752   2.29672971 -0.10618594 -0.4056112  -0.1130075
    -0.40914816  0.59889114  0.91461116]
    p50 — loss: 1.6511, state: [ 0.09393216 -0.09233418  1.74941605  0.10060048 -0.121089    0.10059506
    -0.11280567 -0.01937947  0.39339563]
    p75 — loss: 2.4204, state: [ 0.09274558 -0.31971264  3.04027811  0.09261094 -0.3570272   0.09599178
    -0.3490764  -0.06378412  0.9159775 ]
    p99 — loss: 10.3178, state: [ 3.60693370e-02 -4.09319880e-01  1.25367755e+00 -6.61705450e-04
    -7.48088400e-01  0.00000000e+00 -7.50000000e-01 -2.73053650e-01
    8.80769700e-01]

    """
    
    #histogram_ood(ood_loss_csv_path) # this is the ood 

    #histogram_overlap(train_losses, ood_losses)

    # now plot these rep_states -- ood states on an existing, in-distribution trajectory
    
    #training_data_csv_path = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "orig_2023_csv_backup"
    training_data_csv_path = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "orig_2023_csv_with_eps"
    plot_ood_on_train_traj(rep_states, path1 = training_data_csv_path)