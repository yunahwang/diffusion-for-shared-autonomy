"""
# action sampling - forward, then reverse diffusion (Generating x_k, then x_i)
# pass those sets of actions, concatenating with states (obs) from whatever files i am reading from
# and then pass through noise_estimation_loss_nb_infer in 'loss_dist_ood.py'
"""
import sys
from pathlib import Path
import pandas as pd
import json
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

FWD_DIFF_RATIO = 1.0
ACTION_SAMPLING_NUM = 5 # NOTE: you can tune

def model_definition():
    from diffusha.data_collection.env import make_env
    from diffusha.diffusion.evaluation.helper import prepare_diffusha


    env_name =  "BlockPushMultimodal-v1"

    env = make_env(
        env_name,
        seed=1,
        test=False
    )

    obs_space = env.observation_space
    act_space = env.action_space

    with open(Path(__file__).parents[1] / "diffusion" / "evaluation" / "configs.json", "r") as f:
            env2config = json.load(f)

    model_dir = Path(__file__).parents[2] / "2023_100_ckpt"
    
    laggy_actor_repeat_prob = 0; noisy_actor_eps = 0

    diffusion = prepare_diffusha(
            env, 
            env2config[env_name], 
            model_dir,
            29999,
            env_name,
            FWD_DIFF_RATIO,
            laggy_actor_repeat_prob,
            noisy_actor_eps
        )
    
    return diffusion, obs_space, act_space

def get_states(dataset_folder):
    states = []
    for csv_file in sorted(Path(dataset_folder).glob("*.csv")):
        df = pd.read_csv(csv_file)
        for row in df.itertuples(index = False):
            states.append(row[0:9])
    return states

def plot_histogram():
    base = Path(__file__).parents[2]

    # NOTE: add/change folder paths here - each folder gets its own color/label
    folder_configs = [
        {
            "folder": base / "2023_100_ckpt" / "0.0" / "left_in_dist" / "trial_0528_thur_state_ood",
            "label": "Train (in-distribution)",
            "color": "steelblue",
        },
        {
            "folder": base / "2023_100_ckpt" / "0.0" / "right_ood" / "trial_0528_thur_state_ood",
            "label": "OOD (flipped)",
            "color": "tomato",
        },
    ]

    plt.figure(figsize=(8, 5))

    for config in folder_configs:
        folder = config["folder"]
        csv_files = sorted(folder.glob("episode_2.csv"))
        if not csv_files:
            print(f"No episode_*.csv found in {folder}")
            continue

        all_losses = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            all_losses.extend(df["loss"].values.tolist())

        plt.hist(all_losses, bins=50, alpha=0.6, label=config["label"], color=config["color"])
        print(f"{config['label']}: {len(all_losses)} steps across {len(csv_files)} episodes")

    plt.xlabel('Noise Estimation Loss')
    plt.ylabel('Count')
    plt.title('Loss Distribution: In-Distribution vs OOD')
    plt.legend()
    plt.tight_layout()
    plt.savefig(base / 'js_state_compare_histogram.png', dpi=150)
    plt.show()
    print(f"Histogram saved to {base / 'js_state_compare_histogram.png'}")

# def plot_histogram():
#     # NOTE: can change
#     base = Path(__file__).parents[1]
#     ood_path   = base / 'ood_losses_output_5_sampling.csv'
#     train_path = base / 'state_losses_output_5_sampling.csv'

#     ood_df   = pd.read_csv(ood_path)
#     train_df = pd.read_csv(train_path)

#     # flatten all action columns across all rows into one array each
#     ood_losses   = ood_df.values.flatten()
#     train_losses = train_df.values.flatten()

#     plt.figure(figsize=(8, 5))
#     plt.hist(train_losses, bins=50, alpha=0.6, label='Train (in-distribution)', color='steelblue')
#     plt.hist(ood_losses,   bins=50, alpha=0.6, label='OOD (flipped)',           color='tomato')
#     plt.xlabel('Noise Estimation Loss')
#     plt.ylabel('Count')
#     plt.title('Loss Distribution: In-Distribution vs OOD')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(base / 'state_compare_histogram.png', dpi=150)
#     #plt.show()
#     print(f"Histogram saved to {base / 'state_compare_histogram.png'}")

def main(dataset_folder):
    from diffusha.actor.assistive import DiffusionAssistedActor
    from diffusha.actor.loss_dist_ood import noise_estimation_loss_nb_infer

    # 1. folder definition - either training data or ood data
    # done already

    # 2. model definition
    diffusion, obs_space, act_space = model_definition()

    # 3. assisted actor definition -- with gamma 1.0
    assisted_actor = DiffusionAssistedActor(
                obs_space = obs_space,
                act_space = act_space,
                diffusion = diffusion,
                behavioral_actor = None,
                fwd_diff_ratio = FWD_DIFF_RATIO
            )

    # 4. grab obs and actions (call multiple actions, hence plural) from #1. and do action sampling with function 'act_without_env' in 'assistive.py'
    print("dataset_folder, ", dataset_folder)
    states = get_states(dataset_folder)
    print(len(states))

    losses_all_state = []
    start_time = time.time()

    for i, state in enumerate(states[:101]):
        # I GUESS 100 is enough to confirm in-distribution state loss
        # 5 actions - 100 states - 512 averaging per state: 10 seconds

        if i % 100 == 0:
            print(f"{i}-th state out of 1000000 states")

        obs = np.array(state[0:7], dtype=np.float32)      # convert tuple → numpy array
        action = np.array(state[7:len(state)], dtype=np.float32)
        
        assisted_actions = []; diffs = []; losses_per_state_action_pair = []
        for j in range(ACTION_SAMPLING_NUM):
            # if j % 5 == 0:
            #     print(f"{j}-th action")
            assisted_action, diff = assisted_actor.act_without_env(obs, action, report_diff = True)
            assisted_actions.append(assisted_action); diffs.append(diff)

            # 5. construct x_0_single with concatenation and pass through noise_esimtation_loss_nb_ifner
            # repeat...
            state_for_loss = np.concatenate([obs, assisted_action])            
            x_0_single = torch.tensor(state_for_loss, dtype = torch.float32).unsqueeze(0)

            noise_estimation_loss = noise_estimation_loss_nb_infer(diffusion, x_0_single, 7, 512)
            # <- so this is (state, action), action denoising for 512 times and loss
            losses_per_state_action_pair.append(noise_estimation_loss)

        losses_all_state.append(losses_per_state_action_pair)

    elapsed_time = time.time() - start_time
    print(f"time took: {elapsed_time}")

    #assert len(losses_all_state) == len(states)

    #print(losses_all_state)
    df = pd.DataFrame(
    losses_all_state,
    columns=[f'action_{j}' for j in range(ACTION_SAMPLING_NUM)]
    )
    csv_path = Path(__file__).parents[1] / 'ood_losses_output_5_sampling.csv' #NOTE
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        plot_histogram()
    else:
        folder = Path(sys.argv[1])
        # if training dataset - "orig", if ood dataset - "flipped"
        if "orig" in folder.name:
            dataset_folder = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "orig_2023_csv_backup"
        elif "flipped" in folder.name or "ood" in folder.name:
            dataset_folder = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "target-flipped" / "realsies_2023_flipped_csv_backup_100" 
        
        
        main(dataset_folder)