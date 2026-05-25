# from pathlib import Path
# import pandas as pd
# import numpy as np
# import random
# import json
# import torch

# from diffusha.data_collection.env import make_env
# from diffusha.diffusion.evaluation.helper import prepare_diffusha
# from diffusha.actor.loss_dist_ood import noise_estimation_loss_nb_infer


# def calculate_losses(all_states, diffusion, Nb=512):
#     losses = []
#     for i, state in enumerate(all_states):
#         if i % 20 == 0:
#             print(i)
#         # Convert 9d tuple to tensor of shape (1, 9)
#         x_0_single = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(diffusion.device)
#         loss = noise_estimation_loss_nb_infer(diffusion, x_0_single, obs_size=7, Nb=Nb)
#         losses.append(loss)
#     return losses # length of 1000 

    
# def main(data_dir):
#     all_csv_files = list(Path(data_dir).glob("*.csv"))
#     selected_files = random.sample(all_csv_files, min(10, len(all_csv_files)))
    
#     all_rows = []
#     for csv_file in sorted(selected_files):
#         df = pd.read_csv(csv_file)
#         sampled_rows = df.sample(n=min(10, len(df)))
#         sampled_rows["source_file"] = csv_file.name
#         all_rows.append(sampled_rows)
    
#     all_rows = pd.concat(all_rows, ignore_index=True)
    
#     all_states = [] # [(9d), (9d), ..., ] <- length of 1000
#     for _, row in all_rows.iterrows():
#         state = tuple(row.iloc[3:12])  # indices 3,4,5,6,7,8,9,10,11
#         all_states.append(state)

#     ### perturbation 1 - action x direction is opposite
#     all_states_action_opposite = []
#     for state in all_states:
#         state_list = list(state)
#         state_list[7] = -state_list[7]  # flip action x
#         all_states_action_opposite.append(tuple(state_list))
#     # conclusion to get out of this -- does changing the action a ton produce large losses

#     ### perturbation 2 - really small perturbation
#     # scale action to ~0 (0.01x)
#     all_states_small = [(*s[:7], s[7]*0.01, s[8]*0.01) for s in all_states]

#     # 6 - replace action with pure random noise
#     rng = np.random.default_rng(42)
#     all_states_random = [(*s[:7], *rng.standard_normal(2).tolist()) for s in all_states]

#     # 8 - add small Gaussian noise to obs (sigma=0.01)
#     rng = np.random.default_rng(42)
#     all_states_gaussian = [(*( np.array(s[:7]) + rng.standard_normal(7)*0.01 ).tolist(), s[7], s[8]) for s in all_states]

#     # add 0.02 to all obs and action dims
#     all_states_small_add = [tuple(x + 0.02 for x in s) for s in all_states]
#     # expectation for this is that losses should still be small

#     # ~10% of goal separation
#     all_states_perturbed = [tuple(x + 0.04 for x in s) for s in all_states]

#     # ~25% of goal separation  
#     all_states_perturbed = [tuple(x + 0.1 for x in s) for s in all_states]

#     # ~50% of goal separation
#     all_states_perturbed = [tuple(x + 0.2 for x in s) for s in all_states]


#     model_dir = Path(__file__).parents[2] / "2023_100_ckpt" 

#     env = make_env(
#         "BlockPushMultimodal-v1",
#         seed = 1,
#         test = False,
#     )
#     with open(Path(__file__).parents[1] / "diffusion" / "evaluation" / "configs.json", "r") as f:
#         env2config = json.load(f)

#     diffusion = prepare_diffusha(
#         env,
#         env2config["BlockPushMultimodal-v1"],
#         model_dir,
#         29999,
#         "BlockPushMultimodal-v1",
#         1.0,
#         0.0,
#         0.0,
#     )
    
#     print(diffusion, flush = True)

#     Nb = 512
#     losses = calculate_losses(all_states, diffusion, Nb=Nb)
#     print("losses, ", losses)
#     print(f"Mean loss: {sum(losses)/len(losses):.6f}")
#     print(f"Min loss:  {min(losses):.6f}")
#     print(f"Max loss:  {max(losses):.6f}")

#     action_opposite_losses = calculate_losses(all_states_action_opposite, diffusion, Nb=Nb)
#     print("action_opposite_losses, ", action_opposite_losses)
#     print(f"Mean action_opposite_losses: {sum(action_opposite_losses)/len(action_opposite_losses):.6f}")
#     print(f"Min action_opposite_losses:  {min(action_opposite_losses):.6f}")
#     print(f"Max action_opposite_losses:  {max(action_opposite_losses):.6f}")
        

# if __name__ == "__main__":
#     data_dir = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "orig_2023_csv_with_eps"
#     main(data_dir)

from pathlib import Path
import pandas as pd
import numpy as np
import random
import json
import torch

from diffusha.data_collection.env import make_env
from diffusha.diffusion.evaluation.helper import prepare_diffusha
from diffusha.actor.loss_dist_ood import noise_estimation_loss_nb_infer


def calculate_losses(all_states, diffusion, Nb=512):
    losses = []
    for i, state in enumerate(all_states):
        if i % 20 == 0:
            print(f"  {i}/{len(all_states)}", flush=True)
        x_0_single = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(diffusion.device)
        loss = noise_estimation_loss_nb_infer(diffusion, x_0_single, obs_size=7, Nb=Nb)
        losses.append(loss)
    return losses


def build_perturbations(all_states):
    rng = np.random.default_rng(42)
    return {
        "baseline":             all_states,
        "flip_action_x":        [(*s[:7], -s[7],        s[8])          for s in all_states],
        "scale_action_0.01x":   [(*s[:7], s[7]*0.01,    s[8]*0.01)     for s in all_states],
        "scale_action_0.05x":   [(*s[:7], s[7]*0.05,    s[8]*0.05)     for s in all_states],
        "scale_action_2x":      [(*s[:7], s[7]*2,       s[8]*2)         for s in all_states],
        # scale training actions up to joystick magnitude (~15x and ~20x)
        "scale_action_15x":  [(*s[:7], s[7]*15.0, s[8]*15.0) for s in all_states],
        "scale_action_20x":  [(*s[:7], s[7]*20.0, s[8]*20.0) for s in all_states],
        "random_action":        [(*s[:7], *rng.standard_normal(2))      for s in all_states],
        "obs_gaussian_0.01":    [(*( np.array(s[:7]) + rng.standard_normal(7)*0.01 ).tolist(), s[7], s[8]) for s in all_states],
        "add_0.02_all":         [tuple(x + 0.02 for x in s)            for s in all_states],
        "add_0.04_all":         [tuple(x + 0.04 for x in s)            for s in all_states],
        "add_0.10_all":         [tuple(x + 0.10 for x in s)            for s in all_states],
        "add_0.20_all":         [tuple(x + 0.20 for x in s)            for s in all_states],
    }


def main(data_dir):
    # --- load data ---
    all_csv_files = list(Path(data_dir).glob("*.csv"))
    selected_files = random.sample(all_csv_files, min(10, len(all_csv_files)))

    all_rows = []
    for csv_file in sorted(selected_files):
        df = pd.read_csv(csv_file)
        sampled_rows = df.sample(n=min(10, len(df)))
        sampled_rows["source_file"] = csv_file.name
        all_rows.append(sampled_rows)

    all_rows = pd.concat(all_rows, ignore_index=True)

    all_states = []
    for j, (_, row) in enumerate(all_rows.iterrows()):
        state = tuple(row.iloc[3:12])
        all_states.append(state)
        if j == 51 or j == 10:
            print("state", state)

    # --- load model ---
    # model_dir = Path(__file__).parents[2] / "2023_100_ckpt"
    # env = make_env("BlockPushMultimodal-v1", seed=1, test=False)
    # with open(Path(__file__).parents[1] / "diffusion" / "evaluation" / "configs.json", "r") as f:
    #     env2config = json.load(f)

    # diffusion = prepare_diffusha(
    #     env, env2config["BlockPushMultimodal-v1"], model_dir,
    #     29999, "BlockPushMultimodal-v1", 1.0, 0.0, 0.0,
    # )
    # print(diffusion, flush=True)

    # # --- run perturbations ---
    # perturbations = build_perturbations(all_states)

    # results = {}
    # summary_rows = []
    # for name, states in perturbations.items():
    #     print(f"\n--- {name} ---", flush=True)
    #     losses = calculate_losses(states, diffusion, Nb=512)
    #     results[name] = losses
    #     arr = np.array(losses)
    #     summary_rows.append({
    #         "perturbation": name,
    #         "mean": arr.mean(),
    #         "std":  arr.std(),
    #         "min":  arr.min(),
    #         "max":  arr.max(),
    #     })
    #     print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")

    # # --- save per-sample losses (one column per perturbation) ---
    # out_dir = Path(__file__).parent
    # losses_df = pd.DataFrame(results)   # each column = one perturbation, each row = one sample
    # losses_df.to_csv(out_dir / "perturbation_losses.csv", index_label="sample_idx")
    # print(f"\nSaved per-sample losses → {out_dir / 'perturbation_losses.csv'}")

    # # --- save summary ---
    # summary_df = pd.DataFrame(summary_rows)
    # summary_df.to_csv(out_dir / "perturbation_summary.csv", index=False)
    # print(f"Saved summary          → {out_dir / 'perturbation_summary.csv'}")


if __name__ == "__main__":
    data_dir = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "orig_2023_csv_with_eps"
    main(data_dir)