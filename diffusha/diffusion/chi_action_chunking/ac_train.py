#!/usr/bin/env python3
from typing import Optional
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.optim import optimizer
from torch.utils.data import IterableDataset, DataLoader
from diffusha.data_collection.env import is_lunarlander, make_env
from diffusha.diffusion.chi_action_chunking.ac_dataset import *
from diffusha.config.default_args import Args
import wandb

# from diffusha.diffusion.chi_action_chunking.ac_ddpm import(
#     DiffusionCore,
#     DiffusionModel,
#     Trainer
# )

T_A = 4
N_ACTION_STEPS = 4

OBS_COLS = [
    "block_x", "block_y", "block_ori",
    "ee_x", "ee_y",
    "ee_target_x", "ee_target_y",
]
ACT_COLS = ["action_x", "action_y"]

class ExpertTransitionDataset(IterableDataset):
    def __init__(
        self, directory, state_dim, action_dim, new_state_dim: int = 0, action_chunk_size: int = T_A,
        obs_cols: list = OBS_COLS, act_cols: list = ACT_COLS
    ) -> None:
        super().__init__()
        self.state_action_dim = state_dim + action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.new_state_dim = new_state_dim
        self.Ta = action_chunk_size

        self.buffer = EpisodeCSVBuffer(
            data_dirs=directory,
            obs_cols=obs_cols,
            act_cols=act_cols,
            action_chunk_size=action_chunk_size,
        )

    def __iter__(self):
        idx = self.buffer.index.copy()
        random.shuffle(idx)
        for ep_idx, t in idx:
            yield self.buffer._get(ep_idx, t)[0]

def main():
    make_eval_env = lambda **kwargs: make_env(
        Args.env_name,
        test=True,
        split_obs=("LunarLander" in Args.env_name),
        terminate_at_any_goal=True,
        **kwargs,
    )
    sample_env = make_eval_env()

    # NOTE:
    # - pilot_obs: may contain goal information  (pilot == user)
    # - copilot_obs: does not contain goal information  (copilot == assistant)
    act_size = sample_env.action_space.low.size
    pilot_obs_size = sample_env.observation_space.low.size
    if "LunarLander" in Args.env_name:
        copilot_obs_size = sample_env.copilot_observation_space.low.size
    else:
        copilot_obs_size = pilot_obs_size

    # print("this is data_dir, ", data_dir)
    if Args.dataset_envs is None:
        data_dir = '/data-dir/replay/blockpush/orig_2023_csv_with_eps'

        dataset = ExpertTransitionDataset(
            data_dir, pilot_obs_size, act_size, new_state_dim=copilot_obs_size
        )

    # get just one sample
    sample = next(iter(dataset))
    print(sample)
    print(sample.shape)  # should be (obs_dim + act_dim * Ta,) = (7 + 2*4,) = (15,)

    # loader = iter(DataLoader(dataset, batch_size=Args.batch_size, num_workers=8))

    # diffusion = DiffusionModel(
    #     # diffusion_core=DiffusionCore(small_noise_dim=copilot_obs_size, obs_noise_level=Args.obs_noise_level, obs_noise_cfg_prob=Args.obs_noise_cfg_prob),
    #     diffusion_core=DiffusionCore(),
    #     num_diffusion_steps=Args.num_diffusion_steps,
    #     input_size=(copilot_obs_size + act_size),
    #     beta_schedule=Args.beta_schedule,
    #     beta_min=Args.beta_min,
    #     beta_max=Args.beta_max,
    #     cond_dim=copilot_obs_size,
    # )

    # trainer = Trainer(
    #     diffusion,
    #     copilot_obs_size,
    #     act_size,
    #     save_every=Args.save_every,
    #     eval_every=Args.eval_every,
    # )

    # trainer.train(
    #     loader,
    #     make_eval_env=make_eval_env,
    #     num_training_steps=Args.num_training_steps,
    #     eval_assistance=True,
    # )


if __name__ == "__main__":
    # Apply patch on multiprocessing library
    from diffusha.utils import patch

    import argparse
    from params_proto.hyper import Sweep

    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_file", type=str, help="sweep file")
    parser.add_argument(
        "-l", "--line-number", type=int, help="line number of the sweep-file"
    )
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        avail_gpus = [0]  # Adjust as you like
        gpu_id = 0 if args.line_number is None else args.line_number % len(avail_gpus)
        cvd = avail_gpus[gpu_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

    # Obtain kwargs from Sweep and update hyperparameters accordingly
    sweep = Sweep(Args).load(args.sweep_file)
    kwargs = list(sweep)[args.line_number]
    print("kwargs, ", kwargs)
    Args._update(kwargs)

    sweep_basename = Path(args.sweep_file).stem
    print("sweep_basename, ", sweep_basename)

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="ac-target-only-2023-100",
        group=f"training-{sweep_basename}",
        config=vars(Args),
        #mode="offline"
        mode="online"
    )
    main()
    wandb.finish()
