#!/usr/bin/env python3
"""
Quick script to inspect observation layout from env and pkl files,
mirroring exactly how train.py sets things up.
"""

from pathlib import Path
import torch
import numpy as np

from diffusha.data_collection.env import make_env
from diffusha.data_collection.generate_data import ReplayBuffer
from diffusha.config.default_args import Args
from diffusha.data_collection.config.default_args import DCArgs


def main():
    # --- 1. Make env exactly like train.py does ---
    sample_env = make_env(
        Args.env_name,
        test=True,
        split_obs=False,  # same as train.py for BlockPush
        terminate_at_any_goal=True,
    )

    act_size = sample_env.action_space.low.size
    pilot_obs_size = sample_env.observation_space.low.size
    copilot_obs_size = pilot_obs_size  # BlockPush: same as pilot

    print("=" * 60)
    print(f"env_name:         {Args.env_name}")
    print(f"pilot_obs_size:   {pilot_obs_size}")
    print(f"copilot_obs_size: {copilot_obs_size}")
    print(f"act_size:         {act_size}")
    print("=" * 60)

    # --- 2. Print a live obs from env reset ---
    obs = sample_env.reset()
    print("\n--- Live obs from env.reset() ---")
    for i, val in enumerate(obs):
        print(f"  obs[{i:2d}] = {val:.6f}")

    # --- 3. Load a pkl file and print a raw sample ---
    # Adjust this path to point to your actual data directory
    data_dirs = [
        f"{Args.blockpush_data_dir}/target/randp_{Args.randp:.1f}",
        f"{Args.blockpush_data_dir}/target-flipped/randp_{Args.randp:.1f}",
    ]

    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"\nSkipping {data_dir} (not found)")
            continue

        pkl_files = list(data_path.glob("*.pkl"))
        if not pkl_files:
            print(f"\nNo pkl files found in {data_dir}")
            continue

        print(f"\n--- Raw pkl sample from {data_dir} ---")
        buff = torch.load(pkl_files[0])
        sample = buff[0]  # first row
        print(f"  full sample shape: {buff.shape}")
        print(f"  full sample (state + action + qval):")
        for i, val in enumerate(sample):
            label = ""
            if i < pilot_obs_size:
                label = f"obs[{i}]"
            elif i < pilot_obs_size + act_size:
                label = f"act[{i - pilot_obs_size}]"
            else:
                label = "q_val"
            print(f"  [{i:2d}] {label:10s} = {val:.6f}")

        # --- 4. Simulate exactly what MultiExpertTransitionDataset yields ---
        print(f"\n--- What the dataset actually yields (new_state_dim={copilot_obs_size}) ---")
        state_dim = pilot_obs_size
        action_dim = act_size
        new_state_dim = copilot_obs_size

        raw = sample[: state_dim + action_dim]  # drop q_val
        state = raw[:new_state_dim]
        act = raw[state_dim : state_dim + action_dim]
        yielded = np.concatenate((state, act), axis=-1)

        print(f"  yielded shape: {yielded.shape}")
        for i, val in enumerate(yielded):
            print(f"  yielded[{i:2d}] = {val:.6f}")

        break  # just inspect first valid dir


if __name__ == "__main__":
    main()