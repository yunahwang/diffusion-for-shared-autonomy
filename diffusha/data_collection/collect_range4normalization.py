#!/usr/bin/env python3
import numpy as np
import pickle
from pathlib import Path
from .generate_data import ReplayBuffer

# match these to your training setup
# data_dirs = [
#     "/path/to/data-dir/replay/blockpush/target/randp_0.0",
#     "/path/to/data-dir/replay/blockpush/target-flipped/randp_0.0",
# ]
data_dirs = [
    Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "target" / "randp_0.0",
    Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "target-flipped" / "randp_0.0",
]

state_dim = 7   # block_translation(2) + block_orientation(1) + ee_translation(2) + ee_target_translation(2)
action_dim = 2
#n_samples = 50000  # how many samples to compute stats from

print(len(data_dirs))
all_obs = []; all_actions = []

for data_dir in data_dirs:
    rb = ReplayBuffer(data_dir, state_dim, action_dim)
    for fname, chunk in rb._file_cache.items():
        all_obs.append(chunk[:, :state_dim])  # just state part, drop action and q_val
        all_actions.append(chunk[:, state_dim:state_dim + action_dim])

all_obs = np.concatenate(all_obs, axis=0)  # (total_N, 7)
all_actions = np.concatenate(all_actions, axis = 0)
print("total samples:", all_obs.shape)
print("total action samples: ", all_actions.shape)

# compute per-key stats
normalizer_stats = {
    "block_translation":     {
        "mean": all_obs[:, 0:2].mean(0),
        "std":  all_obs[:, 0:2].std(0),
        "min":  all_obs[:, 0:2].min(0),
        "max":  all_obs[:, 0:2].max(0),
    },
    "block_orientation":     {
        "mean": all_obs[:, 2:3].mean(0),
        "std":  all_obs[:, 2:3].std(0),
        "min":  all_obs[:, 2:3].min(0),
        "max":  all_obs[:, 2:3].max(0),
    },
    "ee_translation":        {
        "mean": all_obs[:, 3:5].mean(0),
        "std":  all_obs[:, 3:5].std(0),
        "min":  all_obs[:, 3:5].min(0),
        "max":  all_obs[:, 3:5].max(0),
    },
    "ee_target_translation": {
        "mean": all_obs[:, 5:7].mean(0),
        "std":  all_obs[:, 5:7].std(0),
        "min":  all_obs[:, 5:7].min(0),
        "max":  all_obs[:, 5:7].max(0),
    },
    "action": {
        "mean": all_actions.mean(0),
        "std":  all_actions.std(0),
        "min":  all_actions.min(0),
        "max":  all_actions.max(0),
    },
}

for key, stats in normalizer_stats.items():
    print(f"{key}: mean={stats['mean']}, std={stats['std']}, min={stats['min']}, max={stats['max']}")

# save
with open("normalizer_stats.pkl", "wb") as f:
    pickle.dump(normalizer_stats, f)

print("saved to normalizer_stats.pkl")