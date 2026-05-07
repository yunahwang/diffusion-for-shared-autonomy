#!/usr/bin/env python3
"""
Compute in-distribution diffusion losses over the training dataset
and save the full loss distribution + CDF metadata to a pickle file.

This script lives in diffusha/data_collection/ and is run as:
CUDA_VISIBLE_DEVICES="" python -m diffusha.data_collection.collect_losses_in_distribution
"""

import pickle
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import time

# ── adjust these to match your setup ──────────────────────────────────────────
STATE_DIM        = 7      # block_translation(2) + block_orientation(1) +
                          # ee_translation(2) + ee_target_translation(2)
ACTION_DIM       = 2
MAX_SAMPLES      = 1_000 # cap so it doesn't run forever; set None for all # FOR TIME SAKE (1000 samples - 50 ~ 60 minutes)
BATCH_MULTIPLIER = 32     # must match get_avg_diffusion_loss_ndata
NUM_PER_BATCH    = 1      # same
ALPHA_QUANTILES  = [0.50, 0.90, 0.95, 0.99]
MODE             = "limits"  # "limits" or "z_score", must match DaggerLoss init

ENV_NAME       = "BlockPushMultimodal-v1"
FWD_DIFF_RATIO = 0.6
MODEL_STEP     = 29999

# script is in diffusha/data_collection/ → parents[2] = repo root
_ROOT = Path(__file__).parents[2]

DATA_DIRS = [
    _ROOT / "data-dir" / "replay" / "blockpush" / "target"         / "randp_0.0",
    _ROOT / "data-dir" / "replay" / "blockpush" / "target-flipped" / "randp_0.0",
]
MODEL_DIR             = _ROOT / "tr3wtwfz"
NORMALIZER_STATS_PATH = _ROOT / "normalizer_stats.pkl"
OUTPUT_PATH           = _ROOT / "diffusion_loss_cdf.pkl"
CONFIGS_PATH = Path(__file__).parents[1] / "diffusion" / "evaluation" / "configs.json"
# ──────────────────────────────────────────────────────────────────────────────

OBS_KEYS = [
    "block_translation",
    "block_orientation",
    "ee_translation",
    "ee_target_translation",
]
OBS_SLICES = {
    "block_translation":     slice(0, 2),
    "block_orientation":     slice(2, 3),
    "ee_translation":        slice(3, 5),
    "ee_target_translation": slice(5, 7),
}


def load_normalizers(stats_path: Path, mode: str):
    """Rebuild normalizer objects from the saved stats pkl."""
    from diffusha.diffdagger.util import SafeLimitsNormalizer, GaussianNormalizer

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    normalizers = {}
    for key in stats.keys():   # includes "action"
        s = stats[key]
        if mode == "limits":
            dummy = torch.stack([
                torch.tensor(s["min"], dtype=torch.float32),
                torch.tensor(s["max"], dtype=torch.float32),
            ])
            normalizers[key] = SafeLimitsNormalizer(dummy)
        elif mode == "z_score":
            dummy = torch.stack([
                torch.tensor(s["mean"],            dtype=torch.float32),
                torch.tensor(s["mean"] + s["std"], dtype=torch.float32),
                torch.tensor(s["mean"] - s["std"], dtype=torch.float32),
            ])
            n = GaussianNormalizer(dummy)
            n.means = torch.tensor(s["mean"], dtype=torch.float32)
            n.stds  = torch.tensor(s["std"],  dtype=torch.float32).clamp(min=1e-8)
            normalizers[key] = n
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return normalizers


def normalize_obs(obs_vec: torch.Tensor, normalizers: dict) -> torch.Tensor:
    """obs_vec: 1-D tensor of length STATE_DIM -> normalized flat tensor."""
    parts = []
    for key in OBS_KEYS:
        chunk = obs_vec[OBS_SLICES[key]]
        parts.append(normalizers[key].normalize(chunk))
    return torch.cat(parts, dim=-1)


def normalize_action(action_vec: torch.Tensor, normalizers: dict) -> torch.Tensor:
    return normalizers["action"].normalize(action_vec)


def compute_loss_single(diffusion, obs_raw, action_raw, device,
                        batch_multiplier, num_per_batch):
    from diffusha.diffusion.utils import extract

    obs_raw    = obs_raw.to(device)
    action_raw = action_raw.to(device)

    nobs    = obs_raw.unsqueeze(0).unsqueeze(0)
    naction = action_raw.unsqueeze(0).unsqueeze(0)

    N = diffusion.num_diffusion_steps * batch_multiplier

    nobs_repeat    = nobs.repeat(N, 1, 1)
    naction_repeat = naction.repeat(N, 1, 1)

    x_0 = torch.cat([nobs_repeat, naction_repeat], dim=-1)

    timesteps = (
        torch.arange(N, device=device).long() % diffusion.num_diffusion_steps
    )

    KLD_loss = 0.0
    for _ in range(num_per_batch):
        # mirror training exactly — diffuse generates its own e internally
        x_t, e = diffusion.diffuse(x_0, timesteps)
        output = diffusion.model(x_t, timesteps)
        err = e - output
        KLD_loss += err.square().mean().item()

    return KLD_loss / num_per_batch

def main():
    # ── imports ────────────────────────────────────────────────────────────
    # generate_data lives in the same directory (diffusha/data_collection/)
    from .generate_data import ReplayBuffer
    from diffusha.diffusion.ddpm import DiffusionModel, DiffusionCore
    from diffusha.config.default_args import Args
    from diffusha.data_collection.env import make_env

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"device: {device}")

    # ── load env to get obs/act sizes (mirrors prepare_diffusha) ──────────
    env      = make_env(ENV_NAME, seed=0, test=True)
    obs_size = env.observation_space.low.size
    act_size = env.action_space.low.size
    print(f"obs_size={obs_size}  act_size={act_size}")

    # ── load configs.json → Args (same as joystick_diffdagger.py) ─────────
    with open(CONFIGS_PATH, "r") as f:
        env2config = json.load(f)
    Args._update(env2config[ENV_NAME])
    Args.fwd_diff_ratio          = FWD_DIFF_RATIO
    Args.laggy_actor_repeat_prob = 0
    Args.noisy_actor_eps         = 0

    # ── instantiate DiffusionModel (mirrors prepare_diffusha exactly) ─────
    diffusion = DiffusionModel(
        diffusion_core=DiffusionCore(),
        num_diffusion_steps=Args.num_diffusion_steps,
        input_size=(obs_size + act_size),
        beta_schedule=Args.beta_schedule,
        beta_min=Args.beta_min,
        beta_max=Args.beta_max,
        cond_dim=obs_size,
    )

    # ── load checkpoint ────────────────────────────────────────────────────
    model_path = MODEL_DIR / f"step_{MODEL_STEP:08d}.pt"
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    diffusion.model.load_state_dict(checkpoint["model"])
    diffusion.model.eval()
    diffusion.model.to(device)
    print("Model loaded.")

    # ── load normalizers ───────────────────────────────────────────────────
    normalizers = load_normalizers(NORMALIZER_STATS_PATH, MODE)
    print(f"Loaded normalizers from {NORMALIZER_STATS_PATH}")

    # ── iterate dataset, one sample at a time ─────────────────────────────
    # ── load all data first ────────────────────────────────────────────────
    all_obs     = []
    all_actions = []

    for data_dir in DATA_DIRS:
        rb = ReplayBuffer(data_dir, STATE_DIM, ACTION_DIM)
        for fname, chunk in rb._file_cache.items():
            all_obs.append(chunk[:, :STATE_DIM])
            all_actions.append(chunk[:, STATE_DIM:STATE_DIM + ACTION_DIM])

    all_obs     = np.concatenate(all_obs,     axis=0)  # [total_N, 7]
    all_actions = np.concatenate(all_actions, axis=0)  # [total_N, 2]
    total_N = len(all_obs)
    print(f"total obs samples:    {all_obs.shape}")
    print(f"total action samples: {all_actions.shape}")

    # ── iterate per datapoint ──────────────────────────────────────────────
    diffusion_losses = []
    limit = min(total_N, MAX_SAMPLES) if MAX_SAMPLES is not None else total_N

    print("limit, ", limit)

    start_total = time.time()

    for i in range(limit):
        obs_raw    = torch.tensor(all_obs[i],     dtype=torch.float32)
        action_raw = torch.tensor(all_actions[i], dtype=torch.float32)

        start_i = time.time()
        with torch.no_grad():
            loss = compute_loss_single(
                diffusion,
                obs_raw, action_raw,
                device, BATCH_MULTIPLIER, NUM_PER_BATCH,
            )
        elapsed_i = time.time() - start_i

        diffusion_losses.append(loss)

        print(f"  [{i}] loss={loss:.5f}  time={elapsed_i:.3f}s  "
            f"eta={elapsed_i * (limit - i - 1) / 60:.1f}min")

        if len(diffusion_losses) % 500 == 0:
            print(f"  [{len(diffusion_losses)}/{limit}] "
                f"mean={np.mean(diffusion_losses):.5f}  "
                f"std={np.std(diffusion_losses):.5f}")

    total_elapsed = time.time() - start_total
    print(f"\nTotal time: {total_elapsed / 60:.1f} min  ({total_elapsed:.1f}s)")

    # start_total = time.time()

    # for i in range(limit):
    #     obs_raw    = torch.tensor(all_obs[i],     dtype=torch.float32)
    #     action_raw = torch.tensor(all_actions[i], dtype=torch.float32)

    #     with torch.no_grad():
    #         loss = compute_loss_single(
    #             diffusion, normalizers,
    #             obs_raw, action_raw,
    #             device, BATCH_MULTIPLIER, NUM_PER_BATCH,
    #         )

    #     diffusion_losses.append(loss)

    #     print(f"  [{i}] "
    #               f"mean={np.mean(diffusion_losses):.5f}  "
    #               f"std={np.std(diffusion_losses):.5f}")
        
    #     if len(diffusion_losses) % 500 == 0:
    #         print(f"  [{len(diffusion_losses)}/{limit}] "
    #               f"mean={np.mean(diffusion_losses):.5f}  "
    #               f"std={np.std(diffusion_losses):.5f}")

    # ── compute CDF metadata ───────────────────────────────────────────────
    losses_arr = np.array(diffusion_losses)
    quantile_thresholds = {
        alpha: float(np.quantile(losses_arr, alpha))
        for alpha in ALPHA_QUANTILES
    }

    print(f"\nTotal datapoints : {len(losses_arr)}")
    print(f"Mean loss        : {losses_arr.mean():.5f}")
    print(f"Std  loss        : {losses_arr.std():.5f}")
    print(f"Min  loss        : {losses_arr.min():.5f}")
    print(f"Max  loss        : {losses_arr.max():.5f}")
    for alpha, thresh in quantile_thresholds.items():
        print(f"  q{alpha:.2f} threshold : {thresh:.5f}")

    # ── save ───────────────────────────────────────────────────────────────
    save_dict = {
        "losses":              losses_arr,
        "quantile_thresholds": quantile_thresholds,
        "mean":                float(losses_arr.mean()),
        "std":                 float(losses_arr.std()),
        "min":                 float(losses_arr.min()),
        "max":                 float(losses_arr.max()),
        "num_samples":         len(losses_arr),
        "batch_multiplier":    BATCH_MULTIPLIER,
        "num_per_batch":       NUM_PER_BATCH,
        "mode":                MODE,
        "alpha_quantiles":     ALPHA_QUANTILES,
        "env_name":            ENV_NAME,
        "model_step":          MODEL_STEP,
        "data_dirs":           [str(d) for d in DATA_DIRS],
        "timestamp":           datetime.now().isoformat(),
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(save_dict, f)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()