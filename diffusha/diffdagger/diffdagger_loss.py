from collections import deque
import torch
from pathlib import Path
import pickle
from .util import SafeLimitsNormalizer, GaussianNormalizer

OBS_KEYS = ["block_translation", "block_orientation", "ee_translation", "ee_target_translation"]

class DaggerLoss:
    def __init__(self, diffusion, mode="limits"):
        self.diffusion = diffusion
        self.device = "cpu"
        self.mode = mode  # "limits" or "z_score"

        # load stats once at init
        stats_path = Path(__file__).parents[2] / "normalizer_stats.pkl"
        with open(stats_path, "rb") as f:
            self.normalizer_stats = pickle.load(f)

        # build normalizer objects once at init
        self.normalizers = {}
        for key in OBS_KEYS:
            stats = self.normalizer_stats[key]
            if mode == "limits":
                dummy = torch.stack([
                    torch.tensor(stats["min"], dtype=torch.float32),
                    torch.tensor(stats["max"], dtype=torch.float32),
                ])
                self.normalizers[key] = SafeLimitsNormalizer(dummy)
            elif mode == "z_score":
                dummy = torch.stack([
                    torch.tensor(stats["mean"], dtype=torch.float32),
                    torch.tensor(stats["mean"] + stats["std"], dtype=torch.float32),
                    torch.tensor(stats["mean"] - stats["std"], dtype=torch.float32),
                ])
                normalizer = GaussianNormalizer(dummy)
                normalizer.means = torch.tensor(stats["mean"], dtype=torch.float32)
                normalizer.stds = torch.tensor(stats["std"], dtype=torch.float32).clamp(min=1e-8)
                self.normalizers[key] = normalizer

    @torch.no_grad()
    def normalize_obs(self, obs):
        normalized = []
        for key in OBS_KEYS:
            tensor = obs[key].flatten().to(self.device).float()
            normalized.append(self.normalizers[key].normalize(tensor))
        return torch.cat(normalized, dim=-1)

    @torch.no_grad()
    def get_action(self, obs_seq):
        nobs = self.normalize_obs(obs_seq)
        return nobs  # TODO: self.diffusion.get_action(nobs, dagger=True, return_dict=True)

    def __call__(self, obs):
        obs_horizon = 1
        obs_dict = {
            "block_translation":     torch.tensor(obs[0:2], dtype=torch.float32),
            "block_orientation":     torch.tensor(obs[2:3], dtype=torch.float32),
            "ee_translation":        torch.tensor(obs[3:5], dtype=torch.float32),
            "ee_target_translation": torch.tensor(obs[5:7], dtype=torch.float32),
        }
        obs_deque = deque([obs_dict] * obs_horizon, maxlen=obs_horizon)
        obs_seq = {
            key: torch.stack([obs_deque[i][key] for i in range(obs_horizon)])
            .swapaxes(0, 1).float()
            for key in obs_dict.keys()
        }
        return self.get_action(obs_seq)