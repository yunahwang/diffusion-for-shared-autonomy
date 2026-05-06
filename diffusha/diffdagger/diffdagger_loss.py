from collections import deque
import torch
from pathlib import Path
import pickle
from .util import SafeLimitsNormalizer, GaussianNormalizer

OBS_KEYS = ["block_translation", "block_orientation", "ee_translation", "ee_target_translation"]

class DaggerLoss:
    def __init__(self, diffusion, assisted_actor, mode="limits"):
        self.diffusion = diffusion
        self.device = "cpu"
        self.assisted_actor = assisted_actor
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

    # @torch.no_grad()
    # def get_naction(self, obs_seq, nobs, store_output = False, extra_steps = 0, initial_noise = None):
    #     assert nobs.ndim == 7
        
    #     return self.assisted_actor._diffusion_cond_sample(obs_seq[:7], obs_seq[7:])

    # @torch.no_grad()
    # def get_action(self, obs_seq, initial_noise = None, extra_steps = 0,
    #                dagger = False, return_dict = False, obs_batch_size = 1):
    #     nobs = self.normalize_obs(obs_seq)
    #     assert nobs.ndim == 7
    #     naction = self.get_naction(obs_seq, nobs, initial_noise = initial_noise, extra_steps = extra_steps)

    #     return nobs  # actually: this returns unnormalized actions 

    # def __call__(self, obs):
    #     obs_horizon = 1
    #     obs_dict = {
    #         "block_translation":     torch.tensor(obs[0:2], dtype=torch.float32),
    #         "block_orientation":     torch.tensor(obs[2:3], dtype=torch.float32),
    #         "ee_translation":        torch.tensor(obs[3:5], dtype=torch.float32),
    #         "ee_target_translation": torch.tensor(obs[5:7], dtype=torch.float32),
    #     }
    #     obs_deque = deque([obs_dict] * obs_horizon, maxlen=obs_horizon)
    #     obs_seq = {
    #         key: torch.stack([obs_deque[i][key] for i in range(obs_horizon)])
    #         .swapaxes(0, 1).float()
    #         for key in obs_dict.keys()
    #     }
    #     return self.get_action(obs_seq)

    @torch.no_grad()
    def get_naction(self, obs, user_act):
        # _diffusion_cond_sample handles its own internal processing
        return self.assisted_actor._diffusion_cond_sample(obs, user_act).normal_(0, 1)

    @torch.no_grad()
    def get_action(self, obs, user_act, dagger = False):
        if not dagger:
            return self.assisted_actor._diffusion_cond_sample(obs, user_act)
        
        nactions = self.get_naction(obs, user_act)
        nobs = self.normalize_obs(obs_seq)
        
        diffusion_loss = self.get_avg_diffusion_loss_ndata(nobs, nactions, repeat = True)

    def __call__(self, obs, user_act=None):
        # split raw obs into obs and action parts
        raw_obs = obs[:7]
        # user_act comes from outside (raw_action from joystick)
        return self.get_action(raw_obs, user_act)