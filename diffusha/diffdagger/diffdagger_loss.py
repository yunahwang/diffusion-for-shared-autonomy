from collections import deque
import torch

class DaggerLoss:
    def __init__(self, diffusion):
        self.diffusion = diffusion  # your loaded diffusion/policy model
        self.device = "cpu"

    @torch.no_grad()
    def normalize_obs(self, obs):
        state = torch.cat([
            obs["block_translation"].flatten(),
            obs["block_orientation"].flatten(),
            obs["ee_translation"].flatten(),
            obs["ee_target_translation"].flatten(),
        ], dim=-1).to(self.device).float()

        mean = state.mean()
        std = state.std().clamp(min=1e-8)
        return (state - mean) / std

    @torch.no_grad()
    def get_action(self, obs_seq):
        nobs = self.normalize_obs(obs_seq)
        
        return nobs

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
        action_dict = self.get_action(obs_seq)
        return action_dict