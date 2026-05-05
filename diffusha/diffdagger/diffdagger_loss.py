from collections import deque

import torch

def dagger_loss(obs):
    obs_horizon = 1
    latency = 0
    obs_keys = ["state"] # NOTE: come back to this depending on how normalize_obs uses the key
    obs_dict = {"state": torch.tensor(obs, dtype=torch.float32)}

    obs_deque = deque([obs_dict] * (obs_horizon + latency),
                maxlen=obs_horizon + latency,)
    
    obs_seq = {
        key: torch.stack(
            [obs_deque[i][key] for i in range(obs_horizon)]
        )
        .swapaxes(0, 1)
        .float()
        for key in obs_keys
    }

    return obs_seq