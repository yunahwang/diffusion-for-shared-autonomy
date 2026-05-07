from collections import deque
import torch
from pathlib import Path
import pickle
from .util import SafeLimitsNormalizer, GaussianNormalizer
from .cdf import CDF
from ..diffusion.utils import extract

OBS_KEYS = ["block_translation", "block_orientation", "ee_translation", "ee_target_translation", "action"]

class DaggerLoss:
    def __init__(self, diffusion, assisted_actor, mode="limits", alpha=0.95, patience = 2):
        self.diffusion = diffusion
        self.device = "cpu"
        self.assisted_actor = assisted_actor
        self.mode = mode  # "limits" or "z_score"

        self.patience = patience
        self.patience_window = patience

        self.alpha = alpha

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

        cdf_path = Path(__file__).parents[2] / "diffusion_loss_cdf.pkl"
        with open(cdf_path, "rb") as f:
            cdf_data = pickle.load(f)
        self.diffusion_loss_cdf = CDF(cdf_data["losses"].tolist())
        self.diffusion_loss_threshold = self.diffusion_loss_cdf.get_quantile(self.alpha)
        
        self.reset()

    def reset(self):
        self.deque = deque([], maxlen=self.patience_window)


    @torch.no_grad()
    def normalize_action(self, obs_seq, obs, raw_action):
        # _diffusion_cond_sample returns a numpy array of shape (action_dim,)
        action = self.assisted_actor._diffusion_cond_sample(obs, raw_action)
        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device).flatten()
        return self.normalizers["action"].normalize(action_tensor)

    @torch.no_grad()
    def normalize_obs(self, obs_seq):
        normalized = []
        for key in [k for k in OBS_KEYS if k != "action"]:
            print("obs_seq, ", obs_seq)
            tensor = obs_seq[key].flatten().to(self.device).float()
            print("self.normalizers, ", self.normalizers)
            normalized.append(self.normalizers[key].normalize(tensor))
        return torch.cat(normalized, dim=-1)

    def compute_loss(self, obs_batch, action_batch, timesteps, mask_batch=None):
        x_0 = torch.cat([obs_batch, action_batch], dim=-1).to(self.device)
        timesteps = timesteps.to(self.device)
        
        # mirror training exactly — diffuse generates its own noise internally
        x_t, e = self.diffusion.diffuse(x_0, timesteps)
        
        output = self.diffusion.model(x_t, timesteps)
        err = e - output
        
        if mask_batch is not None:
            err = err * mask_batch
        
        print("ERR, ", err.square().mean())
        return err.square().mean()

    @torch.no_grad()
    def get_avg_diffusion_loss_ndata(self, nobs, naction, repeat = True):
        #assert nobs.ndim == 7 and naction.ndim == 2

        batch_multiplier = 32
        num_per_batch = 1

        if repeat:
            KLD_LOSS = 0
            nobs_repeat = nobs.repeat(self.diffusion.num_diffusion_steps * batch_multiplier, 1, 1,)
            naction_repeat = naction.repeat(self.diffusion.num_diffusion_steps * batch_multiplier, 1, 1,)
            timesteps = (torch.arange(
                self.diffusion.num_diffusion_steps * batch_multiplier, device = self.device,).long() % self.diffusion.num_diffusion_steps)
            #print("timesteps shape, ", timesteps.shape)
            for _ in range(num_per_batch):
                x_0_repeat = torch.cat([nobs_repeat, naction_repeat], dim=-1)
                noise = torch.empty_like(x_0_repeat).normal_(0, 1)
                KLD_LOSS += self.compute_loss(
                    nobs_repeat, naction_repeat, timesteps
                ).item()
        
        else:
            KLD_LOSS = 0
            for _ in range (num_per_batch):
                noise = torch.empty_like(x_0_repeat).normal_(0, 1)
                timesteps = (torch.arange(nobs.shape[0], device = self.device).long() % self.diffusion.num_diffusion_steps)
                KLD_LOSS += self.compute_loss(nobs, naction, timesteps, noise).item()
        return KLD_LOSS / num_per_batch

    @torch.no_grad()
    def get_naction(self, obs_seq, obs, nobs, raw_action, store_output = False, extra_steps = 0, initial_noise = None):
        #assert nobs.ndim == 7
        
        #return self.assisted_actor._diffusion_cond_sample(obs_seq, raw_action).normal_(0,1)
        return self.normalize_action(obs_seq, obs, raw_action)

    @torch.no_grad()
    def get_action_and_loss(self, obs_seq, obs, raw_action, initial_noise = None, extra_steps = 0,
                   dagger = True, return_dict = False, obs_batch_size = 1):
        

        nobs = self.normalize_obs(obs_seq)
        #assert nobs.ndim == 7
        nactions = self.get_naction(obs_seq, obs, nobs, raw_action, initial_noise = initial_noise, extra_steps = extra_steps)

        assist_action = self.assisted_actor._diffusion_cond_sample(obs, raw_action)

        if not dagger:
            return assist_action
        
        diffusion_loss = self.get_avg_diffusion_loss_ndata(nobs, nactions, repeat = True)

        # TODO - implement patience here - would have to accumulate and then arbitrate changes here

        cdf_value = self.diffusion_loss_cdf(diffusion_loss)
        is_cdf_value_past_alpha = (cdf_value > self.alpha)
        self.deque.append(diffusion_loss > self.diffusion_loss_threshold)


        if return_dict:
            return dict(
                action=assist_action,
                diffusion_loss=diffusion_loss,
                diffusion_loss_threshold=self.diffusion_loss_threshold,
                query=sum(self.deque) >= self.patience,
            )
        else:
            return assist_action, diffusion_loss, self.diffusion_loss_threshold, cdf_value, is_cdf_value_past_alpha


    def __call__(self, obs, raw_action):
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
        return self.get_action_and_loss(obs_seq, obs, raw_action)