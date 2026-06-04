#!/usr/bin/env python3
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset


OBS_COLS = [
    "block_x", "block_y", "block_ori",
    "ee_x", "ee_y",
    "ee_target_x", "ee_target_y",
]
ACT_COLS = ["action_x", "action_y"]

OBS_HORIZON = 3 # NOTE: Disco does this

# NOTE: BUT, in actual shared autonomy situations seqs will be a lot longer, so this needs tuning
ACTUAL_ACTION_EXECUTION = 2 # NOTE: because our expert action sequences are a lot shorter (len 8 ~ 12)
PREDICTION_HORIZON = 4


class EpisodeCSVBuffer:
    """
    Loads all CSV files from a directory (or list of directories),
    splits them into episodes using the 'episode' column,
    and stores each episode as a (obs, act) numpy array pair.

    Episode boundaries are fully respected — no chunk ever crosses
    two episodes.
    """

    def __init__(
        self,
        data_dirs,
        obs_cols=OBS_COLS,
        act_cols=ACT_COLS,
        action_chunk_size=PREDICTION_HORIZON,
        obs_size=OBS_HORIZON, 
        episode_col="episode",
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]

        self.obs_cols = obs_cols
        self.act_cols = act_cols
        self.Ta = action_chunk_size
        self.To = obs_size
        self.episode_col = episode_col

        # each entry: (obs array, act array) for one episode
        # obs: (N, obs_dim)   act: (N, act_dim)
        self.episodes = []

        csv_files = []
        for d in data_dirs:
            csv_files.extend(sorted(Path(d).rglob("*.csv")))

        assert len(csv_files) > 0, f"No CSV files found in {data_dirs}"

        skipped = 0
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            assert episode_col in df.columns, \
                f"Column '{episode_col}' not found in {csv_path}"
            # print("csv_path, ", csv_path)

            for ep_id, ep_df in df.groupby(episode_col, sort=True):
                # if ep_id % 5 == 0:
                #     break
                ep_df = ep_df.sort_values("step").reset_index(drop=True)
                # print("ep_df, ", ep_df)

                if len(ep_df) <= action_chunk_size:
                    skipped += 1
                    continue

                obs  = ep_df[obs_cols].values.astype(np.float32)  # (N, obs_dim)
                acts = ep_df[act_cols].values.astype(np.float32)  # (N, act_dim)
                self.episodes.append((obs, acts, csv_path))

            # break

        assert len(self.episodes) > 0, "No valid episodes found after filtering!"

        # precompute all valid (episode_idx, t) start indices
        self.index = [
            (ep_idx, t, csv_path)
            for ep_idx, (obs, _, csv_path) in enumerate(self.episodes)
            for t in range(len(obs) - action_chunk_size)
        ]

        print(
            f"Loaded {len(self.episodes)} episodes from {len(csv_files)} CSV files "
            f"({skipped} episodes skipped — too short)\n"
            f"Total valid chunk samples: {len(self.index)}"
        )

    def sample(self):
        """Sample one (obs_t-To+1..obs_t, act_t..act_t+Ta-1) flat vector at random."""
        ep_idx, t, csv_path = random.choice(self.index)
        return self._get(ep_idx, t, csv_path)

    def sample_consecutive(self, n):
        """Sample n consecutive chunk vectors starting from a random valid t."""
        valid = [
            (ep_idx, t)
            for ep_idx, t in self.index
            if t + n <= len(self.episodes[ep_idx][0]) - self.Ta
        ]
        ep_idx, t_start = random.choice(valid)
        return [self._get(ep_idx, t_start + i) for i in range(n)]

    def _get(self, ep_idx, t, csv_path):
        obs, acts, _ = self.episodes[ep_idx]
        
        # collect [O_{t-2}, O_{t-1}, O_t], zero-pad if t < 2
        obs_window = []
        for i in range(t - 2, t + 1):  # i = t-2, t-1, t
            if i < 0:
                obs_window.append(np.zeros(obs.shape[1], dtype=np.float32))
            else:
                obs_window.append(obs[i])
        obs_chunk = np.concatenate(obs_window, axis=0)  # (3 * obs_dim,)

        # action chunk starting at t, zero-pad if near end of episode
        available = acts[t:]
        shortage  = self.Ta - len(available)
        
        if shortage > 0:
            # pad with zeros — episode is over, no action
            pad = np.zeros((shortage, self.act_dim), dtype=np.float32)
            act_chunk = np.concatenate([available, pad], axis=0)
        else:
            act_chunk = available[:self.Ta]
        
        return np.concatenate([obs_chunk, act_chunk.flatten()]), csv_path

    @property
    def obs_dim(self):
        return self.episodes[0][0].shape[1]

    @property
    def act_dim(self):
        return self.episodes[0][1].shape[1]


class ChunkDataset(IterableDataset):
    """
    IterableDataset wrapper around EpisodeCSVBuffer.
    Yields flat vectors: [obs_t | act_t, ..., act_t+Ta-1]
    """

    def __init__(self, buffer):
        super().__init__()
        self.buffer = buffer

    def __iter__(self):
        idx = self.buffer.index.copy()
        random.shuffle(idx)
        for ep_idx, t, csv_path in idx:
            yield self.buffer._get(ep_idx, t, csv_path)


class MultiDirChunkDataset(IterableDataset):
    """
    Same as ChunkDataset but accepts multiple data directories
    and balances sampling equally across them.
    """

    def __init__(
        self,
        data_dirs,
        obs_cols=OBS_COLS,
        act_cols=ACT_COLS,
        action_chunk_size=4,
    ):
        super().__init__()
        self.buffers = [
            EpisodeCSVBuffer(
                data_dirs=d,
                obs_cols=obs_cols,
                act_cols=act_cols,
                action_chunk_size=action_chunk_size,
            )
            for d in data_dirs
        ]

    def __iter__(self):
        combined = [
            (buf_idx, ep_idx, t)
            for buf_idx, buf in enumerate(self.buffers)
            for ep_idx, t in buf.index
        ]
        random.shuffle(combined)
        for buf_idx, ep_idx, t in combined:
            yield self.buffers[buf_idx]._get(ep_idx, t)


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    # ── instantiate ───────────────────────────────────────────────────────────
    buffer = EpisodeCSVBuffer(
        data_dirs=data_dir,
        obs_cols=OBS_COLS,
        act_cols=ACT_COLS,
        action_chunk_size=4,
    )

    print(f"\nobs_dim : {buffer.obs_dim}")
    print(f"act_dim : {buffer.act_dim}")
    print(f"episodes: {len(buffer.episodes)}")
    print(f"index   : {len(buffer.index)} valid (ep_idx, t) pairs")

    # ── test _get at a few specific positions ─────────────────────────────────
    print("\n--- get tests ---")
    for ep_idx, t, csv_path in buffer.index[:5]:
        sample, path = buffer._get(ep_idx, t, csv_path)
        obss = sample[:buffer.To * buffer.obs_dim].reshape(buffer.To, buffer.obs_dim)
        acts = sample[buffer.To * buffer.obs_dim:].reshape(buffer.Ta, buffer.act_dim)
        print(f"ep={ep_idx:3d}  t={t:3d}  obss=\n{obss}\nacts=\n{acts}")
        print(f"  from: {path}")

    # ── test padding: force a sample near episode end ─────────────────────────
    print("\n--- padding test (last timestep of first episode) ---")
    ep_idx = 0
    obs_arr, acts_arr, csv_path = buffer.episodes[0]
    t_last = len(obs_arr) - 1
    sample, _ = buffer._get(ep_idx, t_last, csv_path)
    acts = sample[buffer.To * buffer.obs_dim:].reshape(buffer.Ta, buffer.act_dim)
    print(f"t={t_last}, act chunk (last rows should be zeros):\n{acts}")
    print(f"csv_path={csv_path}")

    # ── test random sample ────────────────────────────────────────────────────
    print("\n--- random sample() ---")
    sample, path = buffer.sample()
    print(f"shape: {sample.shape}  expected: ({buffer.obs_dim * buffer.To + buffer.act_dim * buffer.Ta},)")
    print(f"values: {sample}")
    print(f"path: {path}")

    # ── test dataset iterator ─────────────────────────────────────────────────
    print("\n--- ChunkDataset iterator (first 3 batches) ---")
    from torch.utils.data import DataLoader
    def collate_sample_only(batch):
        # batch is a list of (sample, path) tuples
        samples = [item[0] for item in batch]
        return torch.tensor(np.stack(samples))

    dataset = ChunkDataset(buffer)
    loader  = DataLoader(dataset, batch_size=8, collate_fn=collate_sample_only)
    for i, batch in enumerate(loader):
        print(f"batch {i}: shape={batch.shape}  dtype={batch.dtype}")
        print(batch)
        if i == 2:
            break