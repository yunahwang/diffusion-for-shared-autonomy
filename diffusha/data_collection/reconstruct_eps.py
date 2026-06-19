"""
Offline reward + episode reconstruction from BlockPush pickle chunks.

Column layout (per your CSV):
  0: block_x
  1: block_y
  2: block_rad       (unused by reward)
  3: ee_x
  4: ee_y
  5: ee_target_x     (= target_translation[0], i.e. your user_goal target)
  6: ee_target_y
  7: action_x        (stored inside obs)
  8: action_y
  9: q_val           (always 0 – ignore)

Episode boundary detection:
  The env resets the EE to a fixed home position. In every CSV/pkl we see
  EE_RESET_X = 0.00066170545, EE_RESET_Y = -0.7480884 on the first step of
  each episode. We use that signature to split episodes.

Reward reconstruction mirrors _get_reward() exactly, including all shaping
terms that depend on previous-step values (_prev_dist, _prev_blk2ee,
_prev_blk_pos).

Usage:
    python reconstruct_eps.py \
        --data-dir /path/to/pkl/chunks \
        --user-goal target \
        --out-dir /path/to/dirs 

    # or point at a single CSV for quick testing:
    python reconstruct_eps.py \
        --csv /path/to/file.csv \
        --user-goal target \
        --out rewards_episodes.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── constants ────────────────────────────────────────────────────────────────

# World-frame target positions (from your WORLD_TARGETS + coord_offset 0.4, 0.35)
# The obs stores *reset-coord* values so we need the coord-system the obs lives in.
# From your replay code: ee_x = -row[3], world_xy = obs + coord_offset
# But the obs target cols (5,6) already encode the goal in the same coord frame
# as block_xy, so we read the target directly from the obs. See notes below.

EE_RESET_X = -0.00066170545   # home EE x on episode start #NOTE: make sure this value is negative when doing the opposite
EE_RESET_Y = -0.7480884      # home EE y on episode start
EE_RESET_TOL = 1e-5           # tolerance for float comparison

GOAL_DIST_TOLERANCE = 0.04    # from env source

VALID_GOALS = {"target", "target2", "both"}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_chunk(path: Path) -> np.ndarray:
    """Load one pkl or csv chunk → float32 array (N, 10)."""
    if path.suffix == ".pkl":
        data = torch.load(path, map_location="cpu")
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        return data.astype(np.float32)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, header=0)
        return df.values.astype(np.float32)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def is_episode_start(row: np.ndarray) -> bool:
    """True when the EE is at its home reset position."""
    return (
        abs(row[3] - EE_RESET_X) < EE_RESET_TOL and
        abs(row[4] - EE_RESET_Y) < EE_RESET_TOL
    )


# ── reward reconstruction ─────────────────────────────────────────────────────

def reconstruct_rewards(rows: np.ndarray, user_goal: str) -> list[dict]:
    """
    Replay one episode (list of rows) and compute per-step reward.

    Returns a list of dicts with keys:
        step, block_x, block_y, ee_x, ee_y,
        dist_to_goal, blk2ee,
        reward_shaping, reward_terminal, reward_total,
        done, in_target
    """
    assert len(rows) > 0
    results = []

    # ── initialise stateful accumulators from the FIRST row ──────────────────
    # Mirrors env.__init__ / env.reset() logic:
    #   _ori_dist  = initial dist(block, user_goal_target)
    #   _prev_dist = _ori_dist
    #   _ori_blk2ee = initial dist(block, ee)
    #   _prev_blk2ee = _ori_blk2ee
    #   _prev_blk_pos = initial block pos

    def block_xy(row):  return np.array([row[0], row[1]])
    def ee_xy(row):     return np.array([row[3], row[4]])
    def goal_xy(row):   return np.array([row[5], row[6]])   # user_goal target in obs frame

    r0 = rows[0]
    ori_dist    = np.linalg.norm(block_xy(r0) - goal_xy(r0))
    ori_blk2ee  = np.linalg.norm(block_xy(r0) - ee_xy(r0))

    # guard against degenerate episodes
    if ori_dist < 1e-8:
        ori_dist = 1e-8
    if ori_blk2ee < 1e-8:
        ori_blk2ee = 1e-8

    prev_dist    = ori_dist
    prev_blk2ee  = ori_blk2ee
    prev_blk_pos = block_xy(r0).copy()

    for step, row in enumerate(rows):
        blk  = block_xy(row)
        ee   = ee_xy(row)
        goal = goal_xy(row)

        dist    = np.linalg.norm(blk - goal)
        blk2ee  = np.linalg.norm(blk - ee)

        in_target = dist < GOAL_DIST_TOLERANCE

        # ── replicate _get_reward exactly ────────────────────────────────────
        reward          = 0.0
        reward_shaping  = 0.0

        if in_target:
            # terminal reward
            if user_goal == "both":
                reward = 200.0
            else:
                reward = 200.0   # b0_closest_target == user_goal (only one block tracked)
        else:
            # progress toward goal
            if user_goal == "both":
                reward_shaping += ((prev_dist - dist) / ori_dist) * 100
            else:
                reward_shaping += (max(prev_dist - dist, 0) / ori_dist) * 100

        # EE → block proximity shaping (always applied)
        reward_shaping += ((prev_blk2ee - blk2ee) / ori_blk2ee) * 10

        # block-movement bonus
        reward_shaping += np.linalg.norm(prev_blk_pos - blk) * 10

        reward_total = reward + reward_shaping

        results.append({
            "step":            step,
            "block_x":         blk[0],
            "block_y":         blk[1],
            "block_ori":       row[2],
            "ee_x":            ee[0],
            "ee_y":            ee[1],
            "ee_target_x":     row[5],
            "ee_target_y":     row[6],
            "action_x":        row[7],
            "action_y":        row[8],
            "qval":            row[9],
            "dist_to_goal":    dist,
            "blk2ee":          blk2ee,
            "reward_shaping":  reward_shaping,
            "reward_terminal": reward,
            "reward_total":    reward_total,
            "done":            in_target,
            "in_target":       in_target,
        })

        # update state
        prev_dist    = dist
        prev_blk2ee  = blk2ee
        prev_blk_pos = blk.copy()

        # if in_target:
        #     break   # episode ends here

    return results


# ── episode segmentation ──────────────────────────────────────────────────────

def segment_episodes(data: np.ndarray) -> list[np.ndarray]:
    """
    Split a flat array of rows into per-episode arrays.
    An episode starts whenever the EE is at the home reset position.
    The very first row always starts episode 0.
    """
    boundaries = [i for i, row in enumerate(data) if is_episode_start(row)]
    #print("boundaries, ", boundaries)

    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries

    episodes = []
    for k, start in enumerate(boundaries):
        end = boundaries[k + 1] if k + 1 < len(boundaries) else len(data)
        episodes.append(data[start:end])

    return episodes


# ── main ──────────────────────────────────────────────────────────────────────

def process_files(paths: list[Path], user_goal: str, out_dir: Path):

    out_dir.mkdir(parents = True, exist_ok = True)

    global_ep = 0

    for i, file_path in enumerate(sorted(paths)):

        if i % 10 == 0:
            print(f"going through {i}-th file in total of {len(sorted(paths))} files in directory")

        local_ep = 0
        all_records = []

        print(f"  loading {file_path.name} …")
        data = load_chunk(file_path)

        episodes = segment_episodes(data)
        print(f"    → {len(data)} rows, {len(episodes)} episodes")
        print(f"    → rows in episodes: {sum(len(ep) for ep in episodes)}")  

        for ep_rows in episodes:
            step_records = reconstruct_rewards(ep_rows, user_goal)
            if len(step_records) != len(ep_rows):
                print(f"    ep {local_ep}: {len(ep_rows)} rows in → {len(step_records)} rows out")

        #break
            sum_reward   = sum(r["reward_total"] for r in step_records)
            ep_done      = any(r["done"] for r in step_records)

            for rec in step_records:
                rec["episode"]      = local_ep
                rec["file"]         = file_path.name
                rec["ep_sum_reward"] = sum_reward
                rec["ep_success"]   = ep_done
                all_records.append(rec)

            global_ep += 1
            local_ep += 1

        df = pd.DataFrame(all_records, columns=[
            "episode", "file", "step",
            "block_x", "block_y", "block_ori",
            "ee_x", "ee_y",
            "ee_target_x", "ee_target_y",
            "action_x", "action_y", "qval",
            "dist_to_goal", "blk2ee",
            "reward_shaping", "reward_terminal", "reward_total",
            "done", "in_target", "ep_sum_reward", "ep_success",
        ])

        print(f"length of df {len(df)} rows")
        assert len(df) == len(data)
        

        out_csv = out_dir / (file_path.stem + ".csv")
        df.to_csv(out_csv, index=False)
        print(f"\n✓ saved {len(df)} rows / {local_ep} episodes → {out_csv}")

    # quick summary
    ep_summary = df.groupby("episode").agg(
        file      = ("file", "first"),
        n_steps   = ("step", "count"),
        sum_reward= ("reward_total", "sum"),
        success   = ("ep_success", "first"),
    )
    print("\nEpisode summary (first 20):")
    print(ep_summary.head(20).to_string())
    print(f"\nSuccess rate: {ep_summary['success'].mean():.1%}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct rewards & episodes from BlockPush pkl/csv chunks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-dir", type=Path, help="Directory containing .pkl or .csv chunk files")
    group.add_argument("--csv",      type=Path, help="Single CSV file (for quick testing)")
    parser.add_argument("--user-goal", default="target", choices=list(VALID_GOALS),
                        help="Which target the policy was trained for (default: target)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output folder (will be created if it doesn't exist); "
                             "each input file gets a same-named .csv inside it")
    args = parser.parse_args()

    if args.csv:
        paths = [args.csv]
    else:
        paths = list(args.data_dir.glob("*.pkl")) + list(args.data_dir.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No .pkl or .csv files found in {args.data_dir}")

    print(f"Processing {len(paths)} file(s) with user_goal='{args.user_goal}' …\n")
    process_files(paths, args.user_goal, args.out_dir)


if __name__ == "__main__":
    main()