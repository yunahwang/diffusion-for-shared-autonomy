from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPLACE_WITH_CLEAN = False

CLEAN_EPISODES = {
    ("0.0", "right_ood"): [
        ("trial_1", "episode_8.csv", [2, 8]),
        ("trial_extra_1", "episode_10.csv", [2, 7, 10]),
        ("trial_extra_2", "episode_7.csv", [2, 4, 5, 6, 7])
    ],
    ("0.0", "left_in_dist"): [
        ("trial_5", "episode_10.csv", [1, 3, 4, 5, 6, 7, 8, 10]),
        ("trial_extra_1", "episode_2.csv", [1, 2])
    ]
}

COMPARISONS = [
    ("0.0", "right_ood"),
    ("0.0", "left_in_dist"),
]

PAIR_COLORS = {
    ("0.0", "right_ood"):    "steelblue",
    ("0.0", "left_in_dist"): "coral",
}

# ---------------------------------------------------------------------------
# Joystick data loading (with trial/episode selection)
# ---------------------------------------------------------------------------
def _replace_with_clean_episodes(root, result):
    for (gamma, condition), clean_list in CLEAN_EPISODES.items():
        result[(gamma, condition)] = []
        for trial_dir_name, csv_name, ep_ids in clean_list:
            csv_path = root / gamma / condition / trial_dir_name / csv_name
            if not csv_path.exists():
                print(f"  [warn] clean path not found: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            filtered = df[df["ep"].isin(ep_ids)]
            if filtered.empty:
                print(f"  [warn] no matching ep ids {ep_ids} in {csv_path}")
                continue
            print(f"  [clean] keeping ep {ep_ids} from {csv_path}")
            result[(gamma, condition)].append(filtered)

def _pick_best_trial(trial_dirs):
    return max(trial_dirs, key=lambda x: x[0])[1]

def _pick_last_episode_csv(trial_dir):
    episode_csvs = []
    for csv in trial_dir.glob("episode_*.csv"):
        try:
            ep_num = int(csv.stem.split("_")[1])
            episode_csvs.append((ep_num, csv))
        except (IndexError, ValueError):
            continue
    if not episode_csvs:
        return None
    _, best = max(episode_csvs, key=lambda x: x[0])
    return best

def make_dicts(joystick_root):
    """Load joystick CSVs from 2023_100_ckpt folder with trial/episode selection."""
    root = Path(joystick_root)
    result = defaultdict(list)
    trial_dirs_by_pair = defaultdict(list)

    for d in root.rglob("trial_*"):
        if not d.is_dir():
            continue
        name_parts = d.name.split("_")
        if len(name_parts) != 2:
            continue
        try:
            trial_num = int(name_parts[1])
        except ValueError:
            continue
        rel_parts = d.relative_to(root).parts
        if len(rel_parts) < 3:
            continue
        gamma, condition = rel_parts[0], rel_parts[1]
        trial_dirs_by_pair[(gamma, condition)].append((trial_num, d))

    for (gamma, condition), trials in trial_dirs_by_pair.items():
        best_trial = _pick_best_trial(trials)
        best_csv = _pick_last_episode_csv(best_trial)
        if best_csv is None:
            print(f"  [skip] no episode CSVs in {best_trial}")
            continue
        print(f"Selected: {best_csv}  → gamma={gamma}, condition={condition}")
        df = pd.read_csv(best_csv)
        result[(gamma, condition)].append(df)

    if REPLACE_WITH_CLEAN:
        _replace_with_clean_episodes(root, result)

    return dict(result)

# ---------------------------------------------------------------------------
# Training data loading (flat, no selection)
# ---------------------------------------------------------------------------
def load_training_action_y(training_folder):
    # training_folder = Path(training_folder)
    # print(f"Looking in: {training_folder}")
    """Load action_y from ALL CSVs in training folder recursively."""
    vals = []
    for csv in Path(training_folder).rglob("*.csv"):
        # print("csv, ", csv)
        df = pd.read_csv(csv)
        # print(df)
        try:
            df = pd.read_csv(csv)
            print(df)
            if "action_y" in df.columns:
                vals.extend(df["action_y"].dropna().tolist())
        except Exception as e:
            print(f"  [warn] {csv}: {e}")
        break
    return np.array(vals)

# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
def plot_action_y_histogram(joystick_result, training_folder, out_name="action_y_histogram.png"):
    # --- training: flat load from orig_2023_csv_with_eps ---
    train_vals = load_training_action_y(training_folder)
    print(f"training:  n={len(train_vals)}, mean={train_vals.mean():.4f}, std={train_vals.std():.4f}")

    # --- joystick: from make_dicts result, respecting clean episode selection ---
    joystick_vals = []
    for pair in COMPARISONS:
        for df in joystick_result.get(pair, []):
            if "raw_input_action_y" in df.columns:
                joystick_vals.extend(df["raw_input_action_y"].dropna().tolist())
    joystick_vals = np.array(joystick_vals)
    print(f"joystick:  n={len(joystick_vals)}, mean={joystick_vals.mean():.4f}, std={joystick_vals.std():.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))

    w_t = np.ones(len(train_vals))    / len(train_vals)    * 100
    w_j = np.ones(len(joystick_vals)) / len(joystick_vals) * 100

    ax.hist(train_vals,    bins=80, color="steelblue", alpha=0.6, weights=w_t,
            label=f"training  (n={len(train_vals)}, mean={train_vals.mean():.3f}, std={train_vals.std():.3f})")
    ax.hist(joystick_vals, bins=80, color="coral",     alpha=0.6, weights=w_j,
            label=f"joystick  (n={len(joystick_vals)}, mean={joystick_vals.mean():.3f}, std={joystick_vals.std():.3f})")

    ax.axvline(train_vals.mean(),    color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(joystick_vals.mean(), color="coral",     linestyle="--", linewidth=1.5)

    ax.set_xlabel("action_y")
    ax.set_ylabel("% of samples")
    ax.set_title("action_y distribution: training (orig_2023) vs joystick (clean episodes)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")
    plt.close()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # joystick data — 2023_100_ckpt folder, with trial/episode selection
    joystick_root = Path(__file__).parent  # i.e. 2023_100_ckpt/

    # training data — orig_2023_csv_with_eps, flat load no selection
    training_folder = Path(__file__).parents[1] / "data-dir" / "replay" / "blockpush" / "orig_2023_csv_with_eps"

    joystick_result = make_dicts(joystick_root)

    plot_action_y_histogram(
        joystick_result,
        training_folder,
        out_name="clean_action_y_histogram.png" if REPLACE_WITH_CLEAN else "action_y_histogram.png",
    )