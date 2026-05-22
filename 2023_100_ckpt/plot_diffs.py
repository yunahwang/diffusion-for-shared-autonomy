"""
run from the directory - 2023_100_ckpt

Folder layout assumed:
  <root>/
    <gamma>/           e.g. 0.0, 0.2, ...
      <condition>/     e.g. right_ood, left_in_dist, ...
        trial_<N>/
          episode_<M>.csv

Each CSV has columns:
  ep, which_goal, steps_accum, reward, loss, diff,
  raw_input_action_x, raw_input_action_y,
  assisted_action_x, assisted_action_y, gamma
"""
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
# Config — edit here to define which (gamma, condition) pairs to compare
# ---------------------------------------------------------------------------

# NOTE: 
REPLACE_WITH_CLEAN = True

CLEAN_EPISODES = {
    ("0.0", "right_ood") : [
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
    # add more pairs as needed, e.g. ("0.2", "right_ood")
]

# Assign a colour to every (gamma, condition) pair
PAIR_COLORS = {
    ("0.0", "right_ood"):    "steelblue",
    ("0.0", "left_in_dist"): "coral",
    ("0.2", "right_ood"):    "seagreen",
    ("0.2", "left_in_dist"): "mediumpurple",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _replace_with_clean_episodes(root: Path, result: dict):
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

def _pick_best_trial(trial_dirs: list[tuple[int, Path]]):
    """Return the directory with the highest trial number."""
    return max(trial_dirs, key=lambda x: x[0])[1]


# def _pick_last_episode_csv(trial_dir: Path):
#     """Return the CSV with the highest episode number inside trial_dir."""
#     episode_csvs = []
#     for csv in trial_dir.rglob("episode_*.csv"):
#         parts = csv.name.split("_")
#         if len(parts) != 2:  # excludes trial_extra_1, trial_extra_2 etc
#             continue
#         try:
#             trial_num = int(parts[1])
#         except ValueError:
#             continue
#     if not episode_csvs:
#         return None
#     _, best = max(episode_csvs, key=lambda x: x[0])
#     return best

def _pick_last_episode_csv(trial_dir: Path):
    """Return the CSV with the highest episode number inside trial_dir."""
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


def make_dicts(root_folder):
    """
    Walk root_folder and build:

        { (gamma, condition): [df_from_best_csv, ...] }

    For each (gamma, condition) combination the function finds every
    trial_* sub-directory, picks the one with the highest number, then
    picks the CSV with the highest episode number from that trial.

    If you want *all* episode CSVs from the best trial (not just the last
    one), swap _pick_last_episode_csv for a helper that collects all of them.
    """
    root = Path(root_folder)
    result: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)

    # Group trial_* dirs by their (gamma, condition) parent
    trial_dirs_by_pair: dict[tuple[str, str], list[tuple[int, Path]]] = defaultdict(list)

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
        # d is  <root>/<gamma>/<condition>/trial_<N>
        rel_parts = d.relative_to(root).parts  # ('0.0', 'right_ood', 'trial_1')
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
# Plotting helpers
# ---------------------------------------------------------------------------

def _get_color(gamma: str, condition: str):
    return PAIR_COLORS.get((gamma, condition), "gray")


def _label(gamma: str, condition: str):
    return f"γ={gamma} / {condition}"


def _plot_metric(
    result: dict[tuple[str, str], list[pd.DataFrame]],
    metric: str,
    ylabel: str,
    title: str,
    out_name: str,
    comparisons: list[tuple[str, str]],
    quantile_lines: dict[str, float],
):
    """Generic per-episode line plotter for any scalar metric column."""
    pairs_to_plot = comparisons if comparisons is not None else list(result.keys())
    print("pairs_to_plot, ", pairs_to_plot)

    print("metric, ", metric)

    fig, ax = plt.subplots(figsize=(12, 6))

    for gamma, condition in pairs_to_plot:
        dfs = result.get((gamma, condition))
        if not dfs:
            print(f"  [warn] no data for gamma={gamma}, condition={condition}")
            continue
        color = _get_color(gamma, condition)
        label = _label(gamma, condition)
        first = True
        for df in dfs:
            print(df[metric])
            #break
            for ep_id, ep_df in df.groupby("ep"):
                ax.plot(
                    ep_df["steps_accum"],
                    ep_df[metric],
                    color=color,
                    alpha=0.85,
                    linewidth=0.9,
                    label=label if first else None,
                )
                first = False
        #break

    if quantile_lines:
        for qlabel, val in quantile_lines.items():
            ax.axhline(y=val, color="blue", linestyle=":", linewidth=0.8,
                       label=f"train {qlabel}={val:.4f}")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Run")

    ax.set_xlabel("Accumulated steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_name, dpi=150)
    print(f"Saved {out_name}")
    plt.close()


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_diff_by_episode(
    result: dict[tuple[str, str], list[pd.DataFrame]],
    comparisons: list[tuple[str, str]],
):
    _plot_metric(
        result,
        metric="diff",
        ylabel="Diff (raw − expert)",
        title="Raw action – expert diff per episode",
        out_name="diff_by_episode.png",
        comparisons=comparisons,
        quantile_lines=None
    )


def plot_loss_by_episode(
    result: dict[tuple[str, str], list[pd.DataFrame]],
    comparisons: list[tuple[str, str]],
):
    _plot_metric(
        result,
        metric="loss",
        ylabel="Noise estimation loss",
        title="Comparing in-distr vs ood losses",
        out_name="loss_by_train_or_ood.png",
        comparisons=comparisons,
        quantile_lines={"p75": 0.0744, "p99": 0.3233},
    )


def plot_reward_by_episode(
    result: dict[tuple[str, str], list[pd.DataFrame]],
    comparisons: list[tuple[str, str]],
):
    """Cumulative reward curve (the 'reward' column is already cumulative)."""
    _plot_metric(
        result,
        metric="reward",
        ylabel="Cumulative reward",
        title="Reward per episode",
        out_name="reward_by_episode.png",
        comparisons=comparisons,
    )

# ---------------------------------------------------------------------------
# Ribbon plot: mean ± std per condition, per-step t-test, summary t-test
# ---------------------------------------------------------------------------
 
def _collect_episode_series(
    result: dict[tuple[str, str], list[pd.DataFrame]],
    pair: tuple[str, str],
    metric: str,
) :
    """
    For a given (gamma, condition) pair, return:
      - matrix  : shape (n_episodes, n_steps) — one row per episode,
                  aligned on steps 1..max_steps (shorter episodes get NaN-padded)
      - ep_means: shape (n_episodes,) — mean loss per episode (for summary t-test)
 
    Each episode's steps are re-indexed by their *within-episode* step position
    (1st step, 2nd step, ...) so episodes with different accumulated offsets still
    align correctly.
    """
    dfs = result.get(pair, [])
    episodes = []
    for df in dfs:
        for _, ep_df in df.groupby("ep"):
            episodes.append(ep_df[metric].values)
 
    if not episodes:
        return np.array([]), np.array([])
 
    max_len = max(len(e) for e in episodes)
    matrix = np.full((len(episodes), max_len), np.nan)
    for i, ep in enumerate(episodes):
        matrix[i, : len(ep)] = ep
 
    ep_means = np.nanmean(matrix, axis=1)  # one value per episode
    return matrix, ep_means
 
 
def plot_loss_ribbon_with_ttest(
    result: dict[tuple[str, str], list[pd.DataFrame]],
    pair_a: tuple[str, str],
    pair_b: tuple[str, str],
    metric: str = "loss",
    quantile_lines = None,
    out_name: str = "loss_ribbon_ttest.png",
):
    """
    Two-panel figure:
      Top   : mean ± 1 std ribbon per condition, shaded where per-step
               Welch t-test is significant (p < 0.05).
      Bottom: per-episode mean loss as a bar chart, with summary t-test result.
    """
    from scipy import stats
 
    mat_a, ep_means_a = _collect_episode_series(result, pair_a, metric)
    mat_b, ep_means_b = _collect_episode_series(result, pair_b, metric)
 
    if mat_a.size == 0 or mat_b.size == 0:
        print(f"[warn] missing data for {pair_a} or {pair_b}, skipping ribbon plot")
        return
 
    color_a = _get_color(*pair_a)
    color_b = _get_color(*pair_b)
    label_a = _label(*pair_a)
    label_b = _label(*pair_b)
 
    # Align to the shorter episode length
    n_steps = min(mat_a.shape[1], mat_b.shape[1])
    mat_a = mat_a[:, :n_steps]
    mat_b = mat_b[:, :n_steps]
    steps = np.arange(1, n_steps + 1)
 
    mean_a = np.nanmean(mat_a, axis=0)
    std_a  = np.nanstd(mat_a,  axis=0)
    mean_b = np.nanmean(mat_b, axis=0)
    std_b  = np.nanstd(mat_b,  axis=0)
 
    # Per-step Welch t-test
    p_values = np.array([
        stats.ttest_ind(mat_a[:, s], mat_b[:, s], equal_var=False, nan_policy="omit").pvalue
        for s in range(n_steps)
    ])
    significant = p_values < 0.05
 
    # Summary t-test on all per-step, per-episode values
    all_vals_a_flat = mat_a.flatten()
    all_vals_a_flat = all_vals_a_flat[~np.isnan(all_vals_a_flat)]
    all_vals_b_flat = mat_b.flatten()
    all_vals_b_flat = all_vals_b_flat[~np.isnan(all_vals_b_flat)]
    t_stat, p_summary = stats.ttest_ind(all_vals_a_flat, all_vals_b_flat, equal_var=False)

    from scipy.stats import mannwhitneyu
    u_stat, p_mw = mannwhitneyu(all_vals_a_flat, all_vals_b_flat, alternative='two-sided')
 
    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.4, wspace=0.35)
    ax_top = fig.add_subplot(gs[0, :])
    ax_mid = fig.add_subplot(gs[1, 0])
    ax_mid2 = fig.add_subplot(gs[1, 1])

    #ax_top.set_ylim(bottom=0, top=np.nanmax([mean_a + std_a, mean_b + std_b]) * 1.1)
 
    # ── Top: ribbon ─────────────────────────────────────────────────────────
    for mean, std, color, label in [
        (mean_a, std_a, color_a, label_a),
        (mean_b, std_b, color_b, label_b),
    ]:
        ax_top.plot(steps, mean, color=color, linewidth=2, label=label)
        # ax_top.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
        ax_top.fill_between(steps, np.maximum(mean - std, 0), mean + std, color=color, alpha=0.2)
    ax_top.set_ylim(bottom=0, top=np.nanmax([mean_a + std_a, mean_b + std_b]) * 1.1)

    fine = np.arange(0, 2.5, 0.5)        # 0, 0.5, 1.0, 1.5, 2.0
    coarse = np.arange(4, ax_top.get_ylim()[1] + 1, 2)  # 4, 6, 8, ...
    ax_top.set_yticks(np.concatenate([fine, coarse]))
 
    # # Shade steps where the two conditions differ significantly
    # sig_regions = np.where(significant)[0]
    # if sig_regions.size:
    #     # Group contiguous significant steps into spans for neater shading
    #     breaks = np.where(np.diff(sig_regions) > 1)[0] + 1
    #     spans = np.split(sig_regions, breaks)
    #     for span in spans:
    #         ax_top.axvspan(steps[span[0]], steps[span[-1]], color="gold", alpha=0.25,
    #                        label="p<0.05" if span is spans[0] else None)
 
    if quantile_lines:
        for qlabel, val in quantile_lines.items():
            ax_top.axhline(y=val, color="navy", linestyle=":", linewidth=0.8,
                           label=f"train {qlabel}={val:.4f}")
 
    ax_top.set_xlabel("Step within episode")
    ax_top.set_ylabel(f"{metric} (mean ± 1 std)")
    ax_top.set_title(
        f"Per-step mean ± std:  {label_a}  vs  {label_b}"
    )
    handles, labels_ = ax_top.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax_top.legend(by_label.values(), by_label.keys())

    # ── Mid: total loss range ────────────────────────────────────────────────
    # ── Mid: connected means with episode scatter ────────────────────────────
    x_positions = [0, 1]

    for i, (ep_means, pair) in enumerate([(ep_means_a, pair_a), (ep_means_b, pair_b)]):
        color = _get_color(*pair)
        # vertical range line
        ax_mid.vlines(i, np.min(ep_means), np.max(ep_means), color=color, linewidth=2)
        # individual episode dots
        ax_mid.scatter([i] * len(ep_means), ep_means, color=color, s=40, zorder=4, alpha=0.6)
        # mean dot, larger
        ax_mid.scatter(i, np.mean(ep_means), color=color, s=120, zorder=5,
                    marker='D', label=f"{_label(*pair)}  mean={np.mean(ep_means):.3f}")

        # ax_mid.text(1.05, np.max(ep_means), f"max={np.max(ep_means):.3f}", fontsize=8, va='center', clip_on=False)
        # ax_mid.text(1.05, np.min(ep_means), f"min={np.min(ep_means):.3f}", fontsize=8, va='center', clip_on=False)

    # line connecting the two means
    ax_mid.plot(x_positions, [np.mean(ep_means_a), np.mean(ep_means_b)],
                color="gray", linewidth=1.5, linestyle="--", zorder=3)

    ax_mid.set_xticks([0, 1])
    ax_mid.set_xticklabels([_label(*pair_a), _label(*pair_b)])
    ax_mid.set_xlim(-0.4, 1.4)
    ax_mid.set_ylabel(f"Mean {metric} per episode")
    ax_mid.set_ylim(bottom=0)
    ax_mid.legend(fontsize=9)
    ax_mid.set_title(
    f"Per-episode means\n"
    f"Welch t (means): t={t_stat:.3f}, p={p_summary:.4f} → {'YES' if p_summary < 0.05 else 'NO'}")

    # ── Mid2: same as mid but over all steps (not episode means) ────────────
    for i, (all_vals, pair) in enumerate([(all_vals_a_flat, pair_a), (all_vals_b_flat, pair_b)]):
        color = _get_color(*pair)
        ax_mid2.vlines(i, np.min(all_vals), np.max(all_vals), color=color, linewidth=2)
        ax_mid2.scatter([i] * len(all_vals), all_vals, color=color, s=10, zorder=4, alpha=0.3)
        ax_mid2.scatter(i, np.mean(all_vals), color=color, s=120, zorder=5,
                        marker='D', label=f"{_label(*pair)}  mean={np.mean(all_vals):.3f}")
        # ax_mid2.text(1.05, np.max(all_vals), f"max={np.max(all_vals):.3f}", fontsize=8, va='center', clip_on=False)
        # ax_mid2.text(1.05, np.min(all_vals), f"min={np.min(all_vals):.3f}", fontsize=8, va='center', clip_on=False)

    ax_mid2.plot([0, 1], [np.mean(all_vals_a_flat), np.mean(all_vals_b_flat)],
                color="gray", linewidth=1.5, linestyle="--", zorder=3)
    ax_mid2.set_xticks([0, 1])
    ax_mid2.set_xticklabels([_label(*pair_a), _label(*pair_b)])
    ax_mid2.set_xlim(-0.4, 1.4)
    ax_mid2.set_ylabel(f"{metric} (all steps)")
    ax_mid2.set_ylim(bottom=0)
    ax_mid2.legend(fontsize=9)
    #ax_mid2.set_title(f"All steps, all episodes  |  Welch t-test: t={t_stat:.3f}, p={p_summary:.4f}  →  significant: {'YES' if p_summary < 0.05 else 'NO'}")
    ax_mid2.set_title(
    f"All steps, all episodes\n"
    f"Mann-Whitney U (distrib): p={p_mw:.4f} → {'YES' if p_mw < 0.05 else 'NO'}")

    
    plt.tight_layout()
    plt.savefig(out_name, dpi = 150)
    print(f"saved {out_name}")
    plt.close()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root_folder = Path(__file__).parent
    result = make_dicts(root_folder)

    #plot_diff_by_episode(result, comparisons=COMPARISONS)
    plot_loss_by_episode(result, comparisons=COMPARISONS)
    #plot_reward_by_episode(result, comparisons=COMPARISONS)

    if REPLACE_WITH_CLEAN: 
        out_name = "clean_loss_ribbon_ttest.png"
    else:
        out_name = "loss_ribbon_ttest.png"

    plot_loss_ribbon_with_ttest(
    result,
    pair_a=("0.0", "right_ood"),
    pair_b=("0.0", "left_in_dist"),
    metric="loss",
    quantile_lines={"p75": 0.0744, "p99": 0.3233},
    out_name=out_name
)


