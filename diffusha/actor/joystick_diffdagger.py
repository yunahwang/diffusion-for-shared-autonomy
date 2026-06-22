#!/usr/bin/env python3
"""Adopted from https://github.com/cbschaff/rsa/blob/master/lunar_lander/joystick_agent.py"""
# resources - https://www.pygame.org/docs/ref/joystick.html
# look under xbox 360 controller (pygame 2.x)

# import pygame

import numpy as np
import pandas as pd

import torch

import imageio

import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg') # use this for realtime plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

import pygame

import json
from pathlib import Path
import os
import time

from statsmodels.distributions.empirical_distribution import ECDF
import pickle

import warnings
warnings.filterwarnings("ignore")

from diffusha.actor import Actor
from diffusha.actor.assistive import DiffusionAssistedActor
from diffusha.diffusion.evaluation.helper import prepare_diffusha

# from diffusha.diffdagger.diffdagger_loss import DaggerLoss
from diffusha.actor.loss_dist_ood import noise_estimation_loss_nb_infer

import sys
sys.path.append(str(Path(__file__).parents[2] / "DiffDAgger"))
from diffdagger.util.cdf import CDF

#####################################
# Change these to match your joystick
UP_AXIS = 4  # AKA ；up(negative) and down(positive)
SIDE_AXIS = 3  # AKA ；left and right
#####################################
np.set_printoptions(precision=12, suppress=False)

class JoystickDiffDaggerActor(Actor):
    """Joystick Controller for Block Pushing."""

    def __init__(self, env, fps=50):
        """Init."""
        self.env = env
        self.human_agent_action = np.array([0., 0.], dtype=np.float32)
        pygame.init()
        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x)
                     for x in range(pygame.joystick.get_count())]
        #print(joysticks)
        # if len(joysticks) != 1:
        #     raise ValueError("There must be exactly 1 joystick connected."
        #                      f"Found {len(joysticks)}")
        self.joy = joysticks[-1]  # TEMP
        self.joy.init()
        #pygame.init()
        self.t = None
        self.fps = fps
    
    def _get_human_action(self):
        events = pygame.event.get()

        DEADZONE = 0.1

        for event in events:
            if event.type == pygame.JOYAXISMOTION:
                v = 0.0 if abs(event.value) < DEADZONE else event.value

                if event.axis == UP_AXIS:
                    # print("up/down")
                    self.human_agent_action[1] = -1 * v

                elif event.axis == SIDE_AXIS:
                    # print("left/right")
                    self.human_agent_action[0] = v

        return self.human_agent_action


    def act(self, ob):
        """Act."""
        # self.env.render()
        action = self._get_human_action()
        return action

    def reset(self):
        self.human_agent_action[:] = 0.

def get_effector_xy_from_obs(ob):
    """
    Works for BlockPush-style obs dicts.
    Returns 2D end-effector position.
    """
    #print("ob, ", ob)

    return [ob[3], ob[4]]

def compute_linear_gamma(
    loss: float,
    gamma_min: float = 0.0,
    gamma_max: float = 1.0,
    loss_cap: float = 2.0,
) -> float:
    t = float(np.clip(loss / loss_cap, 0.0, 1.0))
    return gamma_max - t * (gamma_max - gamma_min)


def compute_sigmoid_gamma(
    loss: float,
    gamma_min: float = 0.0,
    gamma_max: float = 1.0,
    sigma_med: float = 1.0,
    sigma_scale: float = 1.0,
) -> float:
    z = (loss - sigma_med) / sigma_scale
    s = 1.0 / (1.0 + np.exp(z))
    return gamma_min + s * (gamma_max - gamma_min)

def build_cdfs():
    """Build CDF objects from the reference CSV files."""
    state_csv_path = Path(__file__).parents[1] / "data_collection" / "state_losses_output_5_sampling.csv"
    ood_csv_path   = Path(__file__).parents[1] / "data_collection" / "flipped(ood)_vs_2023_100_1824.csv"

    # average across action sample columns → one loss scalar per state
    state_ref_losses  = pd.read_csv(state_csv_path).mean(axis=1).values
    action_ref_losses = pd.read_csv(ood_csv_path).mean(axis=1).values

    return CDF(state_ref_losses), CDF(action_ref_losses)

def get_all_values():
    state_csv_path = Path(__file__).parents[1] / "data_collection" / "state_losses_output_5_sampling.csv"
    action_csv_path   = Path(__file__).parents[1] / "data_collection" / "flipped(ood)_vs_2023_100_1824.csv"
    action_theta_pkl_path = Path(__file__).parents[1] / "data_collection" / "action_angles" / "action_target_2023_angle.pkl"

    with open(action_theta_pkl_path, "rb") as f:
        theta_data = pickle.load(f)

    action_theta = []
    for csv_dict in theta_data.values():
        for ep_dict in csv_dict.values():
            action_theta.extend(ep_dict["start_to_cur_thetas"])
    action_theta = np.array(action_theta)

    state_ref_losses  = pd.read_csv(state_csv_path).mean(axis=1).values
    action_ref_losses = pd.read_csv(action_csv_path).mean(axis=1).values

    return state_ref_losses, action_ref_losses, action_theta

def get_mult_bounds():
    state_ref_losses, action_ref_losses, action_theta = get_all_values()

    state_percentiles  = CDF(state_ref_losses)
    action_percentiles = CDF(action_ref_losses)
    ecdf_angles        = CDF(action_theta)

    # can't multiply directly — different lengths
    # instead get the min/max of each percentile range and multiply those
    mult_lo = state_percentiles.min * (ecdf_angles.min * action_percentiles.min)
    mult_hi = state_percentiles.max * (ecdf_angles.max * action_percentiles.max)

    return mult_lo, mult_hi


if __name__ == '__main__':
    from diffusha.data_collection.env import make_env

    no_assist = False # if False, use DiffusionAssistedActor

    # NOTE - change the following lines
    # TODO - may make it into an argument
    draw_trajs = False
    save_trajs = False
    save_csvs = True

    shape = "linear" # other option: "sigmoid"
    shape_save = True

    fwd_diff_ratio = 0.0 

    env_name =  "BlockPushMultimodal-v1"

    env = make_env(
        env_name,
        seed=1,
        test=False
    )

    actor = JoystickDiffDaggerActor(env)

    # get per-point pickle value for action ID
    pkl_path = Path(__file__).parents[2] / "diffusha" / "data_collection" / "action_angles" / "action_target_2023_angle.pkl"
    with open(pkl_path, "rb") as file:
        loaded_dict = pickle.load(file)

    all_per_point_angles = [] # this is across all episodes and csvs
    for csv_dict_key_name, ep_dict in loaded_dict.items():
        for ep_dict_key_name, sample in ep_dict.items():
            per_point_thetas = sample["per_point_thetas"]
            all_per_point_angles.extend(per_point_thetas)

    all_start_current_angles = [] # also across all episodes and csvs
    for csv_dict_key_name, ep_dict in loaded_dict.items():
        for ep_dict_key_name, sample in ep_dict.items():
            start_current_thetas = sample["start_to_cur_thetas"]
            all_start_current_angles.extend(start_current_thetas)

    #ecdf_action_angles = ECDF(all_per_point_angles)
    ecdf_action_angles = ECDF(all_start_current_angles)
    print("ecdf instance created, ", ecdf_action_angles)

    mult_lo, mult_hi = get_mult_bounds()
    print("mult_lo, mult_hi, ", mult_lo, mult_hi)

    if no_assist:
        for _ in range(10000):
            ob = env.reset()
            env.render()
            done = False
            reward = 0.0

            while not done:
                env.render()
                ob, r, done, _ = env.step(actor.act(ob))
                reward += r
            print(reward)

    else:

        obs_space = env.observation_space
        act_space = env.action_space
        print("obs_space, ", obs_space)
        print("act_space, ", act_space)

        with open(Path(__file__).parents[1] / "diffusion" / "evaluation" / "configs.json", "r") as f:
            env2config = json.load(f)

        model_dir = Path(__file__).parents[2] / "2023_100_ckpt"

        # NOTE: change here 
        raw_to_which_side = "in_dist_left" # because left is baseline
        trial_num = "0622_mon_theta_anal_16_pat_on_increase"

        subdir_path = model_dir / str(fwd_diff_ratio) / raw_to_which_side / ("trial_" + str(trial_num))


        if save_csvs:
            os.makedirs(subdir_path, exist_ok=True)

            data_subdir_path = subdir_path / "data"
            os.makedirs(data_subdir_path, exist_ok=True)

        # columns of the csv is as follows: episode, which_side, total step, reward, loss, diff, raw, expert, gamma 

        laggy_actor_repeat_prob = 0; noisy_actor_eps = 0

        diffusion = prepare_diffusha(
            env, 
            env2config[env_name], 
            model_dir,
            29999,
            env_name,
            1.0,
            #fwd_diff_ratio,
            laggy_actor_repeat_prob,
            noisy_actor_eps
        )

        print(diffusion, flush=True)

        # load assisted actor
        assisted_actor = DiffusionAssistedActor(
            obs_space = obs_space,
            act_space = act_space,
            diffusion = diffusion,
            behavioral_actor = None,
            #fwd_diff_ratio = fwd_diff_ratio
            fwd_diff_ratio=1.0
        )   

        print(assisted_actor)

        raw_line_id = None
        assisted_line_id = None
        text_id = None

        plt.ion()
        fig = plt.figure(figsize=(12, 6))
        
        ax = fig.add_subplot(1,2,1)
        ax_loss = fig.add_subplot(1,2,2)
        # plt.show(block = False)

        # csv logging containers
        # columns of the csv is as follows: 
        # episode, which_side, total step, reward, loss, diff, raw, expert, gamma 

        eps = []; steps_accum = []; rewards = []; diffs = []; gammas = []; state_losses = []; action_losses = []
        # raw_input_action_x = []; raw_input_action_y = []
        assisted_action_x = []; assisted_action_y = []
        which_side = []
        raw_x_log = []; raw_y_log = []

        obs_block_x = []; obs_block_y = []; obs_block_ori = []
        obs_ee_x = []; obs_ee_y = []
        obs_ee_target_x = []; obs_ee_target_y = []

        # # save as gif
        # frames = []

        ax_loss_r = None
        episode_losses=[]

        current_fwd_diff_ratio = 0.0 #NOTE: can tune
        state_cdf, action_cdf = build_cdfs()

        # NOTE - new
        ecdf_scores = []


        # load human demonstrator
        for ep in range(1, 4):
            # NOTE: change this number

            raw_input_action_x = []; raw_input_action_y = []

            recov_pat= 0

            if draw_trajs:
                #gif_name = time.strftime(time_rn)+".gif" # just change to mp4 later through online converter
                gif_name = "episode_" + str(ep) + ".gif"
                gif_full_path = subdir_path / gif_name

            if save_csvs:
                csv_name = "episode_" + str(ep) + ".csv"
                csv_full_path = subdir_path / csv_name
                

            ob = env.reset()
            
            done = False
            reward = 0.0
            step_i = 0

            ee_log = []
            raw_log      = []
            assisted_log = []
            loss_log_state = []
            loss_log_action = []
            ob_log = []
            ob_action_log = []

            frames = []

            traj_ids     = {"raw": [], "assisted": []}

            patience_count = 0

            #plt.show(block = False)

            while not done:
                env.render()

                ob_log.append(ob.copy())
                #print("ob", len(ob))

                ee_xy = get_effector_xy_from_obs(ob)
                ee_log.append(ee_xy)

                raw_action = actor.act(ob)
                print("[before] raw action, ", raw_action, flush=True)

                # 7th column: raw action - take [1]
                raw_input_action_x.append(raw_action.copy()[0])
                raw_input_action_y.append(raw_action.copy()[1])
                raw_x_log.append(raw_action.copy()[0])
                raw_y_log.append(raw_action.copy()[1])

                ob_action_log.append(np.concatenate([ob[:7], raw_action]))

                obs_block_x.append(ob[0])
                obs_block_y.append(ob[1])
                obs_block_ori.append(ob[2])
                obs_ee_x.append(ob[3])
                obs_ee_y.append(ob[4])
                obs_ee_target_x.append(ob[5])
                obs_ee_target_y.append(ob[6])

                """
                Diffdagger highlight!! + state ood loss first!
                """
                # Call diffdagger loss
                # Build x_0_single from current ob and raw_action
                # ob_tensor     = torch.tensor(ob.copy(), dtype=torch.float32).unsqueeze(0)   # (1, 7)
                # #action_tensor = torch.tensor(raw_action,  dtype=torch.float32).unsqueeze(0) # (1, 2)
                # x_0_single    = torch.cat([ob_tensor, action_tensor], dim=-1)               # (1, 9)
                ob_np = np.array(ob[:7], dtype=np.float32)
                raw_action_np = np.array(raw_action, dtype=np.float32)
                
                sampled_losses = []
                for _ in range(5):
                    sampled_action, _ = assisted_actor.act_without_env(ob_np, raw_action_np, report_diff=True)
                    #print("sampled_action, ", sampled_action)
                    state_for_loss = np.concatenate([ob_np, sampled_action])
                    x_0_single = torch.tensor(state_for_loss, dtype = torch.float32).unsqueeze(0)
                    sampled_losses.append(noise_estimation_loss_nb_infer(diffusion, x_0_single, obs_size=7, Nb=512))

                state_loss = float(np.mean(sampled_losses))

                # nb_loss = noise_estimation_loss_nb_infer(diffusion, x_0_single, obs_size=7, Nb=512)
                # print("state_loss", state_loss)
                loss_log_state.append(state_loss)
                state_losses.append(state_loss)

                """
                Diffdagger mod part 2 - action ood loss
                """
                ob_tensor = torch.tensor(ob, dtype = torch.float32).unsqueeze(0)
                raw_action_tensor = torch.tensor(raw_action, dtype = torch.float32).unsqueeze(0)
                x_0 = torch.cat([ob_tensor, raw_action_tensor], dim = -1)
                action_loss = noise_estimation_loss_nb_infer(diffusion, x_0, obs_size=7, Nb=512)
                # print("action_loss, ", action_loss)
                loss_log_action.append(action_loss)
                action_losses.append(action_loss)

                # NOTE - new June 19th
                """
                get angles of raw_actions
                """

                if len(raw_input_action_x) >= 0: # could be 2
                    start_x, start_y = raw_input_action_x[0], raw_input_action_y[0]
                    #prev_x, prev_y = raw_input_action_x[-2], raw_input_action_y[-2]
                    
                    # implementing per point to see if it's obvious at any given timepoint if it's ID vs OOD
                    # dx = raw_action[0] - prev_x
                    # dy = raw_action[1] - prev_y

                    # implementing the start to current angle logic
                    dx = raw_action[0] - start_x
                    dy = raw_action[1] - start_y

                    theta = np.array(np.arctan2(dx, dy))
                    ecdf_angle = ecdf_action_angles(theta)
                    print(f"theta, {theta}, ecdf_score, {ecdf_angle}")
                    ecdf_scores.append(ecdf_angle)

                    """
                    GET QUANTILE VALUES from each state and action loss
                    """
                    state_percentile  = float(state_cdf(state_loss))
                    action_percentile = float(action_cdf(action_loss))

                    print(f"state_loss:  {state_loss:.4f}  → CDF percentile: {state_percentile:.3f}")
                    print(f"action_loss: {action_loss:.4f}  → CDF percentile: {action_percentile:.3f}")

                    # """
                    # Compute correct gamma by multiplying quantile values and mapping linearly to the gamma values
                    # """
                    # percentile_mult = state_percentile * action_percentile

                    # # NOTE. ver1 - vanilla. without any patience term so instant reaction. still doesn't accept ood-direction behavior very well
                    # #current_fwd_diff_ratio = float(np.clip(1.0 - percentile_mult, 0.0, 1.0))
                    # #print(f"percentile_mult: {percentile_mult:.3f} → next fwd_diff_ratio: {current_fwd_diff_ratio:.3f}")

                    # # NOTE. ver2 - both losses over some threshold, patience of two steps in a row, ratio either 0.0 or 1.0
                    # PATIENCE_THRESHOLD = 2
                    # STATE_THRESH = 0.7
                    # ACTION_THRESH = 0.7

                    # both_ood = (state_percentile >= STATE_THRESH) and (action_percentile >= ACTION_THRESH)

                    # if both_ood:
                    #     patience_count += 1
                    #     if patience_count >= PATIENCE_THRESHOLD:
                    #         current_fwd_diff_ratio = 0.0
                    # else:
                    #     patience_count = 0  # reset if either drops back below threshold
                    #     current_fwd_diff_ratio = 1.0

                    # print(f"patience: {patience_count}, fwd_diff_ratio: {current_fwd_diff_ratio:.2f}")

                    # # NOTE. ver3 - state_ood * (angle_ecdf * action_ood)
                    mult = state_percentile * (ecdf_angle * action_percentile)
                    # current_fwd_diff_ratio = float(np.clip(1.0 - mult, 0.0, 1.0))
                    # print(f"action adjusted: {ecdf_angle * action_percentile:.3f}, mult: {mult:.3f} → next gamma: {current_fwd_diff_ratio:.3f}")
                    # print("*************")

                    # NOTE. ver 4 - same as ver3 above + exponential moving average but fast dropping so that more responsive to OOD situation
                    gamma_prev = current_fwd_diff_ratio
                    # # gamma_raw = float(np.clip(1.0 - mult, 0.0, 0.8))
                    # POWER = 0.5
                    # mult_stretched = mult ** POWER
                    # gamma_raw = float(np.clip(1.0 - mult_stretched, 0.0, 1.0))
                    # if gamma_raw < gamma_prev: 
                    #     # signifying entering OOD state
                    #     ALPHA = 1.0
                    # else:
                    #     ALPHA = 0.1
                    # gamma_smooth = ALPHA * gamma_raw + (1 - ALPHA) * gamma_prev
                    # current_fwd_diff_ratio = gamma_smooth
                    # print(f"mult raw: {mult:.3f} → mult power: {mult_stretched:.3f} → next gamma: {current_fwd_diff_ratio:.3f}")
                    # print("*************")

                    # NOTE. NO ver 5 - similar to ver 4 but scale mult before passing on to computing value
                    # mult = state_percentile * (ecdf_angle * action_percentile)

                    # mult_scaled = (mult - mult_lo) / (mult_hi - mult_lo)
                    # mult_scaled = float(np.clip(mult_scaled, 0.0, 1.0))

                    # gamma_prev = current_fwd_diff_ratio
                    # gamma_raw = 1.0 - mult_scaled

                    # if gamma_raw < gamma_prev:
                    #     ALPHA = 1.0
                    # else:
                    #     ALPHA = 0.1

                    # gamma_smooth = ALPHA * gamma_raw + (1 - ALPHA) * gamma_prev
                    # current_fwd_diff_ratio = gamma_smooth
                    # print(f"mult: {mult:.3f}, scaled: {mult_scaled:.3f}, raw: {gamma_raw:.3f} → next gamma: {current_fwd_diff_ratio:.3f}")
                    # print("*************")

                    # NOTE. ver6 - sigmoid
                    # sigmoid: values above center get pushed toward 1, below get pushed toward 0
                    CENTER = 0.5  # the midpoint — above this gets amplified, below gets suppressed
                    STEEPNESS = 12  # higher = sharper separation, try 6-12

                    mult_sigmoid = 1.0 / (1.0 + np.exp(-STEEPNESS * (mult - CENTER)))
                    gamma_raw = float(np.clip(1.0 - mult_sigmoid, 0.0, 1.0))

                    if gamma_raw < gamma_prev:
                        # dropping toward OOD — always allow instantly
                        ALPHA = 1.0
                        recov_pat = 0  # reset patience
                    else:
                        # potentially recovering toward ID
                        if gamma_raw > 0.2:
                            recov_pat += 1
                        else:
                            recov_pat = 0

                        if recov_pat >= 5:
                            ALPHA = 0.1  # allow slow recovery
                        else:
                            ALPHA = 0.0  # hold current gamma, don't recover yet

                    gamma_smooth = ALPHA * gamma_raw + (1 - ALPHA) * gamma_prev
                    current_fwd_diff_ratio = gamma_smooth
                    print(f"mult: {mult:.3f}, sig: {gamma_raw:.3f}, patience: {recov_pat}, ratio: {current_fwd_diff_ratio:.3f}")
                    print("******")

                # use last step's ratio for this step's action
                assisted_actor.fwd_diff_ratio = current_fwd_diff_ratio
                assisted_action, diff = assisted_actor.act_without_env(ob, raw_action, report_diff=True)
                #print("assisted_action, ", assisted_action, flush=True)

                # 6th column: diffs
                diffs.append(diff)

                raw_log.append((ee_xy, raw_action.copy())) 
                

                assisted_log.append((ee_xy, assisted_action.copy())) 
                
                # 8th column: assisted action - take [1]
                assisted_action_x.append(assisted_action.copy()[0])
                assisted_action_y.append(assisted_action.copy()[1])

                if draw_trajs:
                    ax.clear()

                    ax_loss.clear()

                    ####################################
                    if ax_loss_r is not None:
                        ax_loss_r.remove()

                    ax_loss.plot(loss_log_state, 'purple', linewidth=2)
                    ax_loss.set_xlim(0,100)
                    ax_loss.set_title('noise estimation loss')
                    #ax_loss.set_xlabel('step')
                    ax_loss.set_ylabel('loss')

                    # Quantile reference lines
                    quantiles = {
                        # NOTE. ACTION ood
                        # 'p25': 0.0204,
                        # 'p50': 0.0378,
                        # 'p75': 0.0744,
                        # 'p99': 0.3233,
                        'p75': 0.0964, 
                        'p99': 0.1747
                    }
                    for label, val in quantiles.items():
                        ax_loss.axhline(y=val, color='blue', linestyle=':', linewidth=0.8, label=f'train {label}={val:.4f}')
                        # ax_loss.text(len(loss_log) * 0.01, val, label, color='blue', fontsize=7, va='bottom')

                    ax_loss_r = ax_loss.twinx()
                    ax_loss_r.set_ylim(ax_loss.get_ylim())
                    ax_loss_r.set_yticks(list(quantiles.values()))
                    #ax_loss_r.set_yticklabels(list(quantiles.keys()), color='blue', fontsize=6)
                    ax_loss_r.tick_params(axis='y', length=0)

                    ax_loss.legend(fontsize=7, loc='upper right')

                    ####################
                    size = 0.12
                    half = size / 2

                    for tx, ty, bcolor, icolor, label in [
                        ( 0.1, 0.0, 'green', 'green', 'target 1'),
                        (-0.1, 0.0, 'red',   'red',   'target 2'),
                    ]:
                        ax.add_patch(Rectangle(
                            (tx - half, ty - half), size, size,
                            linewidth=2, edgecolor=bcolor, facecolor=icolor, alpha=0.4, label=label))

                    # Real EE path
                    ee_xs = [p[0] for p in ee_log]
                    ee_ys = [p[1] for p in ee_log]
                    ax.plot(ee_xs, ee_ys, 'k--', linewidth=2, label='actual EE')

                    # Arrow at current step: raw intent
                    ax.annotate('', xy=(ee_xs[-1] + raw_action[0], ee_ys[-1] + raw_action[1]),
                                xytext=(ee_xs[-1], ee_ys[-1]),
                                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

                    # Arrow at current step: assisted intent
                    ax.annotate('', xy=(ee_xs[-1] + assisted_action[0], ee_ys[-1] + assisted_action[1]),
                                xytext=(ee_xs[-1], ee_ys[-1]),
                                arrowprops=dict(arrowstyle='->', color='orange', lw=2))

                    ax.legend(handles=[
                        mpatches.Patch(color='black',  label='actual EE'),
                        mpatches.Patch(color='blue',   label='raw action'),
                        mpatches.Patch(color='orange', label='assisted action'),
                        # mpatches.Patch(color='green',  label='target 1'),
                        # mpatches.Patch(color='red',    label='target 2'),
                    ])
                    ax.set_xlim(-0.4, 0.4)
                    ax.set_ylim(-1.0, 0.3)
                    ax.set_aspect('equal')
                    ax.set_title(f'Episode {ep}  Step {step_i}')

                    plt.pause(0.001)

                    fig.canvas.draw()

                    # grab the raw RGBA buffer and convert to image
                    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    frames.append(frame)


                # 1st col: episode
                eps.append(ep)

                # 2nd col: which_side
                which_side.append(np.nan)

                ob, r, done, info = env.step(assisted_action) # NOTE - when 0.0 show raw_action
                #print("done, ", done)
                reward += r
                step_i += 1

                # 3rd col: step_accum
                steps_accum.append(step_i)

                # 4th col: rewards
                rewards.append(reward)

                # last col: gamma
                gammas.append(current_fwd_diff_ratio)
                # print("which_side, ", which_side)

                if current_fwd_diff_ratio < 0.1:
                    time.sleep(0.5) # NOTE - don't do this for gamma 0.0s
                else: 
                    time.sleep(0.3)
            
            #episode_losses.append(loss_log_state.copy())

            print("episode reward: ", reward)

            current_fwd_diff_ratio = 0.0
            if done and info.get('finished', False) or (done and info['state'] in ('target', 'target2')):
                side_val = 1 if info['state'] == 'target' else 0
                #which_side = [side_val] * len(steps_accum)  # backfill all steps
                del which_side[-1]
                which_side.append(side_val)

            # print("which_side, after, ", which_side)

            # save as dataframe, then as csv
            if save_csvs:
                df = pd.DataFrame({
                    "ep": eps,
                    "which_goal": which_side,
                    "steps_accum": steps_accum,
                    "reward": rewards,
                    # "loss": losses, 
                    "diff": diffs,
                    "raw_input_action_x": raw_x_log,
                    "raw_input_action_y": raw_y_log,
                    "assisted_action_x": assisted_action_x,
                    "assisted_action_y": assisted_action_y,
                    "gamma": gammas,
                    "state_losses": state_losses, 
                    "action_losses": action_losses, 
                    "ecdf_score": ecdf_scores

                })
                df.to_csv(csv_full_path, index = False)

                # actual data
                csv_name = "episode_" + str(ep) + ".csv"
                actual_data_full_path = data_subdir_path / csv_name
                # data_df = pd.DataFrame({
                #     "block_x": obs_block_x,
                #     "block_y": obs_block_y,
                #     "block_ori": obs_block_ori,
                #     "ee_x": obs_ee_x,
                #     "ee_y": obs_ee_y,
                #     "ee_tgt_x": obs_ee_target_x,
                #     "ee_tgt_y": obs_ee_target_y,
                #     "raw_action_x": raw_input_action_x,
                #     "raw_action_y": raw_input_action_y
                # })
                # data_df.to_csv(actual_data_full_path, index = False)

            if save_trajs:
                # NOTE
                imageio.mimsave(gif_full_path, frames, fps = 2) # fps = 2 is matching time.sleep(0.5), this is equiv to time.sleep(1) 


