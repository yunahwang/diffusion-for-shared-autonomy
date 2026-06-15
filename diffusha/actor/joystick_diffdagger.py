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


import warnings
warnings.filterwarnings("ignore")

from diffusha.actor import Actor
from diffusha.actor.assistive import DiffusionAssistedActor
from diffusha.diffusion.evaluation.helper import prepare_diffusha

# from diffusha.diffdagger.diffdagger_loss import DaggerLoss
from diffusha.actor.loss_dist_ood import noise_estimation_loss_nb_infer

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

if __name__ == '__main__':
    from diffusha.data_collection.env import make_env

    no_assist = False # if False, use DiffusionAssistedActor

    # NOTE - change the following lines
    # TODO - may make it into an argument
    draw_trajs = True
    save_trajs = False
    save_csvs = False

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
        raw_to_which_side = "right_ood" # because left is baseline
        trial_num = "0528_thur_state_ood"

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
        raw_input_action_x = []; raw_input_action_y = []
        assisted_action_x = []; assisted_action_y = []
        which_side = []

        obs_block_x = []; obs_block_y = []; obs_block_ori = []
        obs_ee_x = []; obs_ee_y = []
        obs_ee_target_x = []; obs_ee_target_y = []

        # # save as gif
        # frames = []

        ax_loss_r = None
        episode_losses=[]

        # load human demonstrator
        for ep in range(1, 3):
            # NOTE: change this number

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

            #plt.show(block = False)

            while not done:
                env.render()

                ob_log.append(ob.copy())
                #print("ob", len(ob))

                ee_xy = get_effector_xy_from_obs(ob)
                ee_log.append(ee_xy)

                raw_action = actor.act(ob)
                print("[before] raw action, ", raw_action, flush=True)

                ob_action_log.append(np.concatenate([ob[:7], raw_action]))

                # TODO - should now arbitrate between different fwd_diff_ratio, so, based on loss quantiles then should i create new assisted actors everytime
                assisted_action, diff = assisted_actor.act_without_env(ob, raw_action, report_diff=True)
                #print("assisted_action, ", assisted_action, flush=True)

                # 6th column: diffs
                diffs.append(diff)

                raw_log.append((ee_xy, raw_action.copy())) 
                
                # 7th column: raw action - take [1]
                raw_input_action_x.append(raw_action.copy()[0])
                raw_input_action_y.append(raw_action.copy()[1])

                assisted_log.append((ee_xy, assisted_action.copy())) 
                
                # 8th column: assisted action - take [1]
                assisted_action_x.append(assisted_action.copy()[0])
                assisted_action_y.append(assisted_action.copy()[1])

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
                print("state_loss", state_loss)
                loss_log_state.append(state_loss)
                state_losses.append(state_loss)

                """
                Diffdagger mod part 2 - action ood loss
                """
                ob_tensor = torch.tensor(ob, dtype = torch.float32).unsqueeze(0)
                raw_action_tensor = torch.tensor(raw_action, dtype = torch.float32).unsqueeze(0)
                x_0 = torch.cat([ob_tensor, raw_action_tensor], dim = -1)
                action_loss = noise_estimation_loss_nb_infer(diffusion, x_0, obs_size=7, Nb=512)
                print("action_loss, ", action_loss)
                loss_log_action.append(action_loss)
                action_losses.append(action_loss)

                """
                Compute correct gamma as per diffdagger loss
                """
                # FIRST GET CDF OF EACH

                    # TODO: create helper function for plotting losses against gammas

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

                # ob, r, done, _ = env.step(raw_action)
                ob, r, done, info = env.step(raw_action) # NOTE - when 0.0 show raw_action
                #print("done, ", done)
                reward += r
                step_i += 1

                # 3rd col: step_accum
                steps_accum.append(step_i)

                # 4th col: rewards
                rewards.append(reward)

                # last col: gamma
                gammas.append(fwd_diff_ratio)
                # print("which_side, ", which_side)

                time.sleep(0.1) # NOTE - don't do this for gamma 0.0s
            
            #episode_losses.append(loss_log_state.copy())

            print("episode reward: ", reward)
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
                    "raw_input_action_x": raw_input_action_x,
                    "raw_input_action_y": raw_input_action_y,
                    "assisted_action_x": assisted_action_x,
                    "assisted_action_y": assisted_action_y,
                    "gamma": gammas,

                })
                df.to_csv(csv_full_path, index = False)

                # actual data
                csv_name = "episode_" + str(ep) + ".csv"
                actual_data_full_path = data_subdir_path / csv_name
                data_df = pd.DataFrame({
                    "block_x": obs_block_x,
                    "block_y": obs_block_y,
                    "block_ori": obs_block_ori,
                    "ee_x": obs_ee_x,
                    "ee_y": obs_ee_y,
                    "ee_tgt_x": obs_ee_target_x,
                    "ee_tgt_y": obs_ee_target_y,
                    "raw_action_x": raw_input_action_x,
                    "raw_action_y": raw_input_action_y
                })
                data_df.to_csv(actual_data_full_path, index = False)

            if save_trajs:
                # NOTE
                imageio.mimsave(gif_full_path, frames, fps = 2) # fps = 2 is matching time.sleep(0.5), this is equiv to time.sleep(1) 


