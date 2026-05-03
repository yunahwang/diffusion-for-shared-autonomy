#!/usr/bin/env python3
"""Adopted from https://github.com/cbschaff/rsa/blob/master/lunar_lander/joystick_agent.py"""

import time
import pygame
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import json
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from diffusha.actor import Actor
from diffusha.actor.assistive import DiffusionAssistedActor
from diffusha.diffusion.evaluation.helper import prepare_diffusha

#####################################
# Change these to match your joystick
UP_AXIS = 3  # AKA ；up(negative) and down(positive)
SIDE_AXIS = 2  # AKA ；left and right
#####################################
np.set_printoptions(precision=12, suppress=False)

class JoystickActor(Actor):
    """Joystick Controller for Block Pushing."""
    """Joystick Controller for Lunar Lander.""" # was

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
                    print("up/down")
                    self.human_agent_action[0] = -1 * v

                elif event.axis == SIDE_AXIS:
                    print("left/right")
                    self.human_agent_action[1] = v

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
    print("ob, ", ob)

    return [ob[3], ob[4]]


if __name__ == '__main__':
    from diffusha.data_collection.env import make_env

    
    no_assist = False # if False, use DiffusionAssistedActor
    model_id = "xpmbcyvo" # if gamma 0.4 "xpmbcyvo", gamma 0.1 "2sl9lz97", gamma 0.8 "lnxdni8n"
    draw_trajs = True
    fwd_diff_ratio = 1.0 # NOTE - change this

    env_name =  "BlockPushMultimodal-v1"

    env = make_env(
        env_name,
        seed=1,
        test=False
    )

    actor = JoystickActor(env)

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
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))
        plt.show(block = False)

        obs_space = env.observation_space
        act_space = env.action_space
        print("obs_space, ", obs_space)
        print("act_space, ", act_space)

        with open(Path(__file__).parents[1] / "diffusion" / "evaluation" / "configs.json", "r") as f:
            env2config = json.load(f)

        #model_dir = Path(__file__).parents[2] / "data-dir" / "ddpm" / "diffusha" / model_id 
        
        model_dir = Path(__file__).parents[2] / "tr3wtwfz" 
        laggy_actor_repeat_prob = 0; noisy_actor_eps = 0

        diffusion = prepare_diffusha(
            env, 
            env2config[env_name], 
            model_dir,
            29999,
            env_name,
            fwd_diff_ratio,
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
            fwd_diff_ratio = fwd_diff_ratio
        )   

        print(assisted_actor)

        raw_line_id = None
        assisted_line_id = None
        text_id = None

        # load human demonstrator
        for ep in range(10000):
            ob = env.reset()
            
            done = False
            reward = 0.0
            step_i = 0

            ee_log = []
            raw_log      = []
            assisted_log = []

            traj_ids     = {"raw": [], "assisted": []}

            # # Before the while loop, get initial EE position
            # raw_ee_x, raw_ee_y = ob[3], ob[4]  # starting position is same for both
            # ass_ee_x, ass_ee_y = ob[3], ob[4]

            while not done:
                env.render()

                ee_xy = get_effector_xy_from_obs(ob)
                ee_log.append(ee_xy)

                raw_action = actor.act(ob)
                print("[before] raw action, ", raw_action, flush=True)

                assisted_action, diff = assisted_actor.act_without_env(ob, raw_action, report_diff=True)
                print("assisted_action, ", assisted_action, flush=True)

                raw_log.append((ee_xy, raw_action.copy()))
                assisted_log.append((ee_xy, assisted_action.copy()))

                if draw_trajs:
                    ax.clear()

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

                #ob, r, done, _ = env.step(raw_action)
                ob, r, done, _ = env.step(assisted_action)
                reward += r
                step_i += 1
            print("episode reward: ", reward)
