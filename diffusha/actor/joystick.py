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



def draw_action_arrows(pb_client, ob, raw_action, assisted_action,
                       raw_line_id=None, assisted_line_id=None,
                       z=0.08, scale=0.25):
    """
    Draws:
      green arrow = raw joystick action
      red arrow   = assisted action

    Returns updated line ids so they can be replaced next frame.
    """
    ee_xy = get_effector_xy_from_obs(ob)

    start = [float(ee_xy[0]), float(ee_xy[1]), float(z)]

    raw_end = [
        float(ee_xy[0] + scale * raw_action[0]),
        float(ee_xy[1] + scale * raw_action[1]),
        float(z),
    ]
    assisted_end = [
        float(ee_xy[0] + scale * assisted_action[0]),
        float(ee_xy[1] + scale * assisted_action[1]),
        float(z),
    ]

    print("ee_xy, ", ee_xy, " raw_action, ", raw_action, "raw_end, ", raw_end)

    raw_line_id = pb_client.addUserDebugLine(
        start,
        raw_end,
        lineColorRGB=[0, 1, 0],   # green
        lineWidth=10,
        lifeTime=0.08,
        replaceItemUniqueId=(-1 if raw_line_id is None else raw_line_id),
    )

    assisted_line_id = pb_client.addUserDebugLine(
        start,
        assisted_end,
        lineColorRGB=[1, 0, 0],   # red
        lineWidth=10,
        lifeTime=0.08,
        replaceItemUniqueId=(-1 if assisted_line_id is None else assisted_line_id),
    )

    return raw_line_id, assisted_line_id

def draw_action_trajectory(pb_client, raw_log, assisted_log,
                            traj_ids=None, z=0.05, lifetime=0.12):
    """
    Treats action[0], action[1] as x, y coordinates and draws
    line segments connecting consecutive points.
    green = raw, red = assisted.
    Only draws the newest segment each step (from prev to current).
    """
    traj_ids = traj_ids or {"raw": [], "assisted": []}

    if len(raw_log) >= 2:
        prev_ee, prev_r = raw_log[-2]
        curr_ee, curr_r = raw_log[-1]
        rid = pb_client.addUserDebugLine(
            [float(prev_ee[0] + prev_r[0]), float(prev_ee[1] + prev_r[1]), z],
            [float(curr_ee[0] + curr_r[0]), float(curr_ee[1] + curr_r[1]), z],
            lineColorRGB=[0, 1, 0],
            lineWidth=2,
            lifeTime=lifetime,
        )
        traj_ids["raw"].append(rid)

    if len(assisted_log) >= 2:
        prev_ee, prev_a = assisted_log[-2]
        curr_ee, curr_a = assisted_log[-1]
        aid = pb_client.addUserDebugLine(
            [float(prev_ee[0] + prev_a[0]), float(prev_ee[1] + prev_a[1]), z],
            [float(curr_ee[0] + curr_a[0]), float(curr_ee[1] + curr_a[1]), z],
            lineColorRGB=[1, 0, 0],
            lineWidth=2,
            lifeTime=lifetime,
        )
        traj_ids["assisted"].append(aid)

    return traj_ids


if __name__ == '__main__':
    from diffusha.data_collection.env import make_env

    # TODO - change the following things
    no_assist = False # if False, use DiffusionAssistedActor
    model_id = "xpmbcyvo" # if gamma 0.4 "xpmbcyvo", gamma 0.1 "2sl9lz97", gamma 0.8 "lnxdni8n"
    draw_arrows = True
    fwd_diff_ratio = 0.2

    env_name =  "BlockPushMultimodal-v1"

    env = make_env(
        #"LunarLander-v3",
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

            raw_log      = []
            assisted_log = []
            traj_ids     = {"raw": [], "assisted": []}

            while not done:
                env.render()
                
                raw_action = actor.act(ob)
                print("[before] raw action, ", raw_action, flush = True)
                assisted_action, diff = assisted_actor.act_without_env(ob, raw_action, report_diff = True)

                print("assisted_action, ", assisted_action, flush = True)
                #print("diff, ", diff, flush = True)

                ee_xy = get_effector_xy_from_obs(ob)
                raw_log.append((ee_xy, raw_action.copy()))
                assisted_log.append((ee_xy, assisted_action.copy()))

                if draw_arrows:
                    ax.clear()
                    # After ax.clear(), draw the two targets
                    # ax.scatter([ 0.1, -0.1], [0.0, 0.0], 
                    #     c='blue', marker='*', s=200, zorder=6, label='targets')
                    
                    size = 0.12
                    half = size / 2

                    for tx, ty, color, label in [
                        ( 0.1, 0.0, 'blue',   'target 1'),
                        (-0.1, 0.0, 'purple', 'target 2'),
                    ]:
                        ax.add_patch(Rectangle(
                            (tx - half, ty - half), size, size,
                            linewidth=2, edgecolor=color, facecolor='lightyellow', alpha=0.4, label=label))
                    ee_xs = [p[0][0] for p in raw_log]
                    ee_ys = [p[0][1] for p in raw_log]

                    raw_end_xs = [p[0][0] + p[1][0] for p in raw_log]
                    raw_end_ys = [p[0][1] + p[1][1] for p in raw_log]

                    ass_end_xs = [p[0][0] for p in assisted_log]
                    ass_end_ys = [p[0][1] for p in assisted_log]

                    # Plot ee trajectory
                    #$ax.plot(ee_xs, ee_ys, 'k--', linewidth=1, label='EE path')

                    # # Current step arrow: ee -> raw action endpoint
                    # ax.annotate('', xy=(raw_end_xs[-1], raw_end_ys[-1]),
                    #             xytext=(ee_xs[-1], ee_ys[-1]),
                    #             arrowprops=dict(arrowstyle='->', color='green', lw=2))

                    # # Current step arrow: ee -> assisted action endpoint
                    # ax.annotate('', xy=(ass_end_xs[-1], ass_end_ys[-1]),
                    #             xytext=(ee_xs[-1], ee_ys[-1]),
                    #             arrowprops=dict(arrowstyle='->', color='red', lw=2))

                    ax.plot(ass_end_xs, ass_end_ys, "k--", linewidth=1)

                    ax.set_xlim(-0.4, 0.4)
                    ax.set_ylim(-0.6, 0.3)

                    # Scatter current ee position
                    #ax.scatter(ee_xs[-1], ee_ys[-1], c='black', zorder=5)

                    plt.pause(0.001)
                    # raw_line_id, assisted_line_id = draw_action_arrows(
                    #     pb,
                    #     ob,
                    #     raw_action,
                    #     assisted_action,
                    #     raw_line_id=raw_line_id,
                    #     assisted_line_id=assisted_line_id,
                    #     z=0,
                    #     scale=1,
                    # )

                    # traj_ids = draw_action_trajectory(
                    #     pb, raw_log, assisted_log,
                    #     traj_ids=traj_ids,
                    #     z=0.05,
                    #     lifetime=1.5,   # 0 = persist until next episode
                    # )

                ob, r, done, _= env.step(assisted_action)
                reward += r
            print("episode reward: ", reward)
