import time
import numpy as np
import pandas as pd
from pathlib import Path
import pybullet
from scipy.spatial.transform import Rotation as R

from diffusha.data_collection.env import make_env
from diffusha.data_collection.env.utils.pose3d import Pose3d

TARGET_TOL = 0.04

WORLD_TARGETS = {
    "green-right": np.array([0.5, 0.35]),
    "red-left": np.array([0.3, 0.35]),
}


def check_ee_goal(world_xy, tol=TARGET_TOL):
    dists = {
        name: np.linalg.norm(world_xy - target_xy)
        for name, target_xy in WORLD_TARGETS.items()
    }

    closest = min(dists, key=dists.get)

    if dists[closest] < tol:
        return closest, dists

    return None, dists

def unwrap(env):

    return env.unwrapped

def get_obs():
    # NOTE: this goes right/green
    folder_path = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "orig_2023_csv_backup" 
    
    # NOTE: this goes left/red
    #folder_path = Path(__file__).parents[2] / "data-dir" / "replay" / "blockpush" / "target-flipped" / "realsies_2023_flipped_csv_backup_100"
    csv_files = sorted(Path(folder_path).glob("*.csv"))
    n_files = len(csv_files)
    all_obs = []

    for i, csv_file in enumerate(csv_files):
        per_file_obs = []
        df = pd.read_csv(csv_file)
        for row in df.itertuples(index=False):
            state = list(row)[0:9]
            ee_x = -row[3]
            ee_y = row[4]
            per_file_obs.append((ee_x, ee_y))
            #break
        #break
        all_obs.append(per_file_obs)
    return folder_path, all_obs


def main():
    env = make_env(
        "BlockPushMultimodal-v1",
        seed=1,
        test=False,
        user_goal="target",
    )

    ob = env.reset()
    env.render()

    base = unwrap(env)
    print("base, ", base)
    # for k, v in base.__dict__.items():
    #     print(k, type(v))
    #     """
    #     _target_ids <class 'list'> <- MAYBE USEFUL
    #     _target_poses <class 'list'> <- MAYBE USEFUL
    #     _ori_dist <class 'numpy.float64'>
    #     _prev_dist <class 'numpy.float64'>
    #     _prev_blk_pos <class 'numpy.ndarray'>
    #     user_goal <class 'str'>
    #     _task <enum 'BlockTaskVariant'>
    #     _connection_mode <class 'int'>
    #     goal_dist_tolerance <class 'float'>
    #     effector_height <class 'float'>
    #     _visuals_mode <class 'str'>
    #     _camera_pose <class 'tuple'>
    #     _camera_orientation <class 'tuple'>
    #     workspace_bounds <class 'numpy.ndarray'>
    #     _image_size <class 'NoneType'>
    #     _camera_instrinsics <class 'tuple'>
    #     _workspace_urdf_path <class 'str'>
    #     action_space <class 'gym.spaces.box.Box'>
    #     observation_space <class 'gym.spaces.dict.Dict'>
    #     _rng <class 'numpy.random.mtrand.RandomState'>
    #     _block_ids <class 'list'>
    #     _previous_state <class 'collections.OrderedDict'>
    #     _robot <class 'diffusha.data_collection.env.utils.xarm_sim_robot.XArmSimRobot'> <- MAYBE USE THIS
    #     _workspace_uid <class 'int'>
    #     _target_id <class 'NoneType'>
    #     _target_pose <class 'NoneType'>
    #     _target_effector_pose <class 'diffusha.data_collection.env.utils.pose3d.Pose3d'> <- MAYBE USEFUL
    #     _pybullet_client <class 'pybullet_utils.bullet_client.BulletClient'>
    #     reach_target_translation <class 'NoneType'>
    #     _saved_state <class 'int'>
    #     _control_frequency <class 'float'>
    #     _step_frequency <class 'float'>
    #     _last_loop_time <class 'NoneType'>
    #     _last_loop_frame_sleep_time <class 'NoneType'>
    #     _sim_steps_per_step <class 'int'>
    #     _ori_blk2ee <class 'numpy.float64'>
    #     _prev_blk2ee <class 'numpy.float64'>
    #     spec <class 'gym.envs.registration.EnvSpec'>
    #     """

    robot = base._robot
    print("robot, ", robot)
    pb = base._pybullet_client
    print("pb, ", pb)

    # get obs
    path, all_obs = get_obs()
    print("path", path)

    start_pose = robot.forward_kinematics()
    fixed_z = start_pose.translation[2]
    fixed_rot = start_pose.rotation

    coord_offset = np.array([0.4, 0.35])

    global_start_time = time.time()

    total_obs_all_files = sum(len(f) for f in all_obs)
    processed_obs_global = 0

    for file_i, file_obs in enumerate(all_obs):
        file_start_time = time.time()
        last_100_time = time.time()

        n_obs_file = len(file_obs)

        for i, obs in enumerate(file_obs):
            processed_obs_global += 1

            if i % 100 == 0 and i > 0:

                # ----- local 100-step timing -----
                now = time.time()

                elapsed_100 = now - last_100_time
                sec_per_step = elapsed_100 / 100.0

                remaining_steps_file = n_obs_file - i
                eta_file_sec = remaining_steps_file * sec_per_step

                remaining_steps_global = (
                    total_obs_all_files - processed_obs_global
                )
                eta_global_sec = remaining_steps_global * sec_per_step

                elapsed_file_total = now - file_start_time
                elapsed_global_total = now - global_start_time

                print(
                    f"[file {file_i}] "
                    f"step {i}/{n_obs_file} | "
                    f"{elapsed_100:.2f}s per 100 | "
                    f"{sec_per_step:.4f}s/step | "
                    f"file ETA: {eta_file_sec/60:.2f} min | "
                    f"global ETA: {eta_global_sec/60:.2f} min | "
                    f"elapsed file: {elapsed_file_total/60:.2f} min | "
                    f"elapsed total: {elapsed_global_total/60:.2f} min"
                )

                last_100_time = now

            ee_x, ee_y = obs

            world_xy = np.array([ee_x, ee_y]) + coord_offset

            goal, dists = check_ee_goal(world_xy)

            if goal is not None and hit_goal is None:
                hit_goal = goal
                hit_step = i
                print(
                    f"[file {file_i}] EE entered {goal} at step {i} "
                    f"dist_target={dists['target']:.4f}, "
                    f"dist_target2={dists['target2']:.4f}"
                )

            target_pose = Pose3d(
                translation=np.array([world_xy[0], world_xy[1], fixed_z]),
                rotation=fixed_rot,
            )

            robot.set_target_effector_pose(target_pose)

            #print("sim_steps_per_step, ", base._sim_steps_per_step) #24

            for _ in range(base._sim_steps_per_step):
                pb.stepSimulation()

            env.render()
            #time.sleep(1 / 30)



if __name__ == "__main__":
    main()

