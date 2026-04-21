#!/usr/bin/env python3
import time
import pybullet
from diffusha.data_collection.env.block_pushing.block_pushing_multimodal_1block import BlockPushMultimodal


class BlockPushGUI(BlockPushMultimodal):
    def __init__(self, *args, **kwargs):
        # Force GUI BEFORE parent init
        self._connection_mode = pybullet.GUI
        super().__init__(*args, **kwargs)


def main():
    env = BlockPushGUI(user_goal="target")
    print("PyBullet connection info:", pybullet.getConnectionInfo())

    obs = env.reset()

    print("Env created.")
    print("Observation keys:", list(obs.keys()))

    for step in range(1000):
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

        print(f"step={step}, reward={reward:.3f}, done={done}")

        time.sleep(0.03)

        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()