import time
import pybullet as p
import pybullet_data

cid = p.connect(p.GUI)
print("cid:", cid)
print("connection info:", p.getConnectionInfo())

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

for _ in range(1000):
    p.stepSimulation()
    time.sleep(1/240)