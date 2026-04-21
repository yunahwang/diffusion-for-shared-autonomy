import pyspacemouse
import time

success = pyspacemouse.open()
if success:
    while 1:
        state = success.read()
        print(state.x, state.y, state.z)
        time.sleep(0.05)
