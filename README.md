# Yuna - Tim OOD detection /& shared autonomy project

## Pre-docker and Docker command for env setup
- if using PyBullet GUI and need x11 forwarding, here are the series of commands you should run:

```bash
xhost +SI:localuser:root
export WANDB_API_KEY=<replace_with_yours>
export CODE_DIR=`pwd`
export DATA_DIR=$CODE_DIR/data-dir
export OUT_DIR=$CODE_DIR/output-dir
sudo docker run -it --gpus '"device=0"' --device /dev/input --entrypoint /bin/bash -e DISPLAY=:1 -e SDL_VIDEODRIVER=x11 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e WANDB_API_KEY=$WANDB_API_KEY -v $CODE_DIR:/code -v $DATA_DIR:/data -v $OUT_DIR:/outdir --workdir /code ripl/diffusion-for-shared-autonomy
cd Documents/diffusion-for-shared-autonomy # this is needed because entrypoint is just at /bin/bash
```
- running joystick
```bash
python -m diffusha.actor.joystick
```
    - joystick does not need separate installation, just plug in to usb-3
    - you use the right handle for up/down operation of the robot end-effector and you use the left trigger (LT text engraved on top) for left/right operation