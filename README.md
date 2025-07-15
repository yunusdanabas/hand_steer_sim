# Hand-Steer-Sim

Hand-Steer-Sim provides a camera-based interface for controlling differential-drive robots in ROS. Static hand signs and continuous steering gestures are converted into `geometry_msgs/Twist` messages, enabling both discrete speed commands and smooth turning.

## Setup

### Requirements
- Ubuntu 20.04--24.04 with ROS Noetic
- Python 3.8+

### Installation
```bash
# clone into your catkin workspace
cd ~/catkin_ws/src
git clone https://github.com/yunusdanabas/hand_steer_sim.git
cd ..

# ROS + Python dependencies
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport \
                 ros-noetic-tf2-ros ros-noetic-controller-manager
pip install -U pip
pip install -e src/hand_steer_sim[realsense]   # drop [realsense] if not needed

# build Gazebo plugins & messages
catkin_make
source devel/setup.bash
```
**GPU delegate** – Install `libtensorflow-lite-gpu2` and pass `use_gpu:=true` in the launch files for faster inference.

Docker images `hand_steer:cpu` and `hand_steer:gpu` are also available. Example GPU run:
```bash
xhost +local:docker
docker run -it --rm --gpus all \
  --name hand_steer_gpu \
  --network host \
  --device /dev/bus/usb:/dev/bus/usb \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/home/user/.Xauthority:ro \
  -e DISPLAY=$DISPLAY -e XAUTHORITY=/home/user/.Xauthority \
  --privileged \
  -v $(pwd)/hand_steer_sim/model:/catkin_ws/src/hand_steer_sim/model \
  hand_steer:gpu
```

## Running

Start the complete camera-to-velocity pipeline:
```bash
roslaunch hand_steer_sim sign_control.launch control_mode:=steering show_image:=true
```
This command brings up a Gazebo robot and publishes velocity on `/robot_diff_drive_controller/cmd_vel`.

CLI utilities installed via `pip install -e .` include `hsim_camera_pub`, `hsim_record_data`, `hsim_test_gest`, `hsim_hand_sign`, `hsim_gest2twist`, `hsim_steer_sign`, and `hsim_wheel2twist`.

## Directory Structure
```text
hand_steer_sim/
├─ launch/            # launch files (camera, inference, control)
├─ scripts/           # Python ROS nodes & CLI tools
├─ model/
│   ├─ static_mode/   # static gesture models and labels
│   └─ steering_mode/ # key-point + LSTM models for wheel-turn dynamics
├─ data/              # recorded CSVs (not in git)
├─ urdf/  config/     # Gazebo robot & controller params
└─ setup.py           # PEP 517 + catkin install recipe
```

## Technologies
- ROS Noetic (rospy, geometry_msgs, cv_bridge, etc.)
- Python / OpenCV / Mediapipe
- TensorFlow Lite for ML inference
- Gazebo for simulation
- Intel RealSense (optional)

## Example Workflow
1. **Record** gestures with `hsim_record_data` (saves CSVs under `data/<timestamp>/`).
2. **Train** models using the notebooks in `model/**/notebooks`.
3. **Test** predictions live with `hsim_test_gest`.
4. **Deploy** using `roslaunch hand_steer_sim sign_control.launch`.

See [`report/EE417_FinalReport_YunusEmreDanabas.pdf`](report/EE417_FinalReport_YunusEmreDanabas.pdf) for an in-depth description of the project.

This repository is released under the MIT License.

