# Hand-Steer-Sim

*A camera-based hand-gesture interface for ROS Noetic differential-drive robots.*

---

## 🚀 Quick Demo

```bash
# Start the full pipeline (camera ▶︎ gesture ▶︎ velocity ▶︎ Gazebo)
roslaunch hand_steer_sim sign_control.launch \
           control_mode:=steering \
           show_image:=true
```

* **Static gestures** – **Stop / Holding-Wheel / Speed-Up / Speed-Down** → change **linear speed**
* **Dynamic wheel gestures** – **Turn-Left(±) / Turn-Right(±)** → change **angular speed**

The stack publishes `geometry_msgs/Twist` on
`/robot_diff_drive_controller/cmd_vel`, so you can either keep the Gazebo robot from the launch file **or** plug in any real robot that subscribes to the same topic.

---

## 📁 Repository at a Glance

```
hand_steer_sim/
├─ launch/            # one-click launch files (camera + inference + control)
├─ scripts/           # pure-Python ROS nodes & CLI utilities
├─ model/
│   ├─ static_mode/   # TFLite model & labels for static gestures
│   └─ steering_mode/ # key-point + LSTM models for wheel-turn dynamics
├─ data/              # (git-ignored) CSVs recorded by hsim_record_data
├─ urdf/  config/     # Gazebo diff-drive robot & controller params
└─ setup.py           # PEP 517 + catkin install recipe
```

### Handy CLI shortcuts (installed by `pip install -e .`)

| command            | purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| `hsim_camera_pub`  | publish webcam / RealSense frames on `/image_raw`              |
| `hsim_record_data` | fullscreen GUI to capture new static & dynamic gesture samples |
| `hsim_test_gest`   | stand-alone live visualiser of model predictions               |
| `hsim_hand_sign`   | ROS node – static gesture → `/gesture/hand_sign`               |
| `hsim_gest2twist`  | ROS node – static gesture → `/cmd_vel` (discrete)              |
| `hsim_steer_sign`  | ROS node – steering (static + dynamic) → two gesture topics    |
| `hsim_wheel2twist` | ROS node – steering gesture → `/cmd_vel` (continuous)          |

---

## 🛠️  Installation

### 1. Native (Ubuntu 20.04 – 24.04, ROS Noetic)

```bash
# clone into your catkin workspace
cd ~/catkin_ws/src
git clone https://github.com/yunusdanabas/hand_steer_sim.git
cd ..

# ROS + Python deps
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport \
                 ros-noetic-tf2-ros ros-noetic-controller-manager
pip install -U pip
pip install -e src/hand_steer_sim[realsense]   # drop [realsense] if not needed

# build Gazebo plugins & msgs
catkin_make
source devel/setup.bash
```

> **GPU delegate** – Install `libtensorflow-lite-gpu2` and pass `use_gpu:=true` in the launch files for faster inference.

### 2. Docker

Two multi-stage images are provided:

| Tag              | Purpose                           |
| ---------------- | --------------------------------- |
| `hand_steer:cpu` | CPU-only development & deployment |
| `hand_steer:gpu` | CUDA 11.8 + TF-Lite GPU delegate  |

#### Run (GPU example)

```bash
xhost +local:docker    # allow X11
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

> The `-v $(pwd)/hand_steer_sim/model:…` mount **persists trained models & CSVs** across container restarts.

---

## 🔄  Workflow: Collect → Train → Deploy

| Stage      | Tool / Notebook                                                                                                  | Hint                                                               |
| ---------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Record** | `hsim_record_data`                                                                                               | Press **Enter** to snapshot; files saved under `data/<timestamp>/` |
| **Train**  | `notebooks/keypoint_classification.ipynb` (static) <br> `notebooks/point_history_classification.ipynb` (dynamic) | Generates `.tflite` models (42-D static, 128-D dynamic)            |
| **Test**   | `hsim_test_gest`                                                                                                 | Live overlay of static **and** dynamic predictions                 |
| **Deploy** | `roslaunch hand_steer_sim sign_control.launch …`                                                                 | Choose `control_mode:=static` **or** `control_mode:=steering`      |

---

## 🧑‍💻 Developer Notes

* **Features**
  *Static* → 21 landmarks → 42-element wrist-relative vector
  *Dynamic* → 16-frame history of 4 MCP joints → 128-element vector
* **Smoothing** – Dynamic IDs are majority-voted over the last 16 frames for robustness.
* **Recorder threading** – A background writer keeps capture FPS high while samples are saved.

Made with ♥ by Yunus Emre Danabaş
