# Hand-Steer-Sim 
*Camera-based hand-gesture interface for differential-drive robots (ROS Noetic)*

---

## 0 . Quick Demo

```bash
# ①  start the ROS graph (sim + gesture control + UI)
roslaunch hand_steer_sim sign_control.launch control_mode:=steering show_image:=true

# ②  wave your hand in front of the webcam / RealSense:
#     • static gestures (Stop / Holding Wheel / Speed Up / Speed Down) change linear speed
#     • dynamic wheel gestures (Turn Left/Right ± Fast) steer the robot
```

The node set will publish `geometry_msgs/Twist` to `/robot_diff_drive_controller/cmd_vel`, so you can attach either the Gazebo mobile-base in the launch file or a real robot that listens on the same topic.

---

## 1 . Project Structure

```text
hand_steer_sim/
├─ launch/               ← one-click launch files (camera + inference + control)
├─ scripts/              ← pure-Python ROS nodes & CLI utilities
├─ model/
│   ├─ static_mode/      ← TFLite model & labels for 21-key-point static gestures
│   └─ steering_mode/    ← key-point + LSTM models for wheel-turn dynamics
├─ data/                 ← (ignored) recording sessions: data/<ts>/<csv>
├─ urdf/ config/         ← Gazebo differential-drive robot
└─ setup.py              ← pip/rosrun installation recipe
```

After `pip install -e .` the following console commands are on your `$PATH`:

| command            | what it does                                                  |
| ------------------ | ------------------------------------------------------------- |
| `hsim_camera_pub`  | Publishes raw images (`/image_raw`) from webcam or RealSense. |
| `hsim_record_data` | GUI tool to record new static / dynamic gesture CSVs.         |
| `hsim_test_gest`   | Stand-alone viewer to live-test the models.                   |
| `hsim_hand_sign`   | ROS node – static gesture → `/gesture/hand_sign`.             |
| `hsim_gest2twist`  | ROS node – static gesture → `/cmd_vel` (discrete).            |
| `hsim_steer_sign`  | ROS node – steering (static + dynamic) → two gesture topics.  |
| `hsim_wheel2twist` | ROS node – steering gesture → `/cmd_vel` (continuous).        |

---

## 2 . Installation

### 2.1 Native (Ubuntu 20.04 ± 24.04, ROS Noetic)

```bash
# clone into your catkin workspace
cd ~/catkin_ws/src
git clone https://github.com/<you>/hand_steer_sim.git
cd ..

# dependencies
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport \
                 ros-noetic-tf2-ros       ros-noetic-controller-manager
pip install -U pip
pip install -e src/hand_steer_sim[realsense]     # add [realsense] if you need it

# build the workspace for Gazebo plugins, etc.
catkin_make
source devel/setup.bash
```

> **GPU delegate** – if you have an NVIDIA card and installed the TF-Lite GPU
> runtime (`sudo apt install libtensorflow-lite-gpu2`), pass `use_gpu:=true`
> in the launch files to accelerate inference.

### 2.2 Docker Images

| image tag        | purpose                                             | build                         |
| ---------------- | --------------------------------------------------- | ----------------------------- |
| `hand_steer:cpu` | development & deployment on CPU-only PCs            | `docker/build_cpu.Dockerfile` |
| `hand_steer:gpu` | leverages NVIDIA GPU via CUDA 12 + TF-Lite delegate | `docker/build_gpu.Dockerfile` |

#### Using a pre-built tarball

If you received `hand_steer_cpu.tar`, load and run:

```bash
docker load -i hand_steer_cpu.tar
xhost +local:docker        # allow GUI
docker run -it --rm \
       --network host \
       --device /dev/video0 \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
       --privileged \
       --name hand_steer hand_steer:cpu
```

Inside the container all ROS and Python tools are pre-sourced.

---

## 3 . Collect → Train → Deploy

| step         | tool                                      | notes                                                                                        |                       |
| ------------ | ----------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------- |
| **Record**   | `hsim_record_data`                        | Full-screen GUI, snapshots with **Enter**. Data saved under `data/<ts>/`.                    |                       |
| **Train**    | the two Jupyter notebooks in `notebooks/` | `keypoint_classification.ipynb` (static) and `point_history_classification.ipynb` (dynamic). |                       |
| **Quantise** | notebooks generate `.tflite`              | 64-byte point-history vectors, 42-D key-point vectors.                                       |                       |
| **Test**     | `hsim_test_gest`                          | Live overlay shows \*static                                                                  | dynamic\* prediction. |
| **Deploy**   | ROS launch files above                    | Switch \`control\_mode:=static                                                               | steering\`.           |

---

## 4 . Developer Notes

* **Feature vectors**
  *Static* 21 hand landmarks → `42` columns: `[x₀…x₂₀, y₀…y₂₀]`, wrist-relative, ±1-normalised.
  *Dynamic* 8 frames × 4 MCPs → `128` columns: first all *x*, then all *y*, expressed in image-relative units.

* **Majority vote smoothing** in `SteeringRecognition` gives more stable dynamic-gesture IDs.

* **Threading & queues** in `hsim_record_data` keep camera FPS near V4L2 max by pushing heavy inference onto a worker thread.

---

## 5 . License

This repository is released under the **MIT License**.
See `LICENSE` for details.

---

Made with luv by **Yunus Emre Danabaş**