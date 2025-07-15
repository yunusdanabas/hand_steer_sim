Below is an **updated, drop-in README.md** followed by a **short blog-style announcement post** you can adapt for your website or LinkedIn.
Feel free to tweak image filenames in the `figures/` folder as you add them.

---

````markdown
# Hand-Steer-Sim  
**Real-Time Gesture Teleoperation for Mobile Robots**

> A **camera-only** hand-gesture interface that turns **webcam / RealSense** input into  
> `geometry_msgs/Twist` commands for differential-drive robots in ROS Noetic.

<div align="center">
  <!-- Replace filenames to match your figures folder -->
  <img src="figures/simplified_system_diagram.png" width="85%" alt="System overview"/>
</div>

---

## ✨ Features

| ✔ | Capability | Notes |
|---|------------|-------|
| 🎥 **Vision-only** | Works with any 30 FPS webcam or Intel RealSense | No depth or wearables |
| 🧠 **Dual-branch ML** | 1 k-param MLP (static) + 6 k-param LSTM (dynamic) | FP16 TFLite models |
| ⚡ **Low-latency** | **13 ms** end-to-end on RTX 4060 Ti (≈ 25 ms laptop CPU) | 30 FPS sustained |
| 📦 **One-command launch** | `roslaunch ... sign_control.launch` | CPU & GPU Dockerfiles |
| 🔄 **Data→Model pipeline** | CLI recorder → notebooks → live test | Fully reproducible |
| 🕹️ **Gazebo + /cmd_vel** | Drop-in for any diff-drive robot | ROS topics only |

---

## 🚀 Quick Demo

```bash
# full camera ▶︎ gesture ▶︎ velocity ▶︎ Gazebo pipeline
roslaunch hand_steer_sim sign_control.launch \
           control_mode:=steering \
           show_image:=true
````

* **Static signs** – **Stop · Holding-Wheel · Speed-Up · Speed-Down** → linear velocity
* **Dynamic wheel gestures** – **Turn-Left(±) · Turn-Right(±) · Forward** → angular velocity

`/robot_diff_drive_controller/cmd_vel` carries the resulting `Twist`, so you can keep the supplied Gazebo robot *or* drive any hardware that subscribes to the same topic.

<div align="center">
  <img src="figures/gazebo_drive.gif" width="70%" alt="Gazebo driving demo"/>
  <p><em>Figure&nbsp;1   Live steering in Gazebo (source video <a href="https://drive.google.com/file/d/1TkqudJsSXfxzetYAVJHKoW3ILWf8vWRE/view">here</a>).</em></p>
</div>

---

## 📈 Performance at a Glance

| Hardware              | Decode (ms) | Inference (ms) | Display (ms) | **Total E2E (ms)** | FPS  |
| --------------------- | ----------- | -------------- | ------------ | ------------------ | ---- |
| **RTX 4060 Ti**       | 0.5         | **8.6**        | 4.0          | **13.2**           | 75.7 |
| ThinkPad E14 (CPU)    | 0.5         | 20.1           | 4.3          | 25.0               | 39.5 |
| ThinkPad E14 + Gazebo | 4.7         | 94.1           | 10.7         | 109.6              | 9.7  |

> *Table 1   Latency breakdown over 1 000 frames (batch size = 1, 960 × 540 @ 30 FPS).*

Overall offline test-set accuracy is **99.7 % (static)** and **99.8 % (dynamic)**.
See `report/EE417_FinalReport_YunusEmreDanabas.pdf` for full confusion matrices and training curves.

---

## 📁 Repository Map

```text
hand_steer_sim/
├─ launch/            # camera + recognition + control (ROS launch files)
├─ scripts/           # Python ROS nodes & CLI tools
├─ model/             # ∟ static_mode/  (MLP)  ∟ steering_mode/ (MLP+LSTM)
├─ data/              # CSV recordings (git-ignored)
├─ urdf/              # Gazebo robot model
├─ config/            # diff-drive controller params
└─ notebooks/         # training notebooks (Jupyter)
```

### CLI shortcuts (installed via `pip install -e .`)

| Command            | Purpose                                                      |
| ------------------ | ------------------------------------------------------------ |
| `hsim_camera_pub`  | Publish webcam / RealSense frames on `/image_raw`            |
| `hsim_record_data` | Full-screen GUI to capture labelled static & dynamic samples |
| `hsim_test_gest`   | Stand-alone viewer for live predictions (no ROS required)    |
| `hsim_hand_sign`   | ROS node – static gesture → `/gesture/hand_sign`             |
| `hsim_gest2twist`  | ROS node – static gesture → `/cmd_vel` (discrete)            |
| `hsim_steer_sign`  | ROS node – static + dynamic wheel gestures                   |
| `hsim_wheel2twist` | ROS node – fused steering → `/cmd_vel` (continuous)          |

---

## 🛠️ Installation

<details>
<summary>Native (Ubuntu 20.04 → 24.04, ROS Noetic)</summary>

```bash
# clone inside a catkin workspace
cd ~/catkin_ws/src
git clone https://github.com/yunusdanabas/hand_steer_sim.git
cd ..

# ROS & Python deps
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport \
                 ros-noetic-tf2-ros ros-noetic-controller-manager
pip install -U pip
pip install -e src/hand_steer_sim[realsense]   # drop [realsense] if not using RealSense

# build Gazebo plugins & messages
catkin_make
source devel/setup.bash
```

> **GPU delegate** – install `libtensorflow-lite-gpu2` and pass `use_gpu:=true` in the launch file for ≈ 2× speed-up.

</details>

<details>
<summary>Docker (CPU & GPU)</summary>

```bash
# example GPU run (persists models & data)
docker run -it --rm --gpus all \
  --network host \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/hand_steer_sim/model:/ws/src/hand_steer_sim/model \
  yunusdanabas/hand_steer_sim:gpu
```

</details>

---

## 🔄 Workflow & Notebooks

| Stage      | Tool / Notebook                                                                                                | Output                         |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **Record** | `hsim_record_data`                                                                                             | CSVs under `data/<timestamp>/` |
| **Train**  | `notebooks/keypoint_classification.ipynb` (static)<br>`notebooks/point_history_classification.ipynb` (dynamic) | `.tflite` models               |
| **Test**   | `hsim_test_gest`                                                                                               | Live overlay of predictions    |
| **Deploy** | `roslaunch hand_steer_sim sign_control.launch`                                                                 | Real-time control              |

---

## 🧑‍💻 Developer Essentials

* **Feature vectors** – 42-D wrist-normalised landmarks (static) · 128-D MCP history (dynamic) <img src="figures/hand_landmarks.png" width="50%" alt="Hand landmarks numbering"/>
* **Smoothing** – dynamic IDs = majority vote across the last 16 frames.
* **Safety** – velocities clamped to ±1 m s⁻¹ (linear) and ±2 rad s⁻¹ (angular).
* **Extending gestures** – add labels in `*.csv`, retrain notebooks, drop new `.tflite` files into `model/`.

---

## 🔭 Roadmap

* **Ackermann-steered vehicle** for more intuitive “steering wheel” control
* **Two-handed gestures** for extra commands (lights, indicators, emergency stop)
* **User study** to quantify learnability & fatigue across diverse participants

---

## 🤝 Acknowledgements

Built as a **solo project** during **EE417 — Computer Vision (Spring 2025, Sabancı University)**.
MediaPipe, TensorFlow Lite, ROS Noetic, and Gazebo form the open-source backbone.

---

## 📜 License

> *No explicit license — feel free to fork, modify, and share.*

<div align="center">
  <img src="figures/hand_steer_banner.png" width="90%" alt="Gesture collage"/>
</div>
```

---

## 📝 Brief Blog-Style Announcement (≈ 260 words)

> *Feel free to post on your website, Medium, or LinkedIn. Replace image paths as needed.*

---

### Hand-Steer Sim — Drive Your Robot with Nothing but a Webcam 🤚🚗

<div align="center">
  <img src="figures/simplified_system_diagram.png" width="70%" alt="System diagram"/>
</div>

I’ve just open-sourced **Hand-Steer Sim**, a real-time teleoperation stack that lets you steer a differential-drive robot with simple hand gestures—no joystick, no depth sensor, no gloves.

**How it works**

* MediaPipe Hands tracks 21 landmarks at 30 FPS.
* A 1 k-param MLP recognises four static commands (*Stop, Holding Wheel, Speed ±*).
* A 6 k-param LSTM observes a 16-frame history of MCP joints to catch dynamic steering gestures (*Turn L/R, Forward*).
* Two ROS nodes fuse the results and publish `Twist` messages to Gazebo—or to any real robot listening on `/cmd_vel`.

On an RTX 4060 Ti the whole loop—from camera frame to `cmd_vel`—averages **13 ms**, sustaining 75 FPS. Even my ThinkPad runs it at \~25 ms. Offline accuracy? **≥ 99 %** for both static and dynamic gestures.

<div align="center">
  <img src="figures/gazebo_drive.gif" width="70%" alt="Gazebo demo"/>
  <p><a href="https://drive.google.com/file/d/1TVqnACMAsV_UAXI_ogMS3fXAkhr-mNMN/view">See the live overlay video →</a></p>
</div>

**Why bother?**

Controllers are great—until they aren’t: cost, accessibility, cabling in field setups. Hand-Steer Sim turns any cheap webcam into an intuitive input device. Everything ships in Docker (CPU & GPU) with a one-line `roslaunch`.

**Try it**

```bash
roslaunch hand_steer_sim sign_control.launch control_mode:=steering
```

Want to customise gestures? Record new samples with `hsim_record_data`, retrain the notebooks, and drop your fresh `.tflite` files into `model/`. Done.

Check out the code, report, and training notebooks on GitHub → **github.com/yunusdanabas/hand\_steer\_sim**. Drop a ⭐ if you think hands are underrated controllers!

---

*Written as my solo capstone for EE417 — Computer Vision (Sabancı University, Spring 2025). Future plans: Ackermann steering, two-handed commands, and a user study on intuitiveness.*