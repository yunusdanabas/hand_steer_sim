# hand_steer_sim

NOT READY JUST SOME NOTES


````markdown
## Distributing the `hand_steer:cpu` Docker Image via Tarball

If you received a `hand_steer_cpu.tar` file instead of building locally, follow these steps on your target machine:

### 1. Prerequisites
- Docker installed and running  
- X server (for Gazebo GUI) and permission to connect  
- Video devices available at `/dev/video0` and `/dev/video1` (or adjust as needed)

### 2. Load the Image
Copy `hand_steer_cpu.tar` onto your PC (e.g. via USB, `scp`, etc.), then run:
```bash
docker load -i hand_steer_cpu.tar
````

This will register the `hand_steer:cpu` image in your local Docker.

### 3. Allow GUI Forwarding

Before launching, permit the container to connect to your X server:

```bash
xhost +local:docker
```

### 4. Run the Container

Start the container with device passthrough and GUI support:

```bash
docker run -it --rm \
  --name hand_steer_cpu \
  --network host \
  --device /dev/video0 \
  --device /dev/video1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --privileged \
  hand_steer:cpu
```

* The container will drop you into a shell as user `yunusdanabasDocker`.
* Your Catkin workspace is mounted at `/catkin_ws` and sourced automatically.
* Gazebo GUI and camera nodes should now launch as in the development setup.

---

When you’re done, simply exit the container (`Ctrl+D` or `exit`)—it will clean itself up (`--rm`).

```
```
