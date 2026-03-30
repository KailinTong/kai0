# Single-Arm + 2-Camera Hardware Setup

Hardware requirements and configuration for running Kai0 inference with **1 Piper arm + 2 Intel RealSense cameras**, adapted from the original dual-arm + triple-camera setup.

---

## Components

| Component | Model | Notes |
|-----------|-------|-------|
| Arm | AgileX Piper (single) | 6-DoF + gripper, CAN interface |
| Head camera | Intel RealSense **D456** (serial: `311322303242`) | Has dedicated RGB sensor, standard color stream |
| Wrist camera | Intel RealSense **D405** (serial: `352122270893`) | **No RGB sensor** — uses infrared stream in RGB8 mode |
| GPU server | 4× NVIDIA RTX 4090 | Only 1 GPU needed for inference (≥8 GB VRAM) |
| Robot IPC / Laptop | Ubuntu 22.04, ROS 2 Humble | Runs robot stack + inference client |

## Comparison with Original Kai0 Setup

| | Original kai0 | This setup |
|---|---|---|
| Arms | 2× Piper (dual-arm) | 1× Piper |
| Cameras | 3× Intel RealSense **D435i** | 1× D456 (head) + 1× D405 (wrist) |
| ROS | ROS 1 Noetic | **ROS 2 Humble** |
| State dim | 14 (7 left + 7 right) | 7 → duplicated to 14 for model |
| Camera streams | All use `/color/image_raw` | D456 uses `/color/image_raw`, D405 uses `/infra1/image_rect_raw` |

## D405 vs D435i — Key Difference

The **D435i** has a dedicated RGB ISP → `enable_color` works, publishes on `/color/image_raw`.

The **D405** is a close-range depth sensor with **no dedicated RGB module**. Its color stream comes from the Stereo Module infrared sensor configured in RGB8 format:

- Launch param: `enable_infra1: true`, `depth_module.infra_format: RGB8`
- Topic: `/camera_wrist/camera_wrist/infra1/image_rect_raw`
- Max resolution: 640×480 @ 15 fps (on USB 2.1)

## USB Considerations

- Both cameras should be on **separate USB controllers** if possible (different sides of the laptop)
- D405 reports `Usb Type Descriptor: 2.1` (limited bandwidth)
- Head camera (D456) runs at 424×240 @ 30 fps
- Wrist camera (D405) runs at 640×480 @ 15 fps (via infra1 stream)
- The inference script resizes all images to 224×224 anyway

## Software Prerequisites

```bash
# ROS 2 Humble (already installed)
# librealsense2 (already installed via ros-humble-librealsense2)

# RealSense ROS 2 wrapper (built from source — apt package is broken)
cd /home/tongadm/projects/piper_ros/src
git clone -b 4.55.1 https://github.com/IntelRealSense/realsense-ros.git
sudo apt install ros-humble-unique-identifier-msgs
cd .. && colcon build --packages-select realsense2_camera_msgs realsense2_camera realsense2_description

# Piper ROS 2
cd /home/tongadm/projects/piper_ros && colcon build && source install/setup.bash

# OpenPi client
cd /home/tongadm/projects/kai0/packages/openpi-client && pip install -e .
```

## Camera Serial Numbers

Find serials with:
```bash
rs-enumerate-devices | grep -E "Name|Serial Number"
```

Current setup:
- D456 (head): `311322303242`
- D405 (wrist): `352122270893`

---

## Deployment — GPU Server

```bash
# 1. Download checkpoints (one-time, ~several GB)
#    The HF repo uses Task_A/, Task_B/, Task_C/ (not FlattenFold/)
cd ~/workspace/kai0
uv run python scripts/download_checkpoints.py

# If the server cannot reach Hugging Face, download on the laptop and SCP:
#   cd /tmp
#   python3 -c "
#   from huggingface_hub import snapshot_download
#   snapshot_download('OpenDriveLab-org/Kai0',
#                     repo_type='model',
#                     allow_patterns=['Task_A/*', 'README.md'],
#                     local_dir='./kai0_checkpoints')
#   "
#   scp -r /tmp/kai0_checkpoints/Task_A tugraz-kailin@<GPU_SERVER_IP>:~/workspace/kai0/my_checkpoints/

# 2. Check downloaded checkpoints
find ./checkpoints -maxdepth 4 -type d | head -20
# or if using my_checkpoints:
find ./my_checkpoints -maxdepth 4 -type d | head -20

# 3. Start the policy server (uses 1 GPU, ~8 GB VRAM)
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_flatten_fold_normal \
  --policy.dir=./checkpoints/Task_A/best/90000
# NOTE: adjust --policy.dir to match the actual path from step 2
```

> **NOTE:** `--port` must come **before** `policy:checkpoint` (it's a top-level arg).

Available Agilex configs:

| Config | Task |
|---|---|
| `pi05_flatten_fold_normal` | Flatten + fold cloth |
| `pi05_tee_shirt_sort_normal` | T-shirt sorting |
| `pi05_flatten_fold_awbc` | Flatten + fold (advantage-weighted) |
| `pi05_tee_shirt_sort_awbc` | T-shirt sorting (advantage-weighted) |

---

## Deployment — Laptop (Robot IPC)

### Terminal 1: Hardware launch (arm + cameras + RViz)

```bash
cd /home/tongadm/projects/piper_ros
source install/setup.bash
sudo bash can_activate.sh can0

ros2 launch piper start_single_piper_cameras_rviz.launch.py \
  can_port:=can0 auto_enable:=true \
  head_serial:="'311322303242'" wrist_serial:="'352122270893'"
```

### Terminal 2: Verify topics

```bash
source /home/tongadm/projects/piper_ros/install/setup.bash
ros2 topic list | grep camera
ros2 topic hz /camera_head/camera_head/color/image_raw
ros2 topic hz /camera_wrist/camera_wrist/infra1/image_rect_raw
```

### Terminal 3: SSH tunnel (if server not on same LAN)

```bash
ssh -L 8000:localhost:8000 tugraz-kailin@<GPU_SERVER_IP>
```

### Terminal 4: Inference client

```bash
source /home/tongadm/projects/piper_ros/install/setup.bash
cd /home/tongadm/projects/kai0

python train_deploy_alignment/inference/agilex/inference/agilex_inference_ros2_single_arm.py \
  --host <GPU_SERVER_IP> --port 8000 \
  --use_temporal_smoothing --chunk_size 50 \
  --prompt "fold the sleeve"
```

> Use `--host localhost` if using the SSH tunnel.
> Add `--use_right_arm_action` to use the right-arm action slot (dims 7:14).

### What to expect

1. Arm moves to home position
2. Script prints "Press Enter to start inference..." → verify arm is safe, then press Enter
3. Warmup inference runs, then control loop starts
4. **Arm will behave erratically** — this is expected with a dual-arm checkpoint on a single arm
5. Goal: verify the full pipeline works (cameras → server → actions → arm moves)
