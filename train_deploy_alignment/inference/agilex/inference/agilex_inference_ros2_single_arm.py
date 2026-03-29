# -- coding: UTF-8
"""
ROS 2 Humble single-arm + 2-camera inference client for Kai0 Agilex.

Adapted from the ROS 1 (Noetic) dual-arm inference script to work with:
  - 1 AgileX Piper arm via the Piper ROS 2 package (piper_ctrl_single_node)
  - 2 Intel RealSense cameras (D456 = head, D405 = wrist) via realsense2_camera

Model compatibility hack:
  1. 7-dim proprioception is duplicated to 14-dim for the pretrained model.
  2. A black dummy image fills the 3rd camera slot (hand_left).
  3. Only one 7-dim slice of the 14-dim action output is used.

WARNING: The pretrained dual-arm checkpoint will NOT produce meaningful actions
for a single-arm setup. Retraining is required. This is for pipeline testing.

Prerequisites:
  sudo apt install ros-humble-realsense2-camera
  cd /path/to/piper_ros && colcon build && source install/setup.bash
  cd /path/to/kai0/packages/openpi-client && pip install -e .
"""

import argparse
import signal
import sys
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
from cv_bridge import CvBridge

from openpi_client import image_tools, websocket_client_policy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_DIM_SINGLE = 7     # one Piper arm (6 joints + gripper)
STATE_DIM_MODEL  = 14    # what the pretrained model expects
DUMMY_IMG_SHAPE  = (224, 224, 3)

RIGHT_OFFSET = 0.003

# ---------------------------------------------------------------------------
# Globals (kept similar to original for structural parity)
# ---------------------------------------------------------------------------
stream_buffer = None          # StreamActionBuffer
observation_window = None
lang_embeddings = "fold the sleeve"

published_actions_history = []
shutdown_event = threading.Event()


def _make_dummy_image():
    """224x224x3 black BGR image."""
    return np.zeros(DUMMY_IMG_SHAPE, dtype=np.uint8)


# ---------------------------------------------------------------------------
# StreamActionBuffer (unchanged — operates on 14-dim model output)
# ---------------------------------------------------------------------------
class StreamActionBuffer:
    """Action chunk queue with temporal smoothing."""

    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14,
                 smooth_method="temporal"):
        self.chunks = deque()
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()
        self.k = 0
        self.last_action = None

    def integrate_new_chunk(self, actions_chunk: np.ndarray,
                            max_k: int, min_m: int = 8):
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]

            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy()
                            for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if 0 < len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy()
                                    for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return

            new_list = list(new_chunk)
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old

            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) +
                 w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(
                    self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act


# ---------------------------------------------------------------------------
# ROS 2 Observation Collector Node
# ---------------------------------------------------------------------------
class ObservationCollector(Node):
    """
    ROS 2 node that subscribes to:
      - 1 arm joint state topic  (sensor_msgs/JointState, 7-dim)
      - 2 camera image topics    (sensor_msgs/Image)

    Provides thread-safe access to the latest observations.
    """

    def __init__(self, args):
        super().__init__('kai0_inference_node')
        self.bridge = CvBridge()
        self.args = args

        # Latest data (protected by lock)
        self._lock = threading.Lock()
        self._img_head = None
        self._img_wrist = None
        self._joint_state = None      # JointState msg
        self._img_head_stamp = 0.0
        self._img_wrist_stamp = 0.0
        self._joint_stamp = 0.0

        # QoS: best-effort for camera topics, reliable for joint states
        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscriptions
        self.create_subscription(
            Image, args.img_head_topic,
            self._img_head_cb, cam_qos)
        self.create_subscription(
            Image, args.img_wrist_topic,
            self._img_wrist_cb, cam_qos)
        self.create_subscription(
            JointState, args.joint_states_topic,
            self._joint_cb, joint_qos)

        # Publisher for joint commands
        self.joint_cmd_pub = self.create_publisher(
            JointState, args.joint_cmd_topic, 10)

        self.get_logger().info(
            f"Subscribing to: head={args.img_head_topic}, "
            f"wrist={args.img_wrist_topic}, "
            f"joints={args.joint_states_topic}")
        self.get_logger().info(
            f"Publishing cmds to: {args.joint_cmd_topic}")

    def _stamp_to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _img_head_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._lock:
            self._img_head = img
            self._img_head_stamp = self._stamp_to_sec(msg.header.stamp)

    def _img_wrist_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._lock:
            self._img_wrist = img
            self._img_wrist_stamp = self._stamp_to_sec(msg.header.stamp)

    def _joint_cb(self, msg: JointState):
        with self._lock:
            self._joint_state = msg
            self._joint_stamp = self._stamp_to_sec(msg.header.stamp)

    def get_observation(self):
        """
        Return (img_head, img_wrist, joint_positions) or None if any is
        missing. joint_positions is a list of 7 floats.
        """
        with self._lock:
            if (self._img_head is None or self._img_wrist is None
                    or self._joint_state is None):
                return None
            return (
                self._img_head.copy(),
                self._img_wrist.copy(),
                list(self._joint_state.position),
            )

    def publish_joint_cmd(self, positions):
        """Publish a 7-dim joint command."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3',
                     'joint4', 'joint5', 'joint6', 'gripper']
        msg.position = [float(p) for p in positions]
        self.joint_cmd_pub.publish(msg)


# ---------------------------------------------------------------------------
# Observation window helpers
# ---------------------------------------------------------------------------
def jpeg_mapping(img):
    """JPEG encode→decode to match training-time augmentation."""
    img = cv2.imencode(".jpg", img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    return img


def update_observation_window(obs_collector: ObservationCollector):
    """
    Poll the ObservationCollector and append to the global
    observation_window deque.  Returns True on success.
    """
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
        observation_window.append({
            "qpos": None,
            "img_head": None,
            "img_wrist": None,
        })

    obs = obs_collector.get_observation()
    if obs is None:
        return False

    img_head, img_wrist, joint_pos = obs
    img_head = jpeg_mapping(img_head)
    img_wrist = jpeg_mapping(img_wrist)

    # 7-dim → duplicated to 14-dim for model
    qpos_single = np.array(joint_pos[:STATE_DIM_SINGLE])
    qpos = np.concatenate([qpos_single, qpos_single], axis=0)

    observation_window.append({
        "qpos": qpos,
        "img_head": img_head,
        "img_wrist": img_wrist,
    })
    return True


# ---------------------------------------------------------------------------
# Build inference payload
# ---------------------------------------------------------------------------
def build_payload(obs_entry):
    """
    Build the dict payload for the policy server from one observation entry.
    """
    img_head  = obs_entry["img_head"]
    img_wrist = obs_entry["img_wrist"]
    dummy     = _make_dummy_image()

    imgs = [img_head, img_wrist, dummy]
    imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
    imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)

    return {
        "state": obs_entry["qpos"],
        "images": {
            "top_head":   imgs[0].transpose(2, 0, 1),   # CHW
            "hand_right": imgs[1].transpose(2, 0, 1),   # wrist → hand_right
            "hand_left":  imgs[2].transpose(2, 0, 1),   # dummy
        },
        "prompt": lang_embeddings,
    }


# ---------------------------------------------------------------------------
# Inference thread (non-blocking, temporal smoothing)
# ---------------------------------------------------------------------------
def inference_thread_fn(args, obs_collector, policy):
    """
    Background thread: polls latest observation, calls policy.infer(),
    pushes results into stream_buffer.
    """
    global stream_buffer

    period = 1.0 / max(0.1, args.inference_rate)
    while not shutdown_event.is_set():
        try:
            t0 = time.time()
            ok = update_observation_window(obs_collector)
            if not ok:
                time.sleep(0.01)
                continue

            dt_obs = time.time() - t0
            print(f"Get Observation Time {dt_obs:.3f} s")

            latest = observation_window[-1]
            payload = build_payload(latest)

            t1 = time.time()
            actions = policy.infer(payload)["actions"]
            print(f"Inference Time {time.time() - t1:.3f} s")

            if actions is not None and len(actions) > 0:
                max_k = int(args.latency_k)
                min_m = int(args.min_smooth_steps)
                stream_buffer.integrate_new_chunk(
                    actions, max_k=max_k, min_m=min_m)
            else:
                print("actions is None or empty")

            # Throttle to inference_rate
            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)

        except Exception as e:
            print(f"[inference_thread] {e}")
            time.sleep(0.05)


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------
def run_inference_loop(args, obs_collector: ObservationCollector):
    global stream_buffer, lang_embeddings

    # Connect to policy server
    policy = websocket_client_policy.WebsocketClientPolicy(
        args.host, args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    # Determine action slice
    if args.use_right_arm_action:
        arm_slice = slice(7, 14)
        print("[single-arm] Using RIGHT arm action slot (dims 7:14)")
    else:
        arm_slice = slice(0, 7)
        print("[single-arm] Using LEFT arm action slot (dims 0:7)")

    # Move arm to initial position
    arm0 = [0.0, 0.32, -0.36, 0.0, 0.24, 0.0, 0.07]
    print("Publishing initial arm position...")
    for _ in range(50):  # publish for ~1 sec at 50 Hz
        obs_collector.publish_joint_cmd(arm0)
        time.sleep(0.02)

    input("Press Enter to start inference...")
    for _ in range(20):
        obs_collector.publish_joint_cmd(arm0)
        time.sleep(0.02)

    # Warmup: wait for first observation
    print("Waiting for camera + arm topics...")
    while not shutdown_event.is_set():
        ok = update_observation_window(obs_collector)
        if ok:
            break
        time.sleep(0.1)

    # Warmup inference
    try:
        latest = observation_window[-1]
        payload = build_payload(latest)
        _ = policy.infer(payload)
        print("Warmup inference done.")
    except Exception as e:
        print(f"[warmup] {e}")

    # Initialize stream buffer and inference thread
    if args.use_temporal_smoothing:
        stream_buffer = StreamActionBuffer(
            max_chunks=args.buffer_max_chunks,
            decay_alpha=args.exp_decay_alpha,
            state_dim=STATE_DIM_MODEL,
            smooth_method="temporal",
        )
        inf_thread = threading.Thread(
            target=inference_thread_fn,
            args=(args, obs_collector, policy),
            daemon=True,
        )
        inf_thread.start()
    else:
        print("ERROR: Only temporal smoothing mode is supported in this "
              "script. Use --use_temporal_smoothing.")
        return

    # Publish loop
    publish_period = 1.0 / args.publish_rate
    t = 0
    with torch.inference_mode():
        while t < args.max_publish_step and not shutdown_event.is_set():
            act = stream_buffer.pop_next_action()
            if act is not None:
                arm_action = act[arm_slice].copy()
                arm_action[6] = max(0.0, arm_action[6] - RIGHT_OFFSET)
                obs_collector.publish_joint_cmd(arm_action.tolist())
                published_actions_history.append(arm_action.astype(float))
                print(f"Published Step {t}")
                t += 1
            else:
                time.sleep(0.001)
                continue

            time.sleep(publish_period)

    print(f"Inference loop finished after {t} steps.")


# ---------------------------------------------------------------------------
# ROS 2 spin helper
# ---------------------------------------------------------------------------
def spin_node_in_background(node: Node):
    """Spin a ROS 2 node in a daemon thread."""
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_arguments():
    parser = argparse.ArgumentParser(
        description="ROS 2 single-arm + 2-camera Kai0 inference client")

    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)

    # Camera topics
    parser.add_argument("--img_head_topic", type=str,
                        default="/camera_head/camera_head/color/image_raw",
                        help="Head/overhead camera topic (D456)")
    parser.add_argument("--img_wrist_topic", type=str,
                        default="/camera_wrist/camera_wrist/color/image_raw",
                        help="Wrist camera topic (D405)")

    # Arm topics (Piper ROS 2)
    parser.add_argument("--joint_states_topic", type=str,
                        default="/joint_states_single",
                        help="Arm joint state feedback topic")
    parser.add_argument("--joint_cmd_topic", type=str,
                        default="/joint_ctrl_single",
                        help="Arm joint command topic")

    # Server
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    # Control
    parser.add_argument("--publish_rate", type=int, default=30,
                        help="Action publish rate (Hz)")
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--arm_steps_length", type=float, nargs="+",
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2])

    # Temporal smoothing
    parser.add_argument("--use_temporal_smoothing", action="store_true",
                        default=False)
    parser.add_argument("--latency_k", type=int, default=8)
    parser.add_argument("--inference_rate", type=float, default=3.0,
                        help="Inference loop rate (Hz)")
    parser.add_argument("--min_smooth_steps", type=int, default=8)
    parser.add_argument("--buffer_max_chunks", type=int, default=10)
    parser.add_argument("--exp_decay_alpha", type=float, default=0.25)

    # Single-arm
    parser.add_argument("--use_right_arm_action", action="store_true",
                        default=False,
                        help="Use right-arm slot (dims 7:14) from model")

    # Prompt
    parser.add_argument("--prompt", type=str, default=None,
                        help="Language prompt (overrides lang_embeddings)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global lang_embeddings

    args = get_arguments()

    if args.prompt is not None:
        lang_embeddings = args.prompt
        print(f"Using prompt: {lang_embeddings}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Initialize ROS 2
    rclpy.init()

    # Create observation collector node
    obs_collector = ObservationCollector(args)

    # Spin the node in background so callbacks fire
    spin_thread = spin_node_in_background(obs_collector)

    # Handle SIGINT
    def _on_sigint(signum, frame):
        shutdown_event.set()
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        run_inference_loop(args, obs_collector)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        obs_collector.destroy_node()
        rclpy.try_shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
