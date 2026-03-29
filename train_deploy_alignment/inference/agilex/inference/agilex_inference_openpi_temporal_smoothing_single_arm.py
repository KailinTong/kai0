# -- coding: UTF-8
"""
Single-arm + 2-camera adaptation of the Kai0 Agilex inference client.

This script is a self-contained "fastest hack" adapter that lets you test the
Kai0 server/client inference pipeline with:
  - 1 AgileX Piper arm  (7-DoF joint state)
  - 2 cameras           (overhead + wrist)

It keeps the pretrained model's I/O schema intact by:
  1. Duplicating the 7-dim proprioception into the 14-dim state vector.
  2. Injecting a black dummy image for the missing third camera.
  3. Extracting only one arm's 7-dim slice from the 14-dim action output.

WARNING: The pretrained dual-arm checkpoint will NOT produce meaningful actions
for a single-arm setup.  Retraining on single-arm data is required for real
performance.  This script is for pipeline-structure testing only.

Original file: agilex_inference_openpi_temporal_smoothing.py
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
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from openpi_client import image_tools, websocket_client_policy
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants adapted for single-arm + 2 cameras
# ---------------------------------------------------------------------------
CAMERA_NAMES = ["cam_high", "cam_wrist"]   # only 2 real cameras

STATE_DIM_SINGLE = 7    # one Piper arm
STATE_DIM_MODEL  = 14   # what the pretrained model expects

DUMMY_IMG_SHAPE = (224, 224, 3)   # black image for the missing camera

stream_buffer = None   # type: StreamActionBuffer

observation_window = None

lang_embeddings = "fold the sleeve"

RIGHT_OFFSET = 0.003
published_actions_history = []
observed_qpos_history = []
publish_step_global = 0
inferred_chunks = []
inferred_chunks_lock = threading.Lock()
shutdown_event = threading.Event()


# ---------------------------------------------------------------------------
# Helper: build a black dummy image (uint8)
# ---------------------------------------------------------------------------
def _make_dummy_image():
    """Return a 224×224×3 black image in uint8 BGR format."""
    return np.zeros(DUMMY_IMG_SHAPE, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Inference thread (non-blocking, temporal smoothing)
# ---------------------------------------------------------------------------
def inference_fn_non_blocking_fast(args, config, policy, ros_operator):
    """
    Non-blocking inference thread adapted for single-arm + 2 cameras.
    """
    global stream_buffer, observation_window, lang_embeddings

    rate = rospy.Rate(getattr(args, "inference_rate", 4))
    while not rospy.is_shutdown():
        try:
            time1 = time.time()
            update_observation_window(args, config, ros_operator)
            print("Get Observation Time", time.time() - time1, "s")
            time1 = time.time()

            latest_obs = observation_window[-1]

            # --- 2 real images + 1 dummy ---
            img_high  = latest_obs["images"][config["camera_names"][0]]
            img_wrist = latest_obs["images"][config["camera_names"][1]]
            dummy     = _make_dummy_image()

            imgs = [img_high, img_wrist, dummy]
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
            imgs = image_tools.resize_with_pad(np.array(imgs), 224, 224)

            # --- single-arm state duplicated to 14-dim ---
            proprio = latest_obs["qpos"]  # already 14-dim (duplicated)

            # Build payload matching the model's expected schema
            payload = {
                "state": proprio,
                "images": {
                    "top_head":   imgs[0].transpose(2, 0, 1),
                    "hand_right": imgs[1].transpose(2, 0, 1),  # wrist cam → hand_right
                    "hand_left":  imgs[2].transpose(2, 0, 1),  # dummy
                },
                "prompt": lang_embeddings,
            }

            actions = policy.infer(payload)["actions"]
            print("Inference Time", time.time() - time1, "s")
            time1 = time.time()

            if actions is not None and len(actions) > 0:
                max_k = int(getattr(args, "latency_k", 0))
                min_m = int(getattr(args, "min_smooth_steps", 8))
                stream_buffer.integrate_new_chunk(actions, max_k=max_k, min_m=min_m)
            elif actions is None:
                print("actions is None")
            elif len(actions) == 0:
                print("len(actions) == 0")

            print("Append Buffer Time", time.time() - time1, "s")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

        except Exception as e:
            rospy.logwarn(f"[inference_fn_non_blocking_fast] {e}")
            try:
                rate.sleep()
            except Exception:
                try:
                    time.sleep(0.001)
                except Exception:
                    pass
            continue


# ---------------------------------------------------------------------------
# StreamActionBuffer  (unchanged from the original — operates on 14-dim)
# ---------------------------------------------------------------------------
class StreamActionBuffer:
    """
    Maintains a queue of action chunks with temporal smoothing.
    Operates on the full 14-dim action space (model output).
    """
    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14, smooth_method="temporal"):
        self.chunks = deque()
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        self.cur_chunk = deque()
        self.k = 0
        self.last_action = None

    def push_chunk(self, actions_chunk: np.ndarray):
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            dq = deque([a.copy() for a in actions_chunk], maxlen=None)
            self.chunks.append(dq)
            while len(self.chunks) > self.max_chunks:
                self.chunks.popleft()

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
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
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
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
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def start_inference_thread(args, config, policy, ros_operator):
    t = threading.Thread(target=inference_fn_non_blocking_fast, args=(args, config, policy, ros_operator))
    t.daemon = True
    t.start()


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def interpolate_action(args, prev_action, cur_action):
    steps = np.array(args.arm_steps_length)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        "episode_len": args.max_publish_step,
        "state_dim": STATE_DIM_MODEL,   # still 14 for model compatibility
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES,   # 2 real cameras
    }
    return config


# ---------------------------------------------------------------------------
# ROS observation (single arm + 2 cameras)
# ---------------------------------------------------------------------------
def get_ros_observation(args, ros_operator):
    """Fetch the latest synced frame: 2 images + 1 arm."""
    rate = rospy.Rate(args.publish_rate)
    print_flag = True
    time3 = time.time()

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if time.time() - time3 > 0.01:
            print("Get Frame Time is too long", time.time() - time3, "s")
        if not result:
            if print_flag:
                print("sync fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_wrist, puppet_arm) = result
        return (img_front, img_wrist, puppet_arm)


def update_observation_window(args, config, ros_operator):
    """Update the sliding observation window (single-arm variant)."""
    def jpeg_mapping(img):
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
        observation_window.append(
            {
                "qpos": None,
                "images": {
                    config["camera_names"][0]: None,
                    config["camera_names"][1]: None,
                },
            }
        )

    img_front, img_wrist, puppet_arm = get_ros_observation(args, ros_operator)
    img_front = jpeg_mapping(img_front)
    img_wrist = jpeg_mapping(img_wrist)

    # Single-arm: 7-dim state, duplicated to 14-dim for model compatibility
    qpos_single = np.array(puppet_arm.position)   # shape (7,)
    qpos = np.concatenate([qpos_single, qpos_single], axis=0)  # shape (14,)

    observation_window.append(
        {
            "qpos": qpos,
            "images": {
                config["camera_names"][0]: img_front,
                config["camera_names"][1]: img_wrist,
            },
        }
    )


def inference_fn(args, config, policy):
    """Blocking single-shot inference (single-arm variant)."""
    global observation_window, lang_embeddings

    while True and not rospy.is_shutdown():
        img_high  = observation_window[-1]["images"][config["camera_names"][0]]
        img_wrist = observation_window[-1]["images"][config["camera_names"][1]]
        dummy     = _make_dummy_image()

        image_arrs = [img_high, img_wrist, dummy]
        image_arrs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_arrs]
        image_arrs = image_tools.resize_with_pad(np.array(image_arrs), 224, 224)

        proprio = observation_window[-1]["qpos"]

        payload = {
            "state": proprio,
            "images": {
                "top_head":   image_arrs[0].transpose(2, 0, 1),
                "hand_right": image_arrs[1].transpose(2, 0, 1),
                "hand_left":  image_arrs[2].transpose(2, 0, 1),
            },
            "prompt": lang_embeddings,
        }

        time1 = time.time()
        actions = policy.infer(payload)["actions"]
        print(f"Model inference time: {(time.time() - time1)*1000:.3f} ms")
        return actions


# ---------------------------------------------------------------------------
# Main control loop (single arm)
# ---------------------------------------------------------------------------
def model_inference(args, config, ros_operator):
    global lang_embeddings, stream_buffer

    policy = websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")

    max_publish_step = config["episode_len"]
    chunk_size = config["chunk_size"]

    # Determine which 7-dim slice to use from the 14-dim model output
    if args.use_right_arm_action:
        arm_slice = slice(7, 14)
        print("[single-arm] Using RIGHT arm action slot (dims 7:14)")
    else:
        arm_slice = slice(0, 7)
        print("[single-arm] Using LEFT arm action slot (dims 0:7)")

    # Initial position for the single arm
    arm0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]

    ros_operator.puppet_arm_publish_single(arm0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_single(arm0)

    # Warmup inference
    try:
        update_observation_window(args, config, ros_operator)
        latest_obs = observation_window[-1]
        img_high  = latest_obs["images"][config["camera_names"][0]]
        img_wrist = latest_obs["images"][config["camera_names"][1]]
        dummy     = _make_dummy_image()
        image_arrs = [img_high, img_wrist, dummy]
        image_arrs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_arrs]
        image_arrs = image_tools.resize_with_pad(np.array(image_arrs), 224, 224)
        proprio = latest_obs["qpos"]
        payload = {
            "state": proprio,
            "images": {
                "top_head":   image_arrs[0].transpose(2, 0, 1),
                "hand_right": image_arrs[1].transpose(2, 0, 1),
                "hand_left":  image_arrs[2].transpose(2, 0, 1),
            },
            "prompt": lang_embeddings,
        }
        try:
            _ = policy.infer(payload)
        except Exception as e:
            rospy.logwarn(f"[startup_warmup_infer] {e}")
    except Exception as e:
        rospy.logwarn(f"[startup_warmup_prep] {e}")

    pre_action = np.zeros(config["state_dim"])
    action = None

    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            t = 0
            rate = rospy.Rate(args.publish_rate)
            action_buffer = np.zeros([chunk_size, config["state_dim"]])

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                if shutdown_event.is_set():
                    break
                if args.use_temporal_smoothing:
                    if stream_buffer is None:
                        stream_buffer = StreamActionBuffer(
                            max_chunks=args.buffer_max_chunks,
                            decay_alpha=args.exp_decay_alpha,
                            state_dim=config["state_dim"],
                            smooth_method="temporal",
                        )
                        start_inference_thread(args, config, policy, ros_operator)
                    act = stream_buffer.pop_next_action()
                    if act is not None:
                        if args.ctrl_type == "joint":
                            # Extract only the relevant 7-dim arm action
                            arm_action = act[arm_slice].copy()
                            arm_action[6] = max(0.0, arm_action[6] - RIGHT_OFFSET)
                            ros_operator.puppet_arm_publish_single(arm_action)
                            published_actions_history.append(arm_action.astype(float))
                        else:
                            print("Make sure ctrl_type is joint")
                    else:
                        print("act is None")
                        time.sleep(0.001)
                        continue
                    print("Published Step", t)
                    try:
                        publish_step_global = len(published_actions_history)
                    except Exception:
                        pass

                    rate.sleep()
                    t += 1
                else:
                    print("Make sure to use temporal smoothing")

                if shutdown_event.is_set():
                    break


# ---------------------------------------------------------------------------
# ROS operator (single arm + 2 cameras)
# ---------------------------------------------------------------------------
class RosOperator:
    def __init__(self, args):
        self.communication_thread = None
        self.communication_flag = False
        self.lock = threading.Lock()
        self.robot_base_deque = None
        self.puppet_arm_deque = None       # single arm
        self.img_front_deque = None
        self.img_wrist_deque = None        # single wrist camera
        self.img_front_depth_deque = None
        self.img_wrist_depth_deque = None
        self.bridge = None
        self.puppet_arm_publisher = None   # single arm publisher
        self.endpose_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_wrist_deque = deque()
        self.img_front_deque = deque()
        self.img_wrist_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish_single(self, positions):
        """Publish joint commands to the single arm."""
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = [
            "joint0", "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6",
        ]
        joint_state_msg.position = positions
        self.puppet_arm_publisher.publish(joint_state_msg)

    def endpose_publish_single(self, pose):
        """Publish end-effector pose to the single arm."""
        endpose_msg = PosCmd()
        endpose_msg.x, endpose_msg.y, endpose_msg.z = pose[:3]
        endpose_msg.roll, endpose_msg.pitch, endpose_msg.yaw = pose[3:6]
        endpose_msg.gripper = pose[6]
        self.endpose_publisher.publish(endpose_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, target):
        """Move the single arm smoothly to a target position."""
        rate = rospy.Rate(self.args.publish_rate)
        arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_deque) != 0:
                arm = list(self.puppet_arm_deque[-1].position)
            if arm is None:
                rate.sleep()
                continue
            else:
                break

        symbol = [1 if target[i] - arm[i] > 0 else -1 for i in range(len(target))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            diff = [abs(target[i] - arm[i]) for i in range(len(target))]
            flag = False
            for i in range(len(target)):
                if diff[i] < self.args.arm_steps_length[i]:
                    arm[i] = target[i]
                else:
                    arm[i] += symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = [
                "joint0", "joint1", "joint2", "joint3",
                "joint4", "joint5", "joint6",
            ]
            joint_state_msg.position = arm
            self.puppet_arm_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, target):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(
            target=self.puppet_arm_publish_continuous, args=(target,)
        )
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        """Synchronise and return the latest frame (2 images + 1 arm)."""
        if (
            len(self.img_front_deque) == 0
            or len(self.img_wrist_deque) == 0
            or len(self.puppet_arm_deque) == 0
        ):
            return False

        if self.args.use_depth_image and (
            len(self.img_front_depth_deque) == 0
            or len(self.img_wrist_depth_deque) == 0
        ):
            return False

        if self.args.use_depth_image:
            frame_time = min([
                self.img_wrist_deque[-1].header.stamp.to_sec(),
                self.img_front_deque[-1].header.stamp.to_sec(),
                self.img_wrist_depth_deque[-1].header.stamp.to_sec(),
                self.img_front_depth_deque[-1].header.stamp.to_sec(),
            ])
        else:
            frame_time = min([
                self.img_wrist_deque[-1].header.stamp.to_sec(),
                self.img_front_deque[-1].header.stamp.to_sec(),
            ])

        if len(self.img_wrist_deque) == 0 or self.img_wrist_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_deque) == 0 or self.puppet_arm_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_robot_base and (
            len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time
        ):
            return False

        while self.img_wrist_deque[0].header.stamp.to_sec() < frame_time:
            self.img_wrist_deque.popleft()
        img_wrist = self.bridge.imgmsg_to_cv2(self.img_wrist_deque.popleft(), "passthrough")

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), "passthrough")

        while self.puppet_arm_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_deque.popleft()
        puppet_arm = self.puppet_arm_deque.popleft()

        return (img_front, img_wrist, puppet_arm)

    # --- ROS callbacks ---
    def img_wrist_callback(self, msg):
        if len(self.img_wrist_deque) >= 2000:
            self.img_wrist_deque.popleft()
        self.img_wrist_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_wrist_depth_callback(self, msg):
        if len(self.img_wrist_depth_deque) >= 2000:
            self.img_wrist_depth_deque.popleft()
        self.img_wrist_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_callback(self, msg):
        if len(self.puppet_arm_deque) >= 2000:
            self.puppet_arm_deque.popleft()
        self.puppet_arm_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node("joint_state_publisher_single_arm", anonymous=True)
        # 2 cameras
        rospy.Subscriber(
            self.args.img_front_topic, Image,
            self.img_front_callback, queue_size=1000, tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.img_wrist_topic, Image,
            self.img_wrist_callback, queue_size=1000, tcp_nodelay=True,
        )
        if self.args.use_depth_image:
            rospy.Subscriber(
                self.args.img_front_depth_topic, Image,
                self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True,
            )
            rospy.Subscriber(
                self.args.img_wrist_depth_topic, Image,
                self.img_wrist_depth_callback, queue_size=1000, tcp_nodelay=True,
            )
        # 1 arm
        rospy.Subscriber(
            self.args.puppet_arm_topic, JointState,
            self.puppet_arm_callback, queue_size=1000, tcp_nodelay=True,
        )
        # Base
        rospy.Subscriber(
            self.args.robot_base_topic, Odometry,
            self.robot_base_callback, queue_size=1000, tcp_nodelay=True,
        )
        # Publishers (single arm)
        self.puppet_arm_publisher = rospy.Publisher(
            self.args.puppet_arm_cmd_topic, JointState, queue_size=10,
        )
        self.endpose_publisher = rospy.Publisher(
            self.args.endpose_cmd_topic, PosCmd, queue_size=10,
        )
        self.robot_base_publisher = rospy.Publisher(
            self.args.robot_base_cmd_topic, Twist, queue_size=10,
        )


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def get_arguments():
    parser = argparse.ArgumentParser(
        description="Single-arm + 2-camera Kai0 Agilex inference client"
    )
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)

    # Camera topics (2 cameras)
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw",
                        help="Overhead / front camera topic")
    parser.add_argument("--img_wrist_topic", type=str, default="/camera_r/color/image_raw",
                        help="Wrist-mounted camera topic")
    parser.add_argument("--img_front_depth_topic", type=str, default="/camera_f/depth/image_raw")
    parser.add_argument("--img_wrist_depth_topic", type=str, default="/camera_r/depth/image_raw")

    # Single arm topics
    parser.add_argument("--puppet_arm_cmd_topic", type=str, default="/master/joint_right",
                        help="Command topic for the single puppet arm")
    parser.add_argument("--puppet_arm_topic", type=str, default="/puppet/joint_right",
                        help="State topic for the single puppet arm")
    parser.add_argument("--endpose_cmd_topic", type=str, default="/pos_cmd_right",
                        help="End-effector pose command topic")

    # Base
    parser.add_argument("--robot_base_topic", type=str, default="/odom_raw")
    parser.add_argument("--robot_base_cmd_topic", type=str, default="/cmd_vel")
    parser.add_argument("--use_robot_base", action="store_true", default=False)

    # Control
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--arm_steps_length", type=float, nargs="+",
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],
                        help="Max joint change per timestep (7 values for single arm)")
    parser.add_argument("--use_actions_interpolation", action="store_true", default=False)
    parser.add_argument("--use_depth_image", action="store_true", default=False)

    # Server
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ctrl_type", type=str, choices=["joint", "eef"], default="joint")

    # Temporal smoothing
    parser.add_argument("--use_temporal_smoothing", action="store_true", default=False)
    parser.add_argument("--latency_k", type=int, default=8)
    parser.add_argument("--inference_rate", type=float, default=3.0)
    parser.add_argument("--min_smooth_steps", type=int, default=8)
    parser.add_argument("--buffer_max_chunks", type=int, default=10)
    parser.add_argument("--exp_decay_alpha", type=float, default=0.25)

    # Single-arm specific
    parser.add_argument("--use_right_arm_action", action="store_true", default=False,
                        help="Use dims 7:14 (right-arm slot) from model output instead of 0:7 (left-arm slot)")

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    signal.signal(signal.SIGINT, _on_sigint)
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
