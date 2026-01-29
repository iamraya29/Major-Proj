#!/usr/bin/env python3
# corrected_cadrl_node.py  (debug-safe version)

import sys, os, inspect, configparser
import torch
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion

# === PATH SETUP ===
sys.path.insert(0, os.path.expanduser('~/Social-Navigation-PyEnvs/social_gym/src'))
sys.path.append(os.path.expanduser('~/Social-Navigation-PyEnvs/crowd_nav'))
sys.path.append('/home/raya/Social-Navigation-PyEnvs/crowd_nav')
sys.path.append('/home/raya/Social-Navigation-PyEnvs/social_gym/src')

from test_pkg.crowd_nav.policy.policy_factory import policy_factory
from test_pkg.social_gym.src.utils import PRECISION
from test_pkg.crowd_nav.utils.state import ObservableState, FullState, JointState

# === CONSTANTS ===
MODEL_PATH = "/home/raya/Social-Navigation-PyEnvs/crowd_nav/data/output/rl_model.pth"
ROBOT_RADIUS = 0.3
V_PREF = 1.0


# === CADRL POLICY WRAPPER ===
class CADRLPolicyWrapper:
    def __init__(self, device='cpu'):
        self.device = device
        model_class = policy_factory.get('cadrl')
        if model_class is None:
            raise RuntimeError("policy_factory['cadrl'] not found.")
        self.model = model_class()

        # --- Configuration ---
        cfg = configparser.ConfigParser()
        cfg['rl'] = {'gamma': '0.9'}
        cfg['action_space'] = {
            'kinematics': 'holonomic', 'sampling': 'exponential',
            'speed_samples': '5', 'rotation_samples': '16', 'query_env': 'true'
        }
        cfg['cadrl'] = {
            'mlp_dims': '150 100 100 1',  # fixed: use spaces instead of commas
            'multiagent_training': 'false',
            'with_theta_and_omega_visible': 'false'
        }

        cfg['om'] = {'cell_num': '4', 'cell_size': '1.0', 'om_channel_size': '3'}
        cfg['env'] = {
            'time_limit': '50', 'time_step': '0.0125',
            'robot_time_step': '0.25', 'human_num': '5'
        }

        try:
            self.model.configure(cfg)
        except Exception as e:
            print(f"[CADRLPolicyWrapper] configure() failed: {e}")

        if hasattr(self.model, 'set_device'):
            try: self.model.set_device(device)
            except Exception: pass

        self.model.phase = 'test'
        self.model.time_step = 0.0125

        # --- Load weights ---
        if os.path.exists(MODEL_PATH):
            try:
                weights = torch.load(MODEL_PATH, map_location=device)
                if hasattr(self.model, 'load_state_dict'):
                    self.model.load_state_dict(weights)
                    print(f"Loaded CADRL weights from {MODEL_PATH}")
                elif hasattr(self.model, 'get_model'):
                    self.model.get_model().load_state_dict(weights)
                    print(f"Loaded CADRL weights into inner model from {MODEL_PATH}")
            except Exception as e:
                print(f"[CADRLPolicyWrapper] Weight load failed: {e}")
        else:
            print(f"[CADRLPolicyWrapper] No weights at {MODEL_PATH}")

    # === Predict Action ===
    def predict_action(self, robot_state, humans_state):
        # Ensure numeric numpy array, replace None→0.0
        robot_state = np.array([0.0 if x is None else x for x in robot_state], dtype=float)

        # Build FullState safely
        try:
            r_state = FullState(
                px=robot_state[0],
                py=robot_state[1],
                vx=robot_state[3] if len(robot_state) > 3 else 0.0,
                vy=robot_state[4] if len(robot_state) > 4 else 0.0,
                radius=ROBOT_RADIUS,
                gx=robot_state[6] if len(robot_state) > 6 else 0.0,
                gy=robot_state[7] if len(robot_state) > 7 else 0.0,
                v_pref=V_PREF,
                theta=robot_state[2] if len(robot_state) > 2 else 0.0
            )
        except Exception as e:
            print(f"[CADRLPolicyWrapper] FullState failed: {e}")
            r_state = ObservableState(robot_state[0], robot_state[1],
                                      robot_state[3] if len(robot_state) > 3 else 0.0,
                                      robot_state[4] if len(robot_state) > 4 else 0.0,
                                      ROBOT_RADIUS)

        # Build humans
        h_states = []
        for h in humans_state:
            h = np.array([0.0 if x is None else x for x in h], dtype=float)
            try:
                h_states.append(ObservableState(h[0], h[1], h[2], h[3], h[4]))
            except Exception as e:
                print(f"[CADRLPolicyWrapper] Bad human state {h}: {e}")
                h_states.append(ObservableState(h[0], h[1], 0.0, 0.0, 0.3))

        joint = JointState(r_state, h_states)

        # === DEBUG PRINTS ===
        print("\n[DEBUG] --- JointState contents ---")
        print("[DEBUG] Robot FullState:")
        for k, v in r_state.__dict__.items():
            print(f"   {k}: {v}")
        print("[DEBUG] Humans:")
        for i, h in enumerate(h_states):
            print(f"   Human {i}: {h.__dict__}")
        print("[DEBUG] -----------------------------\n")

        # Predict
        action = self.model.predict(joint)

        if hasattr(action, 'vx') and hasattr(action, 'vy'):
            return action.vx, action.vy
        elif isinstance(action, (list, tuple, np.ndarray)) and len(action) >= 2:
            return float(action[0]), float(action[1])
        else:
            raise RuntimeError(f"Unexpected action type: {type(action)}")


# === ROS2 NODE ===
class CADRLNode(Node):
    def __init__(self):
        super().__init__('cadrl_node')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.25, self.control_loop)

        self.wrapper = CADRLPolicyWrapper(device='cpu')
        self.policy = getattr(self.wrapper, 'model', None)
        if self.policy:
            if hasattr(self.policy, 'set_env'):
                try: self.policy.set_env(self.wrapper)
                except Exception: pass
            self.policy.phase = 'test'
            if hasattr(self.policy, 'set_phase'):
                try: self.policy.set_phase('test')
                except Exception: pass
            if hasattr(self.policy, 'set_device'):
                try: self.policy.set_device(torch.device('cpu'))
                except Exception: pass

        self.robot_state = None
        self.humans_state = []
        self.get_logger().info("CADRL Node Initialized with wrapper-based policy.")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        yaw = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w])[2]

        vel, omega = msg.twist.twist.linear, msg.twist.twist.angular
        vx = vel.x if vel.x is not None else 0.0
        vy = vel.y if vel.y is not None else 0.0
        omega_z = omega.z if omega.z is not None else 0.0

        self.robot_state = np.array(
            [pos.x, pos.y, yaw, vx, vy, omega_z, 1.5, 1.5], dtype=float)

        # Dummy human (px,py,vx,vy,radius)
        self.humans_state = [
            np.array([pos.x + 1.0, pos.y, 0.0, 0.0, 0.3], dtype=float)
        ]

    def control_loop(self):
        if self.robot_state is None or not self.humans_state:
            return
        try:
            vx, vy = self.wrapper.predict_action(self.robot_state, self.humans_state)
        except Exception as e:
            self.get_logger().error(f"predict_action failed: {e}")
            return

        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = 0.0
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CADRLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
