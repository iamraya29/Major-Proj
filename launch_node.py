#!/usr/bin/env python3
"""
Launch node - used to setup CADRL network and control turtlebot3 burger using it.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from test_pkg.social_gym.social_nav_sim import SocialNavSim
from test_pkg.social_gym.custom_config.config_ros_example import data
import numpy as np

class LaunchNode(Node):

	def __init__(self):
		super().__init__('launch_node')
		self.get_logger().info("CADRL LaunchNode started.")

		# Publisher and subscriber
		self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
		self.subscription = self.create_subscription(
			Odometry,
			'/odom',
			self.odom_callback,
			10
		)

		# Timer for control loop
		self.timer = self.create_timer(0.25, self.timer_callback)

		# Internal state
		self.odom_msg = None

		######################## customization starts below #####################
		np.random.seed(1002)
		self.social_nav = SocialNavSim(config_data = {"insert_robot": True, "human_policy": "hsfm_new_guo", "headless": False,
												"runge_kutta": False, "robot_visible": True, "robot_radius": 0.3,
												"circle_radius": 7, "n_actors": 0, "randomize_human_positions": True, "randomize_human_attributes": False},
								scenario="circular_crossing", parallelize_robot = False, parallelize_humans = False)
		TIME_STEP = 1/100
		self.social_nav.set_time_step(TIME_STEP)
		self.social_nav.set_robot_time_step(1/4)
		self.social_nav.set_robot_policy(policy_name="orca", runge_kutta=False)# use human motion model as robot policy for now
		# self.social_nav.robot.policy.query_env = False # only for crowdnav policies
		self.get_logger().info("One time setup finished.")

		self.social_nav.run_live()
		##########################################################################

	def odom_callback(self, msg):
		"""Handle incoming odometry messages."""
		self.odom_msg = msg
		# self.get_logger().info("Received odometry message.")

	def timer_callback(self):
		"""Called periodically to publish velocity command."""
		if self.odom_msg is None:
			return
		cmd = Twist()
		cmd.linear.x = 0.0
		cmd.angular.z = 0.0
		self.publisher_.publish(cmd)


def main(args=None):
	rclpy.init(args=args)
	node = LaunchNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
