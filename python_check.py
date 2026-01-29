#!/usr/bin/env/python3
## this will be overriden with /home/krutarth/miniconda3/envs/gymrlenv/bin/python if setup cfg picks the first python3 in /usr/bin/env which is the case after activating conda env

import sys
import rclpy
from rclpy.node import Node

class HelloPython(Node):
    def __init__(self):
        super().__init__('hello_python')
        self.get_logger().info(f"Python version: {sys.version}")
        self.get_logger().info(f"Python executable: {sys.executable}")
        self.create_timer(2.0, self.say_hello)

    def say_hello(self):
        self.get_logger().info("Hello from Miniconda Python!")

def main(args=None):
    rclpy.init(args=args)
    node = HelloPython()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
