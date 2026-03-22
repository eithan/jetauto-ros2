"""Safety monitor for JetAuto autonomous navigation.

Monitors lidar scans for dangerously close obstacles and publishes
zero-velocity commands to emergency-stop the robot.

Also provides a /safety_status topic for other nodes to check.

Features:
- Emergency stop when obstacle < min_obstacle_distance
- Publishes to /cmd_vel with zero twist to override navigation
- Logs warnings when obstacles are close but not critical
- Publishes SafetyStatus to /safety_status
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class SafetyMonitor(Node):
    """Emergency stop and safety monitoring for autonomous navigation."""

    def __init__(self):
        super().__init__('safety_monitor')

        # Parameters
        self.declare_parameter('min_obstacle_distance', 0.15)  # 15cm
        self.declare_parameter('warn_obstacle_distance', 0.30)  # 30cm
        self.declare_parameter('check_frequency', 10.0)
        self.declare_parameter('estop_duration', 1.0)  # hold stop for 1s

        self.min_dist = self.get_parameter('min_obstacle_distance').value
        self.warn_dist = self.get_parameter('warn_obstacle_distance').value
        self.estop_duration = self.get_parameter('estop_duration').value

        # State
        self._latest_scan: Optional[LaserScan] = None
        self._estop_active = False
        self._estop_count = 0

        # Publishers
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._status_pub = self.create_publisher(String, '/safety_status', 10)

        # Subscribers
        self._scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        # Safety check timer
        freq = self.get_parameter('check_frequency').value
        self._check_timer = self.create_timer(1.0 / freq, self._safety_check)

        self.get_logger().info(
            f'Safety monitor active — e-stop at {self.min_dist}m, '
            f'warn at {self.warn_dist}m'
        )

    def _scan_callback(self, msg: LaserScan):
        """Store latest lidar scan."""
        self._latest_scan = msg

    def _safety_check(self):
        """Check for dangerously close obstacles."""
        if self._latest_scan is None:
            return

        scan = self._latest_scan
        min_reading = float('inf')

        # Find closest valid reading
        for i, r in enumerate(scan.ranges):
            if scan.range_min <= r <= scan.range_max:
                if r < min_reading:
                    min_reading = r

        # Publish status
        status = String()
        if min_reading < self.min_dist:
            status.data = f'ESTOP:closest={min_reading:.3f}m'
        elif min_reading < self.warn_dist:
            status.data = f'WARN:closest={min_reading:.3f}m'
        else:
            status.data = f'OK:closest={min_reading:.3f}m'
        self._status_pub.publish(status)

        # Emergency stop
        if min_reading < self.min_dist:
            if not self._estop_active:
                self._estop_active = True
                self._estop_count += 1
                self.get_logger().error(
                    f'🛑 EMERGENCY STOP — obstacle at {min_reading:.3f}m '
                    f'(threshold: {self.min_dist}m) [e-stop #{self._estop_count}]'
                )

            # Publish zero velocity
            stop = Twist()  # all zeros
            self._cmd_pub.publish(stop)

        elif min_reading < self.warn_dist:
            if self._estop_active:
                self.get_logger().info(
                    f'⚠️ E-stop cleared — closest obstacle now at {min_reading:.3f}m'
                )
                self._estop_active = False
            else:
                self.get_logger().debug(
                    f'⚠️ Close obstacle at {min_reading:.3f}m'
                )
        else:
            if self._estop_active:
                self.get_logger().info('✅ E-stop cleared — path clear')
                self._estop_active = False


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f'Safety monitor stopped. Total e-stops: {node._estop_count}'
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
