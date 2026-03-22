"""Safety monitor for JetAuto autonomous navigation.

Monitors lidar scans for dangerously close obstacles and:
1. Publishes zero-velocity commands at HIGH FREQUENCY to override Nav2
2. Cancels the active Nav2 goal so it stops trying to drive
3. Publishes /safety_status for other nodes to check

The key insight: just publishing zero velocity doesn't work because Nav2
republishes cmd_vel immediately. We must CANCEL the goal too.
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose


class SafetyMonitor(Node):
    """Emergency stop and safety monitoring for autonomous navigation."""

    def __init__(self):
        super().__init__('safety_monitor')

        # Parameters
        self.declare_parameter('min_obstacle_distance', 0.20)  # 20cm (was 15cm — too tight)
        self.declare_parameter('warn_obstacle_distance', 0.35)  # 35cm
        self.declare_parameter('check_frequency', 20.0)  # 20Hz (was 10Hz — need to beat Nav2)
        self.declare_parameter('resume_distance', 0.40)  # must back off to 40cm before clearing

        self.min_dist = self.get_parameter('min_obstacle_distance').value
        self.warn_dist = self.get_parameter('warn_obstacle_distance').value
        self.resume_dist = self.get_parameter('resume_distance').value

        # State
        self._latest_scan: Optional[LaserScan] = None
        self._estop_active = False
        self._estop_count = 0
        self._goal_canceled = False

        # Publishers — cmd_vel at high QoS priority
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._status_pub = self.create_publisher(String, '/safety_status', 10)

        # Subscribers
        self._scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        # Nav2 action client — to cancel goals
        self._nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=ReentrantCallbackGroup(),
        )

        # Safety check timer — runs at 20Hz
        freq = self.get_parameter('check_frequency').value
        self._check_timer = self.create_timer(1.0 / freq, self._safety_check)

        self.get_logger().info(
            f'Safety monitor active — e-stop at {self.min_dist}m, '
            f'warn at {self.warn_dist}m, resume at {self.resume_dist}m'
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
            # Always publish zero velocity when in e-stop zone
            stop = Twist()  # all zeros
            self._cmd_pub.publish(stop)

            if not self._estop_active:
                self._estop_active = True
                self._goal_canceled = False
                self._estop_count += 1
                self.get_logger().error(
                    f'🛑 EMERGENCY STOP #{self._estop_count} — '
                    f'obstacle at {min_reading:.3f}m (threshold: {self.min_dist}m)'
                )

            # Cancel Nav2 goal (only once per e-stop event)
            if not self._goal_canceled:
                self._cancel_nav2_goals()
                self._goal_canceled = True

        elif self._estop_active:
            # Use hysteresis — don't clear until obstacle is well clear
            if min_reading >= self.resume_dist:
                self.get_logger().info(
                    f'✅ E-stop cleared — closest obstacle at {min_reading:.3f}m '
                    f'(resume threshold: {self.resume_dist}m)'
                )
                self._estop_active = False
            else:
                # Still too close — keep sending zero velocity
                stop = Twist()
                self._cmd_pub.publish(stop)

    def _cancel_nav2_goals(self):
        """Cancel all active Nav2 goals."""
        self.get_logger().warn('Canceling all Nav2 goals...')
        try:
            cancel_future = self._nav_client.cancel_all_goals_async()
            cancel_future.add_done_callback(self._cancel_done)
        except Exception as e:
            self.get_logger().warn(f'Could not cancel Nav2 goals: {e}')

    def _cancel_done(self, future):
        """Callback for goal cancellation."""
        try:
            result = future.result()
            self.get_logger().info('Nav2 goals canceled')
        except Exception as e:
            self.get_logger().warn(f'Goal cancel result error: {e}')


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
