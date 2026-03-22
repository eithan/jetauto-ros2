"""Safety monitor for JetAuto autonomous navigation.

Monitors lidar scans for dangerously close obstacles and:
1. Publishes zero-velocity commands at HIGH FREQUENCY to override Nav2
2. Cancels the active Nav2 goal so it stops trying to drive
3. Publishes /safety_status for other nodes to check

The key insight: just publishing zero velocity doesn't work because Nav2
republishes cmd_vel immediately. We must CANCEL the goal too.
"""

import math
import time
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from action_msgs.srv import CancelGoal
from tf2_ros import Buffer, TransformListener


class SafetyMonitor(Node):
    """Emergency stop and safety monitoring for autonomous navigation."""

    def __init__(self):
        super().__init__('safety_monitor')

        # Parameters
        self.declare_parameter('min_obstacle_distance', 0.20)  # 20cm
        self.declare_parameter('warn_obstacle_distance', 0.35)  # 35cm
        self.declare_parameter('check_frequency', 20.0)  # 20Hz
        self.declare_parameter('resume_distance', 0.40)  # must back off to 40cm
        self.declare_parameter('stuck_timeout', 5.0)  # seconds without progress = stuck
        self.declare_parameter('stuck_move_threshold', 0.02)  # 2cm minimum movement

        self.min_dist = self.get_parameter('min_obstacle_distance').value
        self.warn_dist = self.get_parameter('warn_obstacle_distance').value
        self.resume_dist = self.get_parameter('resume_distance').value
        self.stuck_timeout = self.get_parameter('stuck_timeout').value
        self.stuck_move_threshold = self.get_parameter('stuck_move_threshold').value

        # How long to hold stuck e-stop before allowing resume
        self.declare_parameter('stuck_hold_time', 15.0)  # hold for 15 seconds
        self.stuck_hold_time = self.get_parameter('stuck_hold_time').value

        # State
        self._latest_scan: Optional[LaserScan] = None
        self._estop_active = False
        self._estop_count = 0
        self._goal_canceled = False

        # Stuck detection state (uses SLAM position, not odometry)
        self._last_cmd_vel: Optional[Twist] = None
        self._last_slam_position: Optional[Tuple[float, float]] = None
        self._last_move_time: float = time.time()
        self._stuck_estop_active = False
        self._stuck_estop_start: float = 0.0  # when stuck e-stop was triggered

        # TF listener for SLAM-based position (immune to wheel slip)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Publishers — cmd_vel at high QoS priority
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._status_pub = self.create_publisher(String, '/safety_status', 10)

        # Subscribers
        self._scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 10,
        )

        # Nav2 cancel service — to cancel goals (Humble compatible)
        self._cancel_client = self.create_client(
            CancelGoal, '/navigate_to_pose/_action/cancel_goal',
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

    def _cmd_vel_callback(self, msg: Twist):
        """Track velocity commands for stuck detection."""
        self._last_cmd_vel = msg

    def _get_slam_position(self) -> Optional[Tuple[float, float]]:
        """Get robot position from SLAM (map frame) — immune to wheel slip."""
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            return (t.transform.translation.x, t.transform.translation.y)
        except Exception:
            return None

    def _check_stuck(self):
        """Check if robot is stuck using SLAM position (not odometry)."""
        pos = self._get_slam_position()
        if pos is None:
            return

        if self._last_slam_position is not None:
            dx = pos[0] - self._last_slam_position[0]
            dy = pos[1] - self._last_slam_position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.stuck_move_threshold:
                self._last_move_time = time.time()
                # Only clear stuck if hold time has elapsed (prevent rapid on/off cycling)
                if self._stuck_estop_active:
                    held = time.time() - self._stuck_estop_start
                    if held >= self.stuck_hold_time:
                        self.get_logger().info(
                            f'✅ Stuck e-stop cleared — robot moved after {held:.0f}s hold'
                        )
                        self._stuck_estop_active = False

        self._last_slam_position = pos

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

        # ── Stuck detection (SLAM-based, immune to wheel slip) ──
        self._check_stuck()

        if self._last_cmd_vel is not None and not self._estop_active:
            cmd = self._last_cmd_vel
            is_commanded = (
                abs(cmd.linear.x) > 0.01 or
                abs(cmd.linear.y) > 0.01 or
                abs(cmd.angular.z) > 0.05
            )

            if is_commanded:
                time_since_move = time.time() - self._last_move_time
                if time_since_move > self.stuck_timeout and not self._stuck_estop_active:
                    self._stuck_estop_active = True
                    self._stuck_estop_start = time.time()
                    self._estop_count += 1
                    self.get_logger().error(
                        f'🛑 STUCK DETECTED #{self._estop_count} — '
                        f'no SLAM movement for {time_since_move:.1f}s despite motor commands. '
                        f'Canceling Nav2 goals.'
                    )
                    stop = Twist()
                    self._cmd_pub.publish(stop)
                    self._cancel_nav2_goals()

                if self._stuck_estop_active:
                    stop = Twist()
                    self._cmd_pub.publish(stop)

    def _cancel_nav2_goals(self):
        """Cancel all active Nav2 goals via cancel service."""
        self.get_logger().warn('Canceling all Nav2 goals...')
        try:
            if not self._cancel_client.service_is_ready():
                self.get_logger().warn('Cancel service not ready')
                return
            request = CancelGoal.Request()
            # Empty goal_info = cancel ALL goals
            future = self._cancel_client.call_async(request)
            future.add_done_callback(self._cancel_done)
        except Exception as e:
            self.get_logger().warn(f'Could not cancel Nav2 goals: {e}')

    def _cancel_done(self, future):
        """Callback for goal cancellation."""
        try:
            result = future.result()
            self.get_logger().info(f'Nav2 goals canceled (return_code={result.return_code})')
        except Exception as e:
            self.get_logger().warn(f'Goal cancel result error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(
            f'Safety monitor stopped. Total e-stops: {node._estop_count}'
        )
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
