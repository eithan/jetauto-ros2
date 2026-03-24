"""Safety monitor for JetAuto autonomous navigation.

Monitors lidar scans for dangerously close obstacles and:
1. Publishes zero-velocity commands at HIGH FREQUENCY to override Nav2
2. Cancels the active Nav2 goal so it stops trying to drive
3. Publishes /safety_status for other nodes to check
4. Executes recovery maneuvers (backup + strafe) when stuck

The key insight: just publishing zero velocity doesn't work because Nav2
republishes cmd_vel immediately. We must CANCEL the goal too.
"""

import math
import time
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PointStamped
from std_msgs.msg import String, Bool
from nav2_msgs.action import NavigateToPose
from action_msgs.srv import CancelGoal
from tf2_ros import Buffer, TransformListener


def _ts():
    """Human-readable timestamp for log messages."""
    return datetime.now().strftime('%H:%M:%S')


class SafetyMonitor(Node):
    """Emergency stop and safety monitoring for autonomous navigation."""

    # Recovery maneuver states
    RECOVERY_NONE = 0
    RECOVERY_BACKUP = 1
    RECOVERY_STRAFE = 2
    RECOVERY_HOLD = 3

    def __init__(self):
        super().__init__('safety_monitor')

        # Parameters
        self.declare_parameter('min_obstacle_distance', 0.20)  # 20cm
        self.declare_parameter('warn_obstacle_distance', 0.35)  # 35cm
        self.declare_parameter('check_frequency', 20.0)  # 20Hz
        self.declare_parameter('resume_distance', 0.40)  # must back off to 40cm
        self.declare_parameter('stuck_timeout', 5.0)  # seconds without progress = stuck
        self.declare_parameter('stuck_move_threshold', 0.02)  # 2cm minimum movement
        self.declare_parameter('stuck_hold_time', 15.0)  # hold for 15 seconds
        self.declare_parameter('startup_grace_period', 10.0)  # ignore stuck for first 10s
        self.declare_parameter('cmd_active_threshold', 5.0)  # commands must be active 5s before stuck counts
        self.declare_parameter('recovery_backup_speed', -0.20)  # m/s backward
        self.declare_parameter('recovery_strafe_speed', 0.25)  # m/s sideways
        self.declare_parameter('recovery_backup_duration', 2.0)  # seconds (~40cm backup)
        self.declare_parameter('recovery_strafe_duration', 1.5)  # seconds (~37cm strafe)

        self.min_dist = self.get_parameter('min_obstacle_distance').value
        self.warn_dist = self.get_parameter('warn_obstacle_distance').value
        self.resume_dist = self.get_parameter('resume_distance').value
        self.stuck_timeout = self.get_parameter('stuck_timeout').value
        self.stuck_move_threshold = self.get_parameter('stuck_move_threshold').value
        self.stuck_hold_time = self.get_parameter('stuck_hold_time').value
        self.startup_grace = self.get_parameter('startup_grace_period').value
        self.cmd_active_threshold = self.get_parameter('cmd_active_threshold').value
        self.recovery_backup_speed = self.get_parameter('recovery_backup_speed').value
        self.recovery_strafe_speed = self.get_parameter('recovery_strafe_speed').value
        self.recovery_backup_duration = self.get_parameter('recovery_backup_duration').value
        self.recovery_strafe_duration = self.get_parameter('recovery_strafe_duration').value

        # State
        self._latest_scan: Optional[LaserScan] = None
        self._estop_active = False
        self._estop_count = 0
        self._goal_canceled = False
        self._node_start_time = time.time()

        # Stuck detection state (uses SLAM position, not odometry)
        self._last_cmd_vel: Optional[Twist] = None
        self._last_slam_position: Optional[Tuple[float, float]] = None
        self._last_slam_yaw: Optional[float] = None
        self._last_move_time: float = time.time()
        self._stuck_estop_active = False
        self._stuck_estop_start: float = 0.0

        # Command activity tracking — when did continuous motor commands start?
        self._cmd_active_since: Optional[float] = None  # None = motors idle

        # Recovery maneuver state
        self._recovery_state = self.RECOVERY_NONE
        self._recovery_start: float = 0.0
        self._recovery_direction: int = 1  # +1 = strafe left, -1 = strafe right (alternates)
        self._recovery_count = 0

        # TF listener for SLAM-based position (immune to wheel slip)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Publishers
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._status_pub = self.create_publisher(String, '/safety_status', 10)
        # Publish stuck locations so frontier_explorer can avoid them
        self._stuck_pub = self.create_publisher(PointStamped, '/stuck_locations', 10)
        # Event publisher for dashboard announcements
        self._event_pub = self.create_publisher(String, '/explore/events', 10)

        # Subscribers
        self._scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 10,
        )

        # Nav2 cancel service
        self._cancel_client = self.create_client(
            CancelGoal, '/navigate_to_pose/_action/cancel_goal',
            callback_group=ReentrantCallbackGroup(),
        )

        # Safety check timer — runs at 20Hz
        freq = self.get_parameter('check_frequency').value
        self._check_timer = self.create_timer(1.0 / freq, self._safety_check)

        self.get_logger().info(
            f'[{_ts()}] Safety monitor active — e-stop at {self.min_dist}m, '
            f'warn at {self.warn_dist}m, resume at {self.resume_dist}m'
        )

    def _publish_event(self, text: str):
        """Publish a safety event for dashboard display/TTS."""
        msg = String()
        msg.data = text
        self._event_pub.publish(msg)

    def _scan_callback(self, msg: LaserScan):
        self._latest_scan = msg

    def _cmd_vel_callback(self, msg: Twist):
        self._last_cmd_vel = msg

    def _get_slam_pose(self) -> Optional[Tuple[float, float, float]]:
        """Get robot position + yaw from SLAM (map frame) — immune to wheel slip."""
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            # Extract yaw from quaternion
            q = t.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return (x, y, yaw)
        except Exception:
            return None

    def _check_stuck(self):
        """Check if robot is stuck using SLAM position + yaw (not odometry).

        Rotation-in-place counts as movement — the robot is not stuck if it's
        actively turning (e.g., Nav2 orienting at goal).
        """
        pose = self._get_slam_pose()
        if pose is None:
            return

        x, y, yaw = pose

        if self._last_slam_position is not None:
            dx = x - self._last_slam_position[0]
            dy = y - self._last_slam_position[1]
            dist = math.sqrt(dx * dx + dy * dy)

            # Check yaw change (handle wraparound)
            yaw_change = 0.0
            if self._last_slam_yaw is not None:
                yaw_change = abs(yaw - self._last_slam_yaw)
                if yaw_change > math.pi:
                    yaw_change = 2.0 * math.pi - yaw_change

            # Robot is "moving" if translating OR rotating significantly
            # 0.05 rad ≈ 3° — enough to detect intentional rotation
            is_moving = dist > self.stuck_move_threshold or yaw_change > 0.05

            if is_moving:
                self._last_move_time = time.time()
                # Clear stuck if hold time has elapsed
                if self._stuck_estop_active and self._recovery_state == self.RECOVERY_NONE:
                    held = time.time() - self._stuck_estop_start
                    if held >= self.stuck_hold_time:
                        self.get_logger().info(
                            f'[{_ts()}] ✅ Stuck e-stop cleared — robot moved after {held:.0f}s hold'
                        )
                        self._stuck_estop_active = False

        self._last_slam_position = (x, y)
        self._last_slam_yaw = yaw

    def _publish_stuck_location(self):
        """Publish the current position as a stuck location for other nodes."""
        pose = self._get_slam_pose()
        if pose is None:
            return
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = pose[0]
        msg.point.y = pose[1]
        msg.point.z = 0.0
        self._stuck_pub.publish(msg)

    def _execute_recovery(self):
        """State machine for recovery maneuver (backup + strafe)."""
        elapsed = time.time() - self._recovery_start
        cmd = Twist()

        if self._recovery_state == self.RECOVERY_BACKUP:
            if elapsed < self.recovery_backup_duration:
                # Drive backward
                cmd.linear.x = self.recovery_backup_speed
                self._cmd_pub.publish(cmd)
            else:
                # Transition to strafe
                self._recovery_state = self.RECOVERY_STRAFE
                self._recovery_start = time.time()
                self.get_logger().info(
                    f'[{_ts()}] 🔄 Recovery: strafing {"left" if self._recovery_direction > 0 else "right"}...'
                )

        elif self._recovery_state == self.RECOVERY_STRAFE:
            if elapsed < self.recovery_strafe_duration:
                # Strafe sideways (mecanum advantage!)
                cmd.linear.y = self.recovery_strafe_speed * self._recovery_direction
                self._cmd_pub.publish(cmd)
            else:
                # Done — transition to hold
                self._recovery_state = self.RECOVERY_HOLD
                self._recovery_start = time.time()
                stop = Twist()
                self._cmd_pub.publish(stop)
                self.get_logger().info(
                    f'[{_ts()}] ✅ Recovery maneuver complete — holding for replan'
                )
                # Reset move time so stuck detection doesn't immediately re-trigger
                self._last_move_time = time.time()

        elif self._recovery_state == self.RECOVERY_HOLD:
            # Hold for a few seconds, then clear
            stop = Twist()
            self._cmd_pub.publish(stop)
            if elapsed >= 3.0:
                self._recovery_state = self.RECOVERY_NONE
                self._stuck_estop_active = False
                # CRITICAL: reset command tracking so stuck detection
                # doesn't immediately re-trigger with stale cumulative time
                self._cmd_active_since = None
                self._last_move_time = time.time()
                self.get_logger().info(
                    f'[{_ts()}] 🟢 Recovery hold complete — resuming exploration'
                )
                self._publish_event('🟢 Recovery complete — resuming')

    def _start_recovery(self):
        """Initiate recovery maneuver after stuck detection."""
        self._recovery_count += 1
        self._recovery_state = self.RECOVERY_BACKUP
        self._recovery_start = time.time()
        # Reset command tracking — recovery is its own thing
        self._cmd_active_since = None
        # Alternate strafe direction each time
        self._recovery_direction = 1 if self._recovery_count % 2 == 1 else -1
        self.get_logger().info(
            f'[{_ts()}] 🔄 Recovery #{self._recovery_count}: backing up...'
        )

    def _safety_check(self):
        """Main safety loop at 20Hz."""
        # If recovery maneuver is active, execute it and skip normal checks
        if self._recovery_state != self.RECOVERY_NONE:
            self._execute_recovery()
            return

        if self._latest_scan is None:
            return

        scan = self._latest_scan
        min_reading = float('inf')

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

        # Lidar-based emergency stop
        if min_reading < self.min_dist:
            stop = Twist()
            self._cmd_pub.publish(stop)

            if not self._estop_active:
                self._estop_active = True
                self._goal_canceled = False
                self._estop_count += 1
                self.get_logger().error(
                    f'[{_ts()}] 🛑 EMERGENCY STOP #{self._estop_count} — '
                    f'obstacle at {min_reading:.3f}m (threshold: {self.min_dist}m)'
                )
                self._publish_event(
                    f'🛑 Emergency stop — obstacle at {min_reading:.2f}m'
                )

            if not self._goal_canceled:
                self._cancel_nav2_goals()
                self._goal_canceled = True

        elif self._estop_active:
            if min_reading >= self.resume_dist:
                self.get_logger().info(
                    f'[{_ts()}] ✅ E-stop cleared — closest obstacle at {min_reading:.3f}m'
                )
                self._estop_active = False
            else:
                stop = Twist()
                self._cmd_pub.publish(stop)

        # ── Stuck detection (SLAM-based) ──
        self._check_stuck()

        # Don't check stuck during startup grace period
        if time.time() - self._node_start_time < self.startup_grace:
            return

        if self._last_cmd_vel is not None and not self._estop_active:
            cmd = self._last_cmd_vel
            is_commanded = (
                abs(cmd.linear.x) > 0.01 or
                abs(cmd.linear.y) > 0.01 or
                abs(cmd.angular.z) > 0.05
            )

            if is_commanded:
                # Track when continuous commands started
                if self._cmd_active_since is None:
                    self._cmd_active_since = time.time()

                # Only consider stuck if commands have been active for cmd_active_threshold
                cmd_duration = time.time() - self._cmd_active_since
                if cmd_duration < self.cmd_active_threshold:
                    return  # too early to call it stuck

                time_since_move = time.time() - self._last_move_time
                if time_since_move > self.stuck_timeout and not self._stuck_estop_active:
                    self._stuck_estop_active = True
                    self._stuck_estop_start = time.time()
                    self._estop_count += 1
                    self.get_logger().error(
                        f'[{_ts()}] 🛑 STUCK DETECTED #{self._estop_count} — '
                        f'no SLAM movement for {time_since_move:.1f}s despite '
                        f'{cmd_duration:.1f}s of motor commands. Starting recovery.'
                    )
                    self._publish_event(
                        f'🛑 Stuck detected — starting recovery maneuver'
                    )
                    # Cancel Nav2 goal first
                    self._cancel_nav2_goals()
                    # Publish stuck location for frontier_explorer
                    self._publish_stuck_location()
                    # Start recovery maneuver (backup + strafe)
                    self._start_recovery()

                if self._stuck_estop_active and self._recovery_state == self.RECOVERY_NONE:
                    stop = Twist()
                    self._cmd_pub.publish(stop)
            else:
                # Motors idle — reset command tracking AND movement clock.
                # When Nav2 pauses for replanning (sends zero vel), the robot is
                # intentionally stopped — not stuck. Reset last_move_time so the
                # "no movement" clock doesn't accumulate across idle periods.
                if self._cmd_active_since is not None:
                    self._last_move_time = time.time()
                self._cmd_active_since = None

    def _cancel_nav2_goals(self):
        """Cancel all active Nav2 goals via cancel service."""
        self.get_logger().warn('Canceling all Nav2 goals...')
        try:
            if not self._cancel_client.service_is_ready():
                self.get_logger().warn('Cancel service not ready')
                return
            request = CancelGoal.Request()
            future = self._cancel_client.call_async(request)
            future.add_done_callback(self._cancel_done)
        except Exception as e:
            self.get_logger().warn(f'Could not cancel Nav2 goals: {e}')

    def _cancel_done(self, future):
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
        # Send a final zero-velocity command before shutting down
        try:
            stop = Twist()
            node._cmd_pub.publish(stop)
        except Exception:
            pass

        # Log summary before destroying the node (avoids "publisher's context
        # is invalid" error that occurs when logging after destroy_node)
        estop_count = node._estop_count
        try:
            node.get_logger().info(
                f'[{_ts()}] Safety monitor stopped. Total e-stops: {estop_count}'
            )
        except Exception:
            print(f'[{_ts()}] Safety monitor stopped. Total e-stops: {estop_count}')

        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
