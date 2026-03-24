"""Frontier-based autonomous exploration for JetAuto.

Reads the occupancy grid from slam_toolbox, identifies frontier cells
(boundaries between known-free and unknown space), clusters them, and
sends the best frontier as a Nav2 goal.

Algorithm:
1. Get current occupancy grid from /map
2. Find frontier cells (free cells adjacent to unknown cells)
3. Cluster nearby frontier cells into frontier regions
4. Score each frontier: size * gain_scale - distance * potential_scale
5. Send the best-scoring frontier centroid as a Nav2 goal
6. Wait for goal completion, repeat

Based on the explore_lite approach but implemented as a standalone ROS2 node
(no extra package dependency needed).
"""

import math
import time
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from tf2_ros import Buffer, TransformListener, LookupException


# Occupancy grid values
FREE = 0
UNKNOWN = -1
OCCUPIED = 100

# Frontier clustering: cells within this distance (in grid cells) are same frontier
CLUSTER_RADIUS = 5


def _ts():
    """Human-readable timestamp for log messages."""
    return datetime.now().strftime('%H:%M:%S')


class FrontierExplorer(Node):
    """Explores unknown space by navigating to frontier boundaries."""

    def __init__(self):
        super().__init__('frontier_explorer')

        # Parameters
        self.declare_parameter('explore_timeout', 300.0)
        self.declare_parameter('min_frontier_size', 5)
        self.declare_parameter('robot_radius', 0.20)
        self.declare_parameter('potential_scale', 3.0)
        self.declare_parameter('gain_scale', 1.0)
        self.declare_parameter('goal_timeout', 60.0)
        self.declare_parameter('transform_tolerance', 0.5)
        self.declare_parameter('replan_interval', 5.0)

        self.explore_timeout = self.get_parameter('explore_timeout').value
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.potential_scale = self.get_parameter('potential_scale').value
        self.gain_scale = self.get_parameter('gain_scale').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.replan_interval = self.get_parameter('replan_interval').value

        # Stuck location blacklist — escalating radius for repeat offenders
        self.declare_parameter('stuck_blacklist_radius', 1.0)  # base radius (1st stuck)
        self.declare_parameter('stuck_location_ttl', 120.0)  # TTL for 1st stuck (2 min)
        self.declare_parameter('stuck_escalation_radius', 0.5)  # extra radius per repeat
        self.declare_parameter('stuck_escalation_ttl', 180.0)  # extra TTL per repeat
        self.declare_parameter('stuck_max_radius', 3.0)  # cap the exclusion zone
        self.declare_parameter('stuck_merge_radius', 1.5)  # merge nearby stuck events
        self.declare_parameter('stuck_path_radius', 0.5)  # fixed radius for path-crossing check
        self.stuck_blacklist_radius = self.get_parameter('stuck_blacklist_radius').value
        self.stuck_location_ttl = self.get_parameter('stuck_location_ttl').value
        self.stuck_escalation_radius = self.get_parameter('stuck_escalation_radius').value
        self.stuck_escalation_ttl = self.get_parameter('stuck_escalation_ttl').value
        self.stuck_max_radius = self.get_parameter('stuck_max_radius').value
        self.stuck_merge_radius = self.get_parameter('stuck_merge_radius').value
        self.stuck_path_radius = self.get_parameter('stuck_path_radius').value

        # State
        self._map: Optional[OccupancyGrid] = None
        self._exploring = False
        self._start_time = None
        self._current_goal_handle = None
        self._failed_goals: List[Tuple[float, float]] = []
        # Stuck locations with escalation: (x, y, first_time, count)
        # count tracks how many times stuck was detected near this spot
        self._stuck_locations: List[Tuple[float, float, float, int]] = []
        self._goals_sent = 0
        self._goals_reached = 0
        # Last successfully reached position — for retreat-first behavior
        self._last_good_position: Optional[Tuple[float, float]] = None
        # Whether we're in "retreat" mode (going back to last good position)
        self._retreating = False
        # Counter for consecutive "all frontiers blocked" ticks
        self._blocked_ticks = 0

        # TF
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Subscribers
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, map_qos
        )

        # Stuck location subscriber (from safety_monitor)
        self._stuck_sub = self.create_subscription(
            PointStamped, '/stuck_locations', self._stuck_callback, 10
        )

        # Nav2 action client
        self._nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=ReentrantCallbackGroup(),
        )

        # Event publisher for dashboard announcements
        self._event_pub = self.create_publisher(String, '/explore/events', 10)

        # Minimum map size before we start exploring (in cells)
        self.declare_parameter('min_map_cells', 500)  # ~500 free cells before starting
        self.min_map_cells = self.get_parameter('min_map_cells').value
        self._map_ready = False

        # Main exploration timer
        self._explore_timer = self.create_timer(
            self.replan_interval, self._explore_tick
        )

        self.get_logger().info(
            f'[{_ts()}] Frontier explorer initialized '
            f'(timeout={self.explore_timeout}s, min_frontier={self.min_frontier_size})'
        )

    def _publish_event(self, text: str):
        """Publish an exploration event for dashboard display/TTS."""
        msg = String()
        msg.data = text
        self._event_pub.publish(msg)

    def _map_callback(self, msg: OccupancyGrid):
        """Store latest map."""
        self._map = msg

    def _stuck_callback(self, msg: PointStamped):
        """Receive stuck locations from safety_monitor — blacklist with escalating radius.

        If stuck happens near an existing stuck zone, escalate that zone
        (bigger radius, longer TTL) instead of creating a new one.
        """
        sx, sy = msg.point.x, msg.point.y
        now = time.time()

        # Check if this is near an existing stuck zone → escalate
        for i, (ex, ey, et, count) in enumerate(self._stuck_locations):
            dist = math.sqrt((sx - ex) ** 2 + (sy - ey) ** 2)
            if dist < self.stuck_merge_radius:
                new_count = count + 1
                # Update position to midpoint, reset timer, bump count
                mx = (ex + sx) / 2.0
                my = (ey + sy) / 2.0
                self._stuck_locations[i] = (mx, my, now, new_count)
                radius = self._get_stuck_radius(new_count)
                ttl = self._get_stuck_ttl(new_count)
                self.get_logger().warn(
                    f'[{_ts()}] ⚠️ ESCALATED stuck zone at ({mx:.2f}, {my:.2f}) — '
                    f'hit #{new_count}, radius={radius:.1f}m, ttl={ttl:.0f}s'
                )
                # Trigger retreat to last known good position
                self._retreating = True
                return

        # New stuck zone
        self._stuck_locations.append((sx, sy, now, 1))
        self.get_logger().warn(
            f'[{_ts()}] ⚠️ New stuck zone at ({sx:.2f}, {sy:.2f}) — '
            f'radius={self.stuck_blacklist_radius}m, ttl={self.stuck_location_ttl:.0f}s'
        )
        # Trigger retreat to last known good position
        self._retreating = True

    def _get_stuck_radius(self, count: int) -> float:
        """Escalating radius: bigger zone for repeat offenders."""
        radius = self.stuck_blacklist_radius + (count - 1) * self.stuck_escalation_radius
        return min(radius, self.stuck_max_radius)

    def _get_stuck_ttl(self, count: int) -> float:
        """Escalating TTL: longer memory for repeat offenders.
        After 3+ stucks at the same spot, it's permanent for the session."""
        if count >= 3:
            return float('inf')  # permanent
        return self.stuck_location_ttl + (count - 1) * self.stuck_escalation_ttl

    def _get_robot_position(self) -> Optional[Tuple[float, float]]:
        """Get robot position in map frame."""
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            return (
                t.transform.translation.x,
                t.transform.translation.y,
            )
        except (LookupException, Exception) as e:
            self.get_logger().warn(f'Could not get robot position: {e}')
            return None

    def _world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices."""
        info = self._map.info
        gx = int((wx - info.origin.position.x) / info.resolution)
        gy = int((wy - info.origin.position.y) / info.resolution)
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid cell indices to world coordinates (cell center)."""
        info = self._map.info
        wx = info.origin.position.x + (gx + 0.5) * info.resolution
        wy = info.origin.position.y + (gy + 0.5) * info.resolution
        return wx, wy

    def _find_frontiers(self) -> List[List[Tuple[int, int]]]:
        """Find and cluster frontier cells in the occupancy grid.

        A frontier cell is a FREE cell that has at least one UNKNOWN neighbor.
        Clusters are groups of frontier cells connected within CLUSTER_RADIUS.
        """
        if self._map is None:
            return []

        info = self._map.info
        w, h = info.width, info.height
        data = np.array(self._map.data, dtype=np.int8).reshape((h, w))

        # Find frontier cells
        frontier_cells = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if data[y, x] != FREE:
                    continue
                # Check 4-connected neighbors for unknown
                if (data[y - 1, x] == UNKNOWN or data[y + 1, x] == UNKNOWN or
                        data[y, x - 1] == UNKNOWN or data[y, x + 1] == UNKNOWN):
                    frontier_cells.append((x, y))

        if not frontier_cells:
            return []

        # Cluster frontier cells using BFS
        frontier_set = set(frontier_cells)
        visited = set()
        clusters = []

        for cell in frontier_cells:
            if cell in visited:
                continue

            # BFS to find connected frontier cells
            cluster = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                cx, cy = queue.popleft()
                cluster.append((cx, cy))

                # Check neighbors within CLUSTER_RADIUS
                for dx in range(-CLUSTER_RADIUS, CLUSTER_RADIUS + 1):
                    for dy in range(-CLUSTER_RADIUS, CLUSTER_RADIUS + 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) in frontier_set and (nx, ny) not in visited:
                            if math.sqrt(dx * dx + dy * dy) <= CLUSTER_RADIUS:
                                visited.add((nx, ny))
                                queue.append((nx, ny))

            if len(cluster) >= self.min_frontier_size:
                clusters.append(cluster)

        return clusters

    def _is_stuck_expired(self, st: float, count: int) -> bool:
        """Check if a stuck zone has expired based on its escalation level."""
        ttl = self._get_stuck_ttl(count)
        if ttl == float('inf'):
            return False  # permanent
        return time.time() - st > ttl

    def _path_crosses_stuck_zone(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> bool:
        """Check if the straight line from (x1,y1) to (x2,y2) passes within
        the stuck zone radius of any active stuck location.

        Uses point-to-line-segment distance formula.
        Radius scales with escalation level.
        """
        if not self._stuck_locations:
            return False

        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-6:
            return False  # robot and goal at same spot

        for sx, sy, st, count in self._stuck_locations:
            if self._is_stuck_expired(st, count):
                continue
            # Use a fixed smaller radius for path-crossing checks (not the escalating
            # zone radius) — this keeps corridors passable while still avoiding the
            # exact stuck spot. Escalated zones (count >= 2) use a bigger path radius.
            path_radius = self.stuck_path_radius if count < 2 else self._get_stuck_radius(count) * 0.7
            # Project stuck point onto the line segment
            t = max(0.0, min(1.0, ((sx - x1) * dx + (sy - y1) * dy) / seg_len_sq))
            # Closest point on segment
            px = x1 + t * dx
            py = y1 + t * dy
            dist = math.sqrt((sx - px) ** 2 + (sy - py) ** 2)
            if dist < path_radius:
                return True

        return False

    def _score_frontier(
        self, cluster: List[Tuple[int, int]], robot_pos: Tuple[float, float]
    ) -> float:
        """Score a frontier cluster: bigger + closer = better."""
        # Centroid in world coordinates
        cx = sum(c[0] for c in cluster) / len(cluster)
        cy = sum(c[1] for c in cluster) / len(cluster)
        wx, wy = self._grid_to_world(int(cx), int(cy))

        # Distance from robot
        dist = math.sqrt(
            (wx - robot_pos[0]) ** 2 + (wy - robot_pos[1]) ** 2
        )

        # Skip frontiers too close to previously failed goals (0.5m radius)
        for fx, fy in self._failed_goals:
            if math.sqrt((wx - fx) ** 2 + (wy - fy) ** 2) < 0.5:
                return -1.0

        # Skip frontiers near active stuck locations (escalating radius + TTL)
        for sx, sy, st, count in self._stuck_locations:
            if self._is_stuck_expired(st, count):
                continue
            radius = self._get_stuck_radius(count)
            if math.sqrt((wx - sx) ** 2 + (wy - sy) ** 2) < radius:
                return -1.0

        # Skip frontiers whose straight-line path passes through stuck zones
        # This prevents Nav2 from routing through invisible obstacles (bed, dog beds)
        if self._path_crosses_stuck_zone(robot_pos[0], robot_pos[1], wx, wy):
            return -1.0

        # Score: size bonus - distance penalty
        size = len(cluster)
        score = size * self.gain_scale - dist * self.potential_scale

        return score

    def _explore_tick(self):
        """Main exploration loop — called periodically."""
        # Check timeout
        if self.explore_timeout > 0 and self._start_time is not None:
            elapsed = time.time() - self._start_time
            if elapsed >= self.explore_timeout:
                self.get_logger().info(
                    f'Exploration timeout reached ({self.explore_timeout}s). '
                    f'Goals sent: {self._goals_sent}, reached: {self._goals_reached}'
                )
                self._explore_timer.cancel()
                return

        # Wait for map
        if self._map is None:
            self.get_logger().info('Waiting for map from slam_toolbox...')
            return

        # Wait for map to be big enough (SLAM needs time to build)
        if not self._map_ready:
            data = self._map.data
            free_cells = sum(1 for c in data if c == FREE)
            total_cells = len(data)
            if free_cells < self.min_map_cells:
                self.get_logger().info(
                    f'Waiting for map to grow: {free_cells}/{self.min_map_cells} '
                    f'free cells (map: {self._map.info.width}x{self._map.info.height})'
                )
                return
            self._map_ready = True
            self.get_logger().info(
                f'[{_ts()}] Map ready! {free_cells} free cells, '
                f'{self._map.info.width}x{self._map.info.height} grid'
            )
            self._publish_event('🗺 Map ready — starting exploration')

        # Wait for Nav2
        if not self._nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for Nav2 action server...')
            return

        # Don't send new goal if one is in progress
        if self._current_goal_handle is not None:
            return

        # Start timer on first tick
        if self._start_time is None:
            self._start_time = time.time()
            self.get_logger().info('Starting frontier exploration!')
            self._publish_event('🚗 Frontier exploration started')

        # ── Retreat-first: after stuck recovery, go back to last good position ──
        if self._retreating and self._last_good_position is not None:
            rx, ry = self._last_good_position
            robot_pos = self._get_robot_position()
            if robot_pos is not None:
                dist_to_safe = math.sqrt(
                    (rx - robot_pos[0]) ** 2 + (ry - robot_pos[1]) ** 2
                )
                if dist_to_safe > 0.3:  # only retreat if we're far enough away
                    self.get_logger().info(
                        f'[{_ts()}] 🔙 Retreating to last safe position '
                        f'({rx:.2f}, {ry:.2f}) — {dist_to_safe:.1f}m away'
                    )
                    self._publish_event(f'🔙 Retreating to safe position — {dist_to_safe:.1f}m away')
                    self._send_goal(rx, ry, is_retreat=True)
                    return
                else:
                    self.get_logger().info(
                        f'[{_ts()}] Already near safe position — skipping retreat'
                    )
            self._retreating = False

        # Get robot position
        robot_pos = self._get_robot_position()
        if robot_pos is None:
            return

        # Verify robot is roughly within the map bounds (0.5m tolerance for edge cases)
        info = self._map.info
        margin = 0.5  # meters of tolerance
        map_x_min = info.origin.position.x - margin
        map_y_min = info.origin.position.y - margin
        map_x_max = info.origin.position.x + info.width * info.resolution + margin
        map_y_max = info.origin.position.y + info.height * info.resolution + margin
        rx, ry = robot_pos

        if not (map_x_min < rx < map_x_max and map_y_min < ry < map_y_max):
            self.get_logger().warn(
                f'[{_ts()}] Robot ({rx:.2f}, {ry:.2f}) far outside map bounds. '
                f'Waiting for SLAM to expand...'
            )
            return

        # Find frontiers
        frontiers = self._find_frontiers()
        if not frontiers:
            self.get_logger().info(
                '🎉 No frontiers found — exploration complete! '
                f'Goals sent: {self._goals_sent}, reached: {self._goals_reached}'
            )
            self._publish_event(
                f'🎉 Exploration complete! Reached {self._goals_reached} of '
                f'{self._goals_sent} goals'
            )
            self._explore_timer.cancel()
            return

        # Score and pick best frontier
        scored = [
            (self._score_frontier(f, robot_pos), f)
            for f in frontiers
        ]
        scored = [(s, f) for s, f in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            self._blocked_ticks += 1

            # First, prune expired stuck zones
            active_stuck = [
                (sx, sy, st, c) for sx, sy, st, c in self._stuck_locations
                if not self._is_stuck_expired(st, c)
            ]
            pruned = len(self._stuck_locations) - len(active_stuck)
            self._stuck_locations = active_stuck

            # After ~30s of being fully blocked, clear first-time (count=1) stuck zones.
            # These are likely Nav2 planning pauses / false positives rather than real walls.
            # Confirmed repeat obstacles (count >= 2) are kept.
            # replan_interval is 5s, so 6 ticks ≈ 30s.
            cleared_first_time = 0
            if self._blocked_ticks >= 6 and self._stuck_locations:
                before = len(self._stuck_locations)
                self._stuck_locations = [
                    (sx, sy, st, c) for sx, sy, st, c in self._stuck_locations
                    if c >= 2  # keep confirmed repeat obstacles
                ]
                cleared_first_time = before - len(self._stuck_locations)
                if cleared_first_time:
                    self._blocked_ticks = 0
                    self.get_logger().warn(
                        f'[{_ts()}] ⚠️ Blocked for 30s — cleared {cleared_first_time} first-time '
                        f'stuck zone(s) to unblock exploration. '
                        f'{len(self._stuck_locations)} confirmed zone(s) remain.'
                    )

            self.get_logger().warn(
                f'All frontiers scored negative (blocked/failed). '
                f'Clearing {len(self._failed_goals)} failed goals'
                f'{f", pruned {pruned} expired stuck zones" if pruned else ""}'
                f'{f", {len(self._stuck_locations)} stuck zone(s) remain" if self._stuck_locations else ""}'
                f' and retrying... (blocked_ticks={self._blocked_ticks})'
            )
            self._failed_goals.clear()
            return

        # Found a valid frontier — reset the blocked counter
        self._blocked_ticks = 0

        # Navigate to best frontier centroid
        best_score, best_cluster = scored[0]
        cx = sum(c[0] for c in best_cluster) / len(best_cluster)
        cy = sum(c[1] for c in best_cluster) / len(best_cluster)
        wx, wy = self._grid_to_world(int(cx), int(cy))

        self.get_logger().info(
            f'[{_ts()}] Exploring frontier at ({wx:.2f}, {wy:.2f}) — '
            f'size={len(best_cluster)}, score={best_score:.1f}, '
            f'frontiers_remaining={len(scored)}'
        )
        self._publish_event(
            f'🧭 Heading to frontier — {len(scored)} areas remaining'
        )

        self._send_goal(wx, wy)

    def _send_goal(self, x: float, y: float, is_retreat: bool = False):
        """Send a navigation goal to Nav2."""
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0  # face forward

        if not is_retreat:
            self._goals_sent += 1
        future = self._nav_client.send_goal_async(
            goal, feedback_callback=self._nav_feedback
        )
        future.add_done_callback(
            lambda f: self._goal_response(f, x, y, is_retreat)
        )

    def _goal_response(self, future, x: float, y: float, is_retreat: bool = False):
        """Handle Nav2 goal acceptance/rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f'Goal ({x:.2f}, {y:.2f}) rejected by Nav2')
            if is_retreat:
                self._retreating = False  # give up on retreat, continue exploring
            else:
                self._failed_goals.append((x, y))
            self._current_goal_handle = None
            return

        self._current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._goal_result(f, x, y, is_retreat)
        )

    def _goal_result(self, future, x: float, y: float, is_retreat: bool = False):
        """Handle Nav2 goal completion."""
        result = future.result()
        status = result.status

        if is_retreat:
            # Retreat goal completed — regardless of outcome, resume exploring
            self._retreating = False
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(
                    f'[{_ts()}] ✅ Retreated to safe position ({x:.2f}, {y:.2f})'
                )
            else:
                self.get_logger().warn(
                    f'[{_ts()}] Retreat to ({x:.2f}, {y:.2f}) ended with status '
                    f'{status} — resuming exploration anyway'
                )
            self._current_goal_handle = None
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'[{_ts()}] ✅ Reached frontier ({x:.2f}, {y:.2f})')
            self._goals_reached += 1
            self._last_good_position = (x, y)
            self._publish_event(f'✅ Reached frontier ({self._goals_reached} explored)')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn(f'[{_ts()}] ❌ Failed to reach ({x:.2f}, {y:.2f}) — aborted')
            self._failed_goals.append((x, y))
            self._publish_event('❌ Frontier unreachable — picking another')
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f'[{_ts()}] ⏹ Goal ({x:.2f}, {y:.2f}) canceled — marking as failed')
            self._failed_goals.append((x, y))
        else:
            self.get_logger().warn(f'Goal ({x:.2f}, {y:.2f}) ended with status {status}')
            self._failed_goals.append((x, y))

        self._current_goal_handle = None

    def _nav_feedback(self, feedback_msg):
        """Log periodic Nav2 feedback."""
        # Could log distance remaining, ETA, etc.
        pass


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        sent = node._goals_sent
        reached = node._goals_reached
        try:
            node.get_logger().info(
                f'[{_ts()}] Explorer stopped. Goals: {sent} sent, {reached} reached'
            )
        except Exception:
            print(f'[{_ts()}] Explorer stopped. Goals: {sent} sent, {reached} reached')

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
