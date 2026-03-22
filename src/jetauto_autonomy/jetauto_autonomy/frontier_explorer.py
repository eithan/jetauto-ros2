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
from typing import List, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from tf2_ros import Buffer, TransformListener, LookupException


# Occupancy grid values
FREE = 0
UNKNOWN = -1
OCCUPIED = 100

# Frontier clustering: cells within this distance (in grid cells) are same frontier
CLUSTER_RADIUS = 5


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

        # State
        self._map: Optional[OccupancyGrid] = None
        self._exploring = False
        self._start_time = None
        self._current_goal_handle = None
        self._failed_goals: List[Tuple[float, float]] = []
        self._goals_sent = 0
        self._goals_reached = 0

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

        # Nav2 action client
        self._nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=ReentrantCallbackGroup(),
        )

        # Minimum map size before we start exploring (in cells)
        self.declare_parameter('min_map_cells', 500)  # ~500 free cells before starting
        self.min_map_cells = self.get_parameter('min_map_cells').value
        self._map_ready = False

        # Main exploration timer
        self._explore_timer = self.create_timer(
            self.replan_interval, self._explore_tick
        )

        self.get_logger().info(
            f'Frontier explorer initialized '
            f'(timeout={self.explore_timeout}s, min_frontier={self.min_frontier_size})'
        )

    def _map_callback(self, msg: OccupancyGrid):
        """Store latest map."""
        self._map = msg

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

        # Skip frontiers too close to previously failed goals
        for fx, fy in self._failed_goals:
            if math.sqrt((wx - fx) ** 2 + (wy - fy) ** 2) < 0.5:
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
                f'Map ready! {free_cells} free cells, '
                f'{self._map.info.width}x{self._map.info.height} grid'
            )

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

        # Get robot position
        robot_pos = self._get_robot_position()
        if robot_pos is None:
            return

        # Verify robot is within the map bounds
        info = self._map.info
        map_x_min = info.origin.position.x
        map_y_min = info.origin.position.y
        map_x_max = map_x_min + info.width * info.resolution
        map_y_max = map_y_min + info.height * info.resolution
        rx, ry = robot_pos

        if not (map_x_min < rx < map_x_max and map_y_min < ry < map_y_max):
            self.get_logger().warn(
                f'Robot ({rx:.2f}, {ry:.2f}) outside map bounds '
                f'({map_x_min:.2f},{map_y_min:.2f})-({map_x_max:.2f},{map_y_max:.2f}). '
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
            self.get_logger().warn(
                'All frontiers scored negative (blocked/failed). '
                'Clearing failed goals and retrying...'
            )
            self._failed_goals.clear()
            return

        # Navigate to best frontier centroid
        best_score, best_cluster = scored[0]
        cx = sum(c[0] for c in best_cluster) / len(best_cluster)
        cy = sum(c[1] for c in best_cluster) / len(best_cluster)
        wx, wy = self._grid_to_world(int(cx), int(cy))

        self.get_logger().info(
            f'Exploring frontier at ({wx:.2f}, {wy:.2f}) — '
            f'size={len(best_cluster)}, score={best_score:.1f}, '
            f'frontiers_remaining={len(scored)}'
        )

        self._send_goal(wx, wy)

    def _send_goal(self, x: float, y: float):
        """Send a navigation goal to Nav2."""
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0  # face forward

        self._goals_sent += 1
        future = self._nav_client.send_goal_async(
            goal, feedback_callback=self._nav_feedback
        )
        future.add_done_callback(
            lambda f: self._goal_response(f, x, y)
        )

    def _goal_response(self, future, x: float, y: float):
        """Handle Nav2 goal acceptance/rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f'Goal ({x:.2f}, {y:.2f}) rejected by Nav2')
            self._failed_goals.append((x, y))
            self._current_goal_handle = None
            return

        self._current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._goal_result(f, x, y)
        )

    def _goal_result(self, future, x: float, y: float):
        """Handle Nav2 goal completion."""
        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'✅ Reached frontier ({x:.2f}, {y:.2f})')
            self._goals_reached += 1
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn(f'❌ Failed to reach ({x:.2f}, {y:.2f}) — aborted')
            self._failed_goals.append((x, y))
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f'⏹ Goal ({x:.2f}, {y:.2f}) canceled')
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
        node.get_logger().info(
            f'Explorer stopped. Goals: {node._goals_sent} sent, '
            f'{node._goals_reached} reached'
        )
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
