"""Virtual obstacle injector for the Nav2 costmap.

When the safety monitor detects stuck, the robot hit something invisible
to the costmap (lidar can't see beds, low tables, etc). This node
injects virtual obstacles into the costmap so Nav2 learns to avoid
those areas for the rest of the session.

Approach:
- Subscribe to /stuck_locations from safety_monitor
- Generate a synthetic PointCloud2 "wall" at each stuck location
- Publish to /virtual_obstacles (consumed by a costmap obstacle layer)
- Obstacles persist for the entire session (no TTL — if you got stuck
  there once, the obstacle is real and won't move)

This is the critical missing piece: the frontier explorer can avoid
sending goals near stuck spots, but Nav2's DWB local planner will still
drive through them. By injecting into the costmap, Nav2 itself routes
around the area.
"""

import math
import struct
import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener


def _create_wall_points(
    cx: float, cy: float, radius: float = 0.8, height: float = 0.5,
    point_spacing: float = 0.05,
) -> List[Tuple[float, float, float]]:
    """Generate a filled circle of points representing a virtual obstacle.

    Creates a disk of points at (cx, cy) with the given radius.
    Points are at multiple heights so the costmap sees them regardless
    of the min/max_obstacle_height settings.
    """
    points = []
    # Fill the circle with a grid of points
    steps = int(radius / point_spacing)
    heights = [0.05, 0.15, 0.30, 0.45]  # multiple heights for robustness

    for ix in range(-steps, steps + 1):
        for iy in range(-steps, steps + 1):
            px = cx + ix * point_spacing
            py = cy + iy * point_spacing
            dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if dist <= radius:
                for h in heights:
                    points.append((px, py, h))

    return points


def _points_to_pointcloud2(
    points: List[Tuple[float, float, float]], frame_id: str = 'map',
    stamp=None,
) -> PointCloud2:
    """Convert a list of (x, y, z) points to a PointCloud2 message."""
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    if stamp:
        msg.header.stamp = stamp

    # Define fields: x, y, z as float32
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12  # 3 x float32
    msg.height = 1
    msg.width = len(points)
    msg.row_step = msg.point_step * msg.width
    msg.is_bigendian = False
    msg.is_dense = True

    # Pack point data
    data = bytearray()
    for x, y, z in points:
        data.extend(struct.pack('<fff', x, y, z))
    msg.data = bytes(data)

    return msg


class ObstacleInjector(Node):
    """Injects virtual obstacles into the costmap at stuck locations."""

    def __init__(self):
        super().__init__('obstacle_injector')

        # Parameters
        self.declare_parameter('obstacle_radius', 0.8)  # meters around stuck point
        self.declare_parameter('publish_rate', 2.0)  # Hz — must re-publish for costmap to keep them
        self.declare_parameter('merge_radius', 1.0)  # merge nearby stuck points

        self.obstacle_radius = self.get_parameter('obstacle_radius').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.merge_radius = self.get_parameter('merge_radius').value

        # Persistent stuck locations (never expire during a session)
        self._stuck_points: List[Tuple[float, float]] = []

        # Subscribe to stuck locations from safety_monitor
        self._stuck_sub = self.create_subscription(
            PointStamped, '/stuck_locations', self._stuck_callback, 10
        )

        # Publish virtual obstacles as PointCloud2
        # The costmap obstacle layer consumes this
        self._obstacle_pub = self.create_publisher(
            PointCloud2, '/virtual_obstacles', 10
        )

        # Periodic republish — costmap obstacle layers with clearing=false
        # still need fresh observations, so we republish the full set
        self._pub_timer = self.create_timer(
            1.0 / self.publish_rate, self._publish_obstacles
        )

        self.get_logger().info(
            f'Obstacle injector active — radius={self.obstacle_radius}m, '
            f'publish_rate={self.publish_rate}Hz'
        )

    def _stuck_callback(self, msg: PointStamped):
        """Receive a stuck location and add to permanent list."""
        new_x, new_y = msg.point.x, msg.point.y

        # Check if near an existing stuck point (merge to avoid duplicates)
        for sx, sy in self._stuck_points:
            dist = math.sqrt((new_x - sx) ** 2 + (new_y - sy) ** 2)
            if dist < self.merge_radius:
                self.get_logger().info(
                    f'Stuck at ({new_x:.2f}, {new_y:.2f}) merged with '
                    f'existing zone at ({sx:.2f}, {sy:.2f})'
                )
                return

        self._stuck_points.append((new_x, new_y))
        self.get_logger().warn(
            f'🚧 New virtual obstacle at ({new_x:.2f}, {new_y:.2f}) — '
            f'total zones: {len(self._stuck_points)}'
        )

    def _publish_obstacles(self):
        """Republish all virtual obstacles as a single PointCloud2."""
        if not self._stuck_points:
            return

        all_points = []
        for sx, sy in self._stuck_points:
            wall = _create_wall_points(sx, sy, radius=self.obstacle_radius)
            all_points.extend(wall)

        if not all_points:
            return

        cloud = _points_to_pointcloud2(
            all_points,
            frame_id='map',
            stamp=self.get_clock().now().to_msg(),
        )
        self._obstacle_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleInjector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
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
