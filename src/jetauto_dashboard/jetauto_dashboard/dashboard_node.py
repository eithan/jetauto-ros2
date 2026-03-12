#!/usr/bin/env python3
"""
Dashboard Node — serves a full-screen web control panel for the JetAuto robot.

Runs a Flask + SocketIO web server alongside a ROS2 node. The browser connects
via WebSocket for real-time state updates and sends commands back to ROS2.
"""

import os
import glob
import time
import subprocess
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, Float32, String

# Optional: jetauto_msgs for detection log
try:
    from jetauto_msgs.msg import DetectedObjectArray
    HAS_DETECTION_MSGS = True
except ImportError:
    HAS_DETECTION_MSGS = False

from flask import Flask, send_from_directory
from flask_socketio import SocketIO


class DashboardNode(Node):
    """ROS2 node that bridges topics to/from a web dashboard."""

    def __init__(self, socketio: SocketIO):
        super().__init__('dashboard_node')
        self.socketio = socketio
        self.start_time = time.time()

        # -- Parameters --
        self.declare_parameter('port', 5000)
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('robot_name', 'JARVIS')
        self.declare_parameter('system_monitor_interval', 30.0)  # seconds
        self.declare_parameter('shutdown_command', 'sudo /sbin/poweroff')

        # -- State --
        self.state = {
            'battery': None,
            'cpu_temp': None,
            'gpu_temp': None,
            'voice_enabled': False,
            'vision_enabled': False,
            'voice_state': 'idle',  # idle | listening | processing | speaking
            'detections': [],       # last 5 detected objects
            'uptime': 0,
        }

        # -- QoS for latched/transient local topics --
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        # -- Subscribers --
        self.create_subscription(
            Float32, '/jetauto/battery', self._on_battery, qos_best_effort)
        self.create_subscription(
            Float32, '/jetauto/system/cpu_temp', self._on_cpu_temp, qos_best_effort)
        self.create_subscription(
            Float32, '/jetauto/system/gpu_temp', self._on_gpu_temp, qos_best_effort)
        self.create_subscription(
            Bool, '/jetauto/voice/enable', self._on_voice_enable, qos_reliable)
        self.create_subscription(
            Bool, '/jetauto/detection/enable', self._on_vision_enable, qos_reliable)
        self.create_subscription(
            String, '/jetauto/voice/state', self._on_voice_state, qos_reliable)

        if HAS_DETECTION_MSGS:
            self.create_subscription(
                DetectedObjectArray, '/detected_objects',
                self._on_detections, qos_best_effort)

        # -- Publishers --
        self.pub_voice_enable = self.create_publisher(Bool, '/jetauto/voice/enable', 1)
        self.pub_vision_enable = self.create_publisher(Bool, '/jetauto/detection/enable', 1)
        self.pub_shutdown = self.create_publisher(Bool, '/jetauto/shutdown', 1)
        self.pub_tts = self.create_publisher(String, '/tts/speak', 1)

        # -- Uptime timer --
        self.create_timer(1.0, self._tick_uptime)

        # -- System monitor (reads temps/battery from sysfs directly) --
        monitor_interval = self.get_parameter('system_monitor_interval').value
        self.create_timer(monitor_interval, self._poll_system)
        # Run once at startup
        self._poll_system()

        self.get_logger().info('Dashboard node initialized')

    # ── Subscription callbacks ─────────────────────────────────────

    def _on_battery(self, msg: Float32):
        self.state['battery'] = round(msg.data, 1)
        self._emit_state()

    def _on_cpu_temp(self, msg: Float32):
        self.state['cpu_temp'] = round(msg.data, 1)
        self._emit_state()

    def _on_gpu_temp(self, msg: Float32):
        self.state['gpu_temp'] = round(msg.data, 1)
        self._emit_state()

    def _on_voice_enable(self, msg: Bool):
        self.state['voice_enabled'] = msg.data
        self._emit_state()

    def _on_vision_enable(self, msg: Bool):
        self.state['vision_enabled'] = msg.data
        self._emit_state()

    def _on_voice_state(self, msg: String):
        self.state['voice_state'] = msg.data
        self._emit_state()

    def _on_detections(self, msg):
        """Append detected objects to the rolling log (max 5)."""
        ts = datetime.now().strftime('%H:%M:%S')
        for obj in msg.objects:
            entry = {
                'label': obj.label,
                'confidence': round(obj.confidence, 2),
                'time': ts,
            }
            self.state['detections'].append(entry)
        # Keep last 5
        self.state['detections'] = self.state['detections'][-5:]
        self._emit_state()

    def _tick_uptime(self):
        self.state['uptime'] = int(time.time() - self.start_time)
        # Emit every 5 seconds to avoid flooding
        if self.state['uptime'] % 5 == 0:
            self._emit_state()

    # ── System monitor (direct sysfs reads) ────────────────────────

    def _poll_system(self):
        """Read CPU/GPU temps and battery from sysfs. Falls back gracefully."""
        # CPU temp — Jetson thermal zones
        cpu_temp = self._read_jetson_temp('CPU')
        if cpu_temp is not None:
            self.state['cpu_temp'] = cpu_temp

        # GPU temp
        gpu_temp = self._read_jetson_temp('GPU')
        if gpu_temp is not None:
            self.state['gpu_temp'] = gpu_temp

        # Battery — try common power_supply paths
        battery = self._read_battery()
        if battery is not None:
            self.state['battery'] = battery

        self._emit_state()

    def _read_jetson_temp(self, name: str):
        """Read temperature from Jetson thermal zones by name.

        Jetson Orin Nano exposes zones like:
          /sys/class/thermal/thermal_zone*/type → 'CPU-therm', 'GPU-therm', etc.
          /sys/class/thermal/thermal_zone*/temp → millidegrees C
        """
        try:
            for zone_dir in glob.glob('/sys/class/thermal/thermal_zone*'):
                type_path = os.path.join(zone_dir, 'type')
                if not os.path.exists(type_path):
                    continue
                with open(type_path) as f:
                    zone_type = f.read().strip()
                if name.lower() in zone_type.lower():
                    temp_path = os.path.join(zone_dir, 'temp')
                    with open(temp_path) as f:
                        raw = int(f.read().strip())
                    # Jetson reports millidegrees
                    temp = raw / 1000.0 if raw > 1000 else float(raw)
                    return round(temp, 1)
        except Exception:
            pass
        return None

    def _read_battery(self):
        """Read battery capacity from power_supply sysfs.

        JetAuto uses a LiPo with a fuel gauge. Common paths:
          /sys/class/power_supply/battery/capacity
          /sys/class/power_supply/BAT*/capacity
        """
        try:
            for ps_dir in glob.glob('/sys/class/power_supply/*'):
                cap_path = os.path.join(ps_dir, 'capacity')
                type_path = os.path.join(ps_dir, 'type')
                if not os.path.exists(cap_path):
                    continue
                # Only read Battery type (not Mains/USB)
                if os.path.exists(type_path):
                    with open(type_path) as f:
                        if f.read().strip() != 'Battery':
                            continue
                with open(cap_path) as f:
                    return float(f.read().strip())
        except Exception:
            pass
        return None

    def _emit_state(self):
        """Push current state to all connected browsers."""
        try:
            self.socketio.emit('state', self.state)
        except AttributeError:
            # socketio.server not initialized yet (early startup)
            pass

    # ── Command handlers (from browser) ────────────────────────────

    def toggle_voice(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.pub_voice_enable.publish(msg)
        self.state['voice_enabled'] = enabled
        self._emit_state()
        self.get_logger().info(f'Voice {"enabled" if enabled else "disabled"}')

    def toggle_vision(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.pub_vision_enable.publish(msg)
        self.state['vision_enabled'] = enabled
        self._emit_state()
        self.get_logger().info(f'Vision {"enabled" if enabled else "disabled"}')

    def request_shutdown(self):
        msg = Bool()
        msg.data = True
        self.pub_shutdown.publish(msg)
        self.get_logger().warn('Shutdown requested from dashboard!')

        # Actually shut down the machine after a brief delay
        shutdown_cmd = self.get_parameter('shutdown_command').value
        self.get_logger().warn(f'Executing: {shutdown_cmd}')

        def _do_shutdown():
            time.sleep(2)  # give the UI a moment to show "shutting down"
            try:
                subprocess.run(shutdown_cmd.split(), check=True)
            except Exception as e:
                self.get_logger().error(f'Shutdown failed: {e}')

        threading.Thread(target=_do_shutdown, daemon=True).start()

    def speak(self, text: str):
        msg = String()
        msg.data = text
        self.pub_tts.publish(msg)


# ── Flask app setup ────────────────────────────────────────────────

def create_app(node: DashboardNode):
    """Create Flask app with SocketIO and wire up the ROS2 node."""
    static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    # Fallback: check installed share path
    if not os.path.isdir(static_dir):
        try:
            from ament_index_python.packages import get_package_share_directory
            static_dir = os.path.join(
                get_package_share_directory('jetauto_dashboard'), 'static')
        except Exception:
            static_dir = os.path.join(os.path.dirname(__file__), 'static')

    app = Flask(__name__, static_folder=static_dir)
    app.config['SECRET_KEY'] = 'jetauto-dashboard'
    socketio = node.socketio

    @app.route('/')
    def index():
        return send_from_directory(static_dir, 'index.html')

    @socketio.on('connect')
    def on_connect():
        node.get_logger().info('Browser connected')
        socketio.emit('state', node.state)
        socketio.emit('config', {
            'robot_name': node.get_parameter('robot_name').value,
        })

    @socketio.on('toggle_voice')
    def on_toggle_voice(data):
        node.toggle_voice(data.get('enabled', False))

    @socketio.on('toggle_vision')
    def on_toggle_vision(data):
        node.toggle_vision(data.get('enabled', False))

    @socketio.on('shutdown')
    def on_shutdown():
        node.request_shutdown()

    @socketio.on('speak')
    def on_speak(data):
        node.speak(data.get('text', ''))

    @socketio.on('request_state')
    def on_request_state():
        socketio.emit('state', node.state)

    return app


def main(args=None):
    rclpy.init(args=args)

    # Create SocketIO first, then node (node needs socketio ref)
    socketio = SocketIO(async_mode='threading', cors_allowed_origins='*')
    node = DashboardNode(socketio)
    app = create_app(node)
    socketio.init_app(app)

    host = node.get_parameter('host').value
    port = node.get_parameter('port').value

    # Run ROS2 spin in a background thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    node.get_logger().info(f'Dashboard serving on http://{host}:{port}')

    try:
        socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
