#!/usr/bin/env python3
"""
Dashboard Node — serves a full-screen web control panel for the JetAuto robot.

Runs a Flask + SocketIO web server alongside a ROS2 node. The browser connects
via WebSocket for real-time state updates and sends commands back to ROS2.
"""

import os
import glob
import time
import signal
import subprocess
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Bool, Float32, UInt16, String

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
        self.declare_parameter('system_monitor_interval', 180.0)  # 3 minutes
        self.declare_parameter('shutdown_command', 'sudo /sbin/poweroff')

        # -- Managed subprocess handles --
        self._voice_proc = None
        self._vision_proc = None

        # -- State --
        self.state = {
            'battery': None,
            'battery_voltage': None,
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
        # Battery: raw millivolts (UInt16) from the robot controller
        self.create_subscription(
            UInt16, '/ros_robot_controller/battery', self._on_battery_mv, qos_best_effort)
        # Also accept pre-computed percentage on /jetauto/battery
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

    def _on_battery_mv(self, msg: Float32):
        """Convert millivolts from robot controller to battery percentage.

        3S LiPo: 9.0V (0%) to 12.6V (100%).
        """
        voltage = msg.data / 1000.0
        pct = max(0, min(100, int((voltage - 9.0) / (12.6 - 9.0) * 100)))
        self.state['battery'] = float(pct)
        self.state['battery_voltage'] = round(voltage, 2)
        self._emit_state()

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
        """Append detected objects to the rolling log (max 5).

        Throttled: only emits to the browser if the set of detected labels
        actually changed, or at most once per second.
        """
        if not msg.objects:
            return

        ts = datetime.now().strftime('%H:%M:%S')
        new_labels = set()
        for obj in msg.objects:
            entry = {
                'label': obj.label,
                'confidence': round(obj.confidence, 2),
                'time': ts,
            }
            new_labels.add(obj.label)
            self.state['detections'].append(entry)

        # Keep last 5
        self.state['detections'] = self.state['detections'][-5:]

        # Throttle: only emit if labels changed or >1s since last emit
        now = time.time()
        old_labels = getattr(self, '_last_det_labels', set())
        last_emit = getattr(self, '_last_det_emit', 0)
        if new_labels != old_labels or (now - last_emit) > 1.0:
            self._last_det_labels = new_labels
            self._last_det_emit = now
            self._emit_state()

    def _tick_uptime(self):
        self.state['uptime'] = int(time.time() - self.start_time)
        # Emit every 5 seconds to avoid flooding
        if self.state['uptime'] % 5 == 0:
            self._emit_state()

    # ── System monitor (direct sysfs reads) ────────────────────────

    def _poll_system(self):
        """Read CPU/GPU temps and battery from sysfs. Falls back gracefully."""
        cpu_temp = self._read_jetson_temp('CPU')
        if cpu_temp is not None:
            self.state['cpu_temp'] = cpu_temp

        gpu_temp = self._read_jetson_temp('GPU')
        if gpu_temp is not None:
            self.state['gpu_temp'] = gpu_temp

        battery = self._read_battery()
        if battery is not None:
            self.state['battery'] = battery

        self._emit_state()

    def _read_jetson_temp(self, name: str):
        """Read temperature from Jetson thermal zones by name."""
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
                    temp = raw / 1000.0 if raw > 1000 else float(raw)
                    return round(temp, 1)
        except Exception:
            pass
        return None

    def _read_battery(self):
        """Read battery capacity from power_supply sysfs."""
        try:
            for ps_dir in glob.glob('/sys/class/power_supply/*'):
                cap_path = os.path.join(ps_dir, 'capacity')
                type_path = os.path.join(ps_dir, 'type')
                if not os.path.exists(cap_path):
                    continue
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
            pass  # socketio.server not initialized yet

    # ── Process management (launch/kill voice & vision nodes) ──────

    def _launch_voice(self):
        """Launch the voice control pipeline via ros2 launch."""
        if self._voice_proc is not None and self._voice_proc.poll() is None:
            return  # already running
        try:
            self._voice_proc = subprocess.Popen(
                ['ros2', 'launch', 'jetauto_voice', 'voice_control.launch.py',
                 'mic_device_index:=1', 'vad_aggressiveness:=2'],
                preexec_fn=os.setsid,
            )
            self.get_logger().info(f'Voice pipeline launched (pid {self._voice_proc.pid})')
        except Exception as e:
            self.get_logger().error(f'Failed to launch voice: {e}')

    def _kill_voice(self):
        """Kill the voice control pipeline gracefully."""
        if self._voice_proc is not None and self._voice_proc.poll() is None:
            pid = self._voice_proc.pid
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGINT)
                try:
                    self._voice_proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        self._voice_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
            self.get_logger().info('Voice pipeline stopped')
        self._voice_proc = None

    def _launch_vision(self):
        """Launch the vision+TTS pipeline (auto-enables detection via launch param)."""
        if self._vision_proc is not None and self._vision_proc.poll() is None:
            return  # already running
        try:
            self._vision_proc = subprocess.Popen(
                ['ros2', 'launch', 'jetauto_tts', 'tts_launch.py'],
                preexec_fn=os.setsid,
            )
            self.get_logger().info(f'Vision pipeline launched (pid {self._vision_proc.pid})')
        except Exception as e:
            self.get_logger().error(f'Failed to launch vision: {e}')

    def _kill_vision(self):
        """Kill the vision detection pipeline gracefully.

        Uses SIGINT first (lets camera driver release cleanly),
        then SIGTERM, then SIGKILL as last resort.
        """
        if self._vision_proc is not None and self._vision_proc.poll() is None:
            pid = self._vision_proc.pid
            try:
                pgid = os.getpgid(pid)
                # SIGINT first — ROS2 launch handles this gracefully
                os.killpg(pgid, signal.SIGINT)
                try:
                    self._vision_proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        self._vision_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
            except Exception:
                pass
            self.get_logger().info('Vision pipeline stopped')
        self._vision_proc = None

    def cleanup_subprocesses(self):
        """Kill any managed subprocesses on shutdown."""
        self._kill_voice()
        self._kill_vision()

    # ── Command handlers (from browser) ────────────────────────────

    def toggle_voice(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.pub_voice_enable.publish(msg)
        self.state['voice_enabled'] = enabled
        self._emit_state()

        if enabled:
            self._launch_voice()
        else:
            self._kill_voice()

        self.get_logger().info(f'Voice {"enabled" if enabled else "disabled"}')

    def toggle_vision(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.pub_vision_enable.publish(msg)
        self.state['vision_enabled'] = enabled
        self._emit_state()

        if enabled:
            self._launch_vision()
        else:
            self._kill_vision()

        self.get_logger().info(f'Vision {"enabled" if enabled else "disabled"}')

    def request_shutdown(self):
        msg = Bool()
        msg.data = True
        self.pub_shutdown.publish(msg)
        self.get_logger().warn('Shutdown requested from dashboard!')

        shutdown_cmd = self.get_parameter('shutdown_command').value
        self.get_logger().warn(f'Executing: {shutdown_cmd}')

        def _do_shutdown():
            time.sleep(2)
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

    @socketio.on('refresh_stats')
    def on_refresh_stats():
        """Force an immediate sysfs read and push to browser."""
        node._poll_system()
        socketio.emit('state', node.state)

    @socketio.on('quit')
    def on_quit():
        """Close the dashboard cleanly — kills everything and exits."""
        node.get_logger().info('Quit requested from dashboard UI')

        def _do_quit():
            time.sleep(0.5)
            node.cleanup_subprocesses()
            # Use os._exit to avoid rclpy spin thread errors
            os._exit(0)

        threading.Thread(target=_do_quit, daemon=True).start()

    return app


def main(args=None):
    rclpy.init(args=args)

    socketio = SocketIO(async_mode='threading', cors_allowed_origins='*')
    node = DashboardNode(socketio)
    app = create_app(node)
    socketio.init_app(app)

    host = node.get_parameter('host').value
    port = node.get_parameter('port').value

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    node.get_logger().info(f'Dashboard serving on http://{host}:{port}')

    try:
        socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup_subprocesses()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
