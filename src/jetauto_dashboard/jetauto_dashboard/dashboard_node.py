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
        self.start_time = time.monotonic()

        # -- Parameters --
        self.declare_parameter('port', 5000)
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('robot_name', 'JARVIS')
        self.declare_parameter('system_monitor_interval', 180.0)  # 3 minutes
        self.declare_parameter('shutdown_command', 'sudo /sbin/poweroff')

        # -- Managed subprocess handles --
        # Each node group is managed independently to avoid lifecycle
        # collisions when voice and vision are toggled separately.
        self._voice_proc = None     # voice_commander_node only
        self._detector_proc = None  # detector_node only
        self._tts_proc = None       # tts_node only (shared by voice & vision)
        self._caption_proc = None   # caption_node (Florence-2, vision only)

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
        # TRANSIENT_LOCAL so late-joining subscriber gets last value — avoids
        # losing the very first 'speaking' publish after tts_node activates.
        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.create_subscription(
            String, '/jetauto/voice/state', self._on_voice_state, qos_latched)
        # Mirror /tts/speaking (Bool) → voice_state as an additional reliability layer
        self.create_subscription(
            Bool, '/tts/speaking', self._on_tts_speaking, qos_reliable)
        # Florence-2 scene captions → TTS
        self.create_subscription(
            String, '/scene_caption', self._on_scene_caption, qos_best_effort)

        if HAS_DETECTION_MSGS:
            self.create_subscription(
                DetectedObjectArray, '/detected_objects',
                self._on_detections, qos_best_effort)

        # -- Publishers --
        self.pub_voice_enable = self.create_publisher(Bool, '/jetauto/voice/enable', 1)
        self.pub_vision_enable = self.create_publisher(Bool, '/jetauto/detection/enable', 1)
        self.pub_shutdown = self.create_publisher(Bool, '/jetauto/shutdown', 1)
        self.pub_tts = self.create_publisher(String, '/tts/speak', 1)
        self.pub_tts_cancel = self.create_publisher(Bool, '/tts/cancel', 1)
        self.pub_volume = self.create_publisher(Float32, '/tts/set_volume', 1)

        # -- Uptime timer --
        self.create_timer(1.0, self._tick_uptime)

        # -- Auto-enable voice on startup --
        self._startup_done = False
        self._startup_timer = self.create_timer(3.0, self._on_startup)

        # -- System monitor (reads temps/battery from sysfs directly) --
        monitor_interval = self.get_parameter('system_monitor_interval').value
        self.create_timer(monitor_interval, self._poll_system)
        # Run once at startup
        self._poll_system()

        self.get_logger().info('Dashboard node initialized')

    # ── Subscription callbacks ─────────────────────────────────────

    def _on_battery_mv(self, msg: Float32):
        """Convert millivolts from robot controller to battery percentage.

        3S LiPo non-linear discharge curve (per-cell voltages × 3):
          4.20V = 100%    (12.60V pack)
          4.10V =  90%    (12.30V)
          3.95V =  75%    (11.85V)
          3.80V =  55%    (11.40V)
          3.70V =  35%    (11.10V)
          3.60V =  20%    (10.80V)
          3.50V =  10%    (10.50V)
          3.40V =   5%    (10.20V)
          3.30V =   2%    ( 9.90V)  ← BMS starts beeping here
          3.00V =   0%    ( 9.00V)
        """
        voltage = msg.data / 1000.0

        # Piecewise-linear interpolation from real LiPo discharge curve
        # (pack_voltage, percentage) pairs
        curve = [
            (12.60, 100), (12.30, 90), (11.85, 75), (11.40, 55),
            (11.10, 35), (10.80, 20), (10.50, 10), (10.20, 5),
            (9.90, 2), (9.00, 0),
        ]

        if voltage >= curve[0][0]:
            pct = 100
        elif voltage <= curve[-1][0]:
            pct = 0
        else:
            for i in range(len(curve) - 1):
                v_hi, p_hi = curve[i]
                v_lo, p_lo = curve[i + 1]
                if voltage >= v_lo:
                    # Linear interpolation within this segment
                    ratio = (voltage - v_lo) / (v_hi - v_lo)
                    pct = int(p_lo + ratio * (p_hi - p_lo))
                    break
            else:
                pct = 0
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

    def _on_tts_speaking(self, msg: Bool):
        """Mirror /tts/speaking → voice_state for reliable face animation.

        Acts as a belt-and-suspenders alongside /jetauto/voice/state.
        Only updates state when TTS starts speaking (always override to
        'speaking') or finishes speaking (revert to 'idle' only if we were
        previously in 'speaking' state — avoids clobbering 'listening' etc
        set by the voice commander).
        """
        if msg.data:
            if self.state['voice_state'] != 'speaking':
                self.state['voice_state'] = 'speaking'
                self._emit_state()
        elif self.state['voice_state'] == 'speaking':
            self.state['voice_state'] = 'idle'
            self._emit_state()

    def _on_scene_caption(self, msg: String):
        """Store latest caption — spoken on demand only (via voice command)."""
        pass  # caption_node → voice_commander handles on-demand speech

    def _on_startup(self):
        """Auto-enable voice once on startup."""
        if not self._startup_done:
            self._startup_done = True
            self._startup_timer.cancel()
            self.toggle_voice(True)
            self.get_logger().info('Auto-enabled voice on startup')

    def _on_detections(self, msg):
        """Update detection state when detected labels change.

        Uses a per-label dict so the same object detected repeatedly on
        consecutive frames does NOT flood the browser with new entries.
        Only emits a state update when the set of detected labels changes.
        """
        if not msg.objects:
            return

        ts = datetime.now().strftime('%H:%M:%S')

        # Build per-label dict — keep best confidence per label in this frame
        frame_dets: dict = {}
        for obj in msg.objects:
            label = obj.label
            conf = round(obj.confidence, 2)
            if label not in frame_dets or conf > frame_dets[label]['confidence']:
                frame_dets[label] = {'label': label, 'confidence': conf, 'time': ts}

        new_labels = frozenset(frame_dets.keys())
        old_labels = getattr(self, '_last_det_labels', frozenset())

        if new_labels != old_labels:
            self._last_det_labels = new_labels
            self.state['detections'] = list(frame_dets.values())[:5]
            self._emit_state()

    def _tick_uptime(self):
        self.state['uptime'] = int(time.monotonic() - self.start_time)
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

    # ── Process management (launch/kill individual node groups) ─────
    #
    # Three independent processes avoid the lifecycle collision that
    # occurred when voice_control.launch.py bundled detector_node +
    # tts_node alongside voice_commander_node, and then tts_launch.py
    # tried to start duplicate detector_node + tts_node instances.

    @staticmethod
    def _proc_alive(proc):
        """Return True if a subprocess handle is running."""
        return proc is not None and proc.poll() is None

    def _kill_proc(self, proc, label: str):
        """Kill a subprocess gracefully: SIGINT → SIGTERM → SIGKILL."""
        if not self._proc_alive(proc):
            return
        pid = proc.pid
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGINT)
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
        self.get_logger().info(f'{label} stopped')

    # -- Voice commander --

    def _launch_voice(self):
        """Launch voice_commander_node only."""
        if self._proc_alive(self._voice_proc):
            return
        try:
            self._voice_proc = subprocess.Popen(
                ['ros2', 'launch', 'jetauto_voice', 'voice_control.launch.py',
                 'mic_device_index:=1', 'vad_aggressiveness:=1'],
                preexec_fn=os.setsid,
            )
            self.get_logger().info(f'Voice commander launched (pid {self._voice_proc.pid})')
        except Exception as e:
            self.get_logger().error(f'Failed to launch voice commander: {e}')

    def _kill_voice(self):
        self._kill_proc(self._voice_proc, 'Voice commander')
        self._voice_proc = None

    # -- Detector --

    def _launch_detector(self):
        """Launch detector_node only (via vision_launch.py)."""
        if self._proc_alive(self._detector_proc):
            return
        try:
            self._detector_proc = subprocess.Popen(
                ['ros2', 'launch', 'jetauto_vision', 'vision_launch.py'],
                preexec_fn=os.setsid,
            )
            self.get_logger().info(f'Detector launched (pid {self._detector_proc.pid})')
        except Exception as e:
            self.get_logger().error(f'Failed to launch detector: {e}')

    def _kill_detector(self):
        self._kill_proc(self._detector_proc, 'Detector')
        self._detector_proc = None

    # -- TTS (shared resource) --

    def _launch_tts(self):
        """Launch tts_node only (via tts_only.launch.py)."""
        if self._proc_alive(self._tts_proc):
            return
        try:
            self._tts_proc = subprocess.Popen(
                ['ros2', 'launch', 'jetauto_tts', 'tts_only.launch.py'],
                preexec_fn=os.setsid,
            )
            self.get_logger().info(f'TTS launched (pid {self._tts_proc.pid})')
        except Exception as e:
            self.get_logger().error(f'Failed to launch TTS: {e}')

    def _kill_tts(self):
        self._kill_proc(self._tts_proc, 'TTS')
        self._tts_proc = None

    def _maybe_kill_tts(self):
        """Kill tts_node only if neither voice nor vision needs it."""
        if not self.state.get('voice_enabled') and not self.state.get('vision_enabled'):
            self._kill_tts()

    def _maybe_kill_detector(self):
        """Kill detector_node only if neither voice nor vision needs it."""
        if not self.state.get('voice_enabled') and not self.state.get('vision_enabled'):
            self._kill_detector()

    # -- Caption (Florence-2, vision only) --

    def _launch_caption(self):
        """Launch caption_node (Florence-2 scene captioning)."""
        if self._proc_alive(self._caption_proc):
            return
        try:
            self._caption_proc = subprocess.Popen(
                ['ros2', 'launch', 'jetauto_vision', 'caption_launch.py'],
                preexec_fn=os.setsid,
            )
            self.get_logger().info(f'Caption launched (pid {self._caption_proc.pid})')
        except Exception as e:
            self.get_logger().error(f'Failed to launch caption: {e}')

    def _kill_caption(self):
        self._kill_proc(self._caption_proc, 'Caption')
        self._caption_proc = None

    def cleanup_subprocesses(self):
        """Kill any managed subprocesses on shutdown."""
        self._kill_voice()
        self._kill_detector()
        self._kill_caption()
        self._kill_tts()

    # ── Command handlers (from browser) ────────────────────────────

    def toggle_voice(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.pub_voice_enable.publish(msg)
        self.state['voice_enabled'] = enabled
        self._emit_state()

        if enabled:
            self._launch_tts()    # TTS needed for voice feedback ("Yes?", etc.)
            self._launch_voice()
        else:
            self._kill_voice()
            self._maybe_kill_tts()  # only kill TTS if vision also off

        self.get_logger().info(f'Voice {"enabled" if enabled else "disabled"}')

    def toggle_vision(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.pub_vision_enable.publish(msg)
        self.state['vision_enabled'] = enabled
        self._emit_state()

        if enabled:
            self._launch_tts()       # TTS needed for announcements
            self._launch_detector()
            self._launch_caption()   # Florence-2 runs silently; speaks on demand
            # Re-publish enable after detector has time to subscribe.
            def _re_enable():
                time.sleep(15)
                if self.state.get('vision_enabled'):
                    self.pub_vision_enable.publish(msg)
            threading.Thread(target=_re_enable, daemon=True).start()
        else:
            # Cancel any queued/current TTS immediately
            cancel_msg = Bool()
            cancel_msg.data = True
            self.pub_tts_cancel.publish(cancel_msg)
            self._kill_caption()
            self._maybe_kill_detector()  # only kill if voice also off
            self._maybe_kill_tts()       # only kill TTS if voice also off

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

    @socketio.on('set_volume')
    def on_set_volume(data):
        vol = max(0.0, min(1.0, float(data.get('volume', 0.8))))
        msg = Float32()
        msg.data = vol
        node.pub_volume.publish(msg)

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
