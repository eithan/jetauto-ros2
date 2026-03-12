#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# start-dashboard.sh — Launch JetAuto dashboard + browser
#
# Usage:
#   ./scripts/start-dashboard.sh           # normal
#   ./scripts/start-dashboard.sh --build   # rebuild first
#   ./scripts/start-dashboard.sh --kiosk   # fullscreen kiosk mode
#
# For autostart on boot, add a systemd service (see below).
# ─────────────────────────────────────────────────────────

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WS_DIR="${SCRIPT_DIR}/../.."          # ~/ros2_ws
DASHBOARD_PORT="${DASHBOARD_PORT:-5000}"
DASHBOARD_URL="http://localhost:${DASHBOARD_PORT}"
BROWSER_DELAY=4                        # seconds to wait before opening browser

# ── Flags ──────────────────────────────────────────────
BUILD=false
KIOSK=false
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=true ;;
    --kiosk) KIOSK=true ;;
  esac
done

# ── Source ROS2 ────────────────────────────────────────
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
fi
if [ -f "${WS_DIR}/install/setup.bash" ]; then
  source "${WS_DIR}/install/setup.bash"
fi

# ── Build (optional) ──────────────────────────────────
if [ "$BUILD" = true ]; then
  echo "🔨 Building jetauto_dashboard..."
  cd "$WS_DIR"
  colcon build --packages-select jetauto_dashboard
  source "${WS_DIR}/install/setup.bash"
  echo "✅ Build complete"
fi

# ── Kill leftover dashboard if port is in use ─────────
if lsof -ti:${DASHBOARD_PORT} >/dev/null 2>&1; then
  echo "🧹 Port ${DASHBOARD_PORT} in use — killing leftover process..."
  kill $(lsof -ti:${DASHBOARD_PORT}) 2>/dev/null || true
  sleep 1
fi

# ── Check dependencies ────────────────────────────────
python3 -c "import flask, flask_socketio" 2>/dev/null || {
  echo "📦 Installing Python dependencies..."
  pip3 install flask flask-socketio simple-websocket
}

# ── Launch dashboard node ─────────────────────────────
echo "🚀 Starting dashboard on ${DASHBOARD_URL}"
ros2 launch jetauto_dashboard dashboard.launch.py &
ROS_PID=$!

# ── Wait for server to come up, then open browser ─────
(
  sleep "$BROWSER_DELAY"

  # Wait until the port is actually responding (max 20s)
  for i in $(seq 1 20); do
    if curl -s -o /dev/null -w '' "${DASHBOARD_URL}" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  # Auto-detect display if not set (e.g. running via SSH or systemd)
  if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    # Try common Jetson display values
    for d in :1 :0; do
      if DISPLAY="$d" xdpyinfo >/dev/null 2>&1; then
        export DISPLAY="$d"
        echo "📺 Found display at DISPLAY=$d"
        break
      fi
    done
  fi

  if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
    # Find a working browser
    BROWSER=""
    for candidate in chromium-browser chromium google-chrome firefox; do
      if command -v "$candidate" >/dev/null 2>&1; then
        BROWSER="$candidate"
        break
      fi
    done

    if [ -z "$BROWSER" ]; then
      echo "⚠️  No browser found — dashboard running at ${DASHBOARD_URL}"
    elif [ "$KIOSK" = true ]; then
      "$BROWSER" --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
        --no-first-run --disable-translate --kiosk "${DASHBOARD_URL}" 2>/dev/null &
      echo "🌐 ${BROWSER} opened in kiosk mode"
    else
      "$BROWSER" --start-maximized --no-first-run "${DASHBOARD_URL}" 2>/dev/null &
      echo "🌐 ${BROWSER} opened"
    fi
  else
    echo "⚠️  No display detected — dashboard running headless at ${DASHBOARD_URL}"
  fi
) &

# ── Cleanup on exit ───────────────────────────────────
cleanup() {
  echo ""
  echo "🛑 Shutting down dashboard..."
  kill "$ROS_PID" 2>/dev/null
  # Kill browser kiosk if we started it
  if [ "$KIOSK" = true ] && [ -n "${BROWSER:-}" ]; then
    pkill -f "${BROWSER}.*kiosk.*${DASHBOARD_PORT}" 2>/dev/null || true
  fi
  wait
}
trap cleanup EXIT INT TERM

# ── Keep running ──────────────────────────────────────
wait "$ROS_PID"
