#!/usr/bin/env bash
# run_sim.sh — full integration sim run (detector + sim node)
#
# Usage:
#   ./scripts/run_sim.sh               # GPU (device=0), default model
#   ./scripts/run_sim.sh --cpu         # force CPU (useful on laptop/dev)
#   ./scripts/run_sim.sh --model yolov8m.pt --cpu
#
# Requires: ROS2 Jazzy, colcon workspace built, ultralytics installed
# Log: /tmp/jetauto_sim.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARAMS_FILE="${WORKSPACE}/src/jetauto_vision/config/vision_params.yaml"
LOG_FILE="/tmp/jetauto_sim.log"

# ── Argument parsing ─────────────────────────────────────────────── #
DEVICE=""
MODEL_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)   DEVICE="cpu"; shift ;;
        --model) MODEL_OVERRIDE="$2"; shift 2 ;;
        *)       echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Source ROS2 + workspace ──────────────────────────────────────── #
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${WORKSPACE}/install/setup.bash"

# ── Build extra ros-args ─────────────────────────────────────────── #
EXTRA_ARGS=()
[[ -n "$DEVICE" ]]         && EXTRA_ARGS+=(-p "device:=${DEVICE}")
[[ -n "$MODEL_OVERRIDE" ]] && EXTRA_ARGS+=(-p "model_name:=${MODEL_OVERRIDE}")

# ── Cleanup on exit ──────────────────────────────────────────────── #
DETECTOR_PID=""
cleanup() {
    if [[ -n "$DETECTOR_PID" ]] && kill -0 "$DETECTOR_PID" 2>/dev/null; then
        echo "[sim] Stopping detector (pid $DETECTOR_PID)..."
        kill "$DETECTOR_PID" 2>/dev/null || true
        wait "$DETECTOR_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── Start detector ───────────────────────────────────────────────── #
echo "[sim] Starting detector node..."
rm -f "$LOG_FILE"

ros2 run jetauto_vision detector_node \
    --ros-args \
    --params-file "${PARAMS_FILE}" \
    -p image_topic:=/camera/color/image_raw \
    "${EXTRA_ARGS[@]}" \
    &> /tmp/jetauto_detector.log &
DETECTOR_PID=$!

# ── Wait for node to appear ──────────────────────────────────────── #
echo "[sim] Waiting for detector node to register..."
for i in $(seq 1 20); do
    if ros2 node list 2>/dev/null | grep -q detector_node; then
        break
    fi
    sleep 0.5
    if [[ $i -eq 20 ]]; then
        echo "[sim] ERROR: detector_node never appeared. Check /tmp/jetauto_detector.log"
        exit 1
    fi
done

# ── Lifecycle: configure → activate ─────────────────────────────── #
echo "[sim] Configuring detector..."
ros2 lifecycle set /detector_node configure
echo "[sim] Activating detector..."
ros2 lifecycle set /detector_node activate

# ── Run sim ──────────────────────────────────────────────────────── #
echo "[sim] Launching scenario runner..."
echo ""
ros2 run jetauto_sim sim_node

# ── Print log path ───────────────────────────────────────────────── #
echo ""
echo "[sim] Full log: ${LOG_FILE}"
