#!/usr/bin/env bash
# ============================================================
# deploy.sh — Sync code to the JetAuto robot and rebuild
#
# Usage:
#   ./deploy.sh                  # sync + build
#   ./deploy.sh --sync-only      # sync without building
#   ./deploy.sh --build-only     # build on robot (no sync)
#
# Configuration: edit the variables below or export them
# before running the script.
# ============================================================

set -euo pipefail

# -- Configuration --
ROBOT_USER="${ROBOT_USER:-ubuntu}"
ROBOT_HOST="${ROBOT_HOST:-192.168.1.50}"
ROBOT_WS="${ROBOT_WS:-/home/ubuntu/ros2_ws}"
SSH_KEY="${SSH_KEY:-}"  # e.g. ~/.ssh/jetauto_rsa (empty = default key)

# -- Derived --
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
SSH_OPTS=""
if [[ -n "${SSH_KEY}" ]]; then
    SSH_OPTS="-i ${SSH_KEY}"
fi
SSH_TARGET="${ROBOT_USER}@${ROBOT_HOST}"

# -- Colors --
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[deploy]${NC} $*"; }
err()  { echo -e "${RED}[deploy]${NC} $*" >&2; }

# -- Functions --
do_sync() {
    log "Syncing src/ to ${SSH_TARGET}:${ROBOT_WS}/src/jetauto-ros2/src/"
    rsync -avz --delete \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        ${SSH_KEY:+-e "ssh -i ${SSH_KEY}"} \
        "${SRC_DIR}/" \
        "${SSH_TARGET}:${ROBOT_WS}/src/jetauto-ros2/src/"

    log "Syncing requirements.txt"
    rsync -avz \
        ${SSH_KEY:+-e "ssh -i ${SSH_KEY}"} \
        "${SCRIPT_DIR}/requirements.txt" \
        "${SSH_TARGET}:${ROBOT_WS}/src/jetauto-ros2/"

    log "Sync complete ✓"
}

do_build() {
    log "Building on robot..."
    # shellcheck disable=SC2029
    ssh ${SSH_OPTS} "${SSH_TARGET}" bash -lc "'
        source /opt/ros/humble/setup.bash &&
        cd ${ROBOT_WS} &&
        colcon build --packages-select jetauto_msgs jetauto_vision jetauto_tts jetauto_voice --symlink-install &&
        echo \"Build complete ✓\"
    '"
}

# -- Main --
case "${1:-all}" in
    --sync-only)
        do_sync
        ;;
    --build-only)
        do_build
        ;;
    all|"")
        do_sync
        do_build
        ;;
    *)
        err "Unknown option: $1"
        echo "Usage: $0 [--sync-only|--build-only]"
        exit 1
        ;;
esac

log "Done! Launch with:"
log "  ssh ${SSH_TARGET} 'source ${ROBOT_WS}/install/setup.bash && ros2 launch jetauto_tts tts_launch.py'"
