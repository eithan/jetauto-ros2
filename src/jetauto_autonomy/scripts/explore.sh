#!/bin/bash
# Launch full autonomous exploration stack with clean log management
# Usage: ./explore.sh
# E-stop: Ctrl+C (kills everything), or run ./estop.sh from another terminal

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/ros2_ws/src/jetauto-ros2/src/jetauto_autonomy/config"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Cleanup function — runs on Ctrl+C
cleanup() {
    echo -e "\n${RED}🛑 Stopping all nodes...${NC}"
    
    # Send zero velocity FIRST (immediate stop)
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
      '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' 2>/dev/null &
    
    # Kill our specific nodes
    pkill -f frontier_explorer 2>/dev/null
    pkill -f safety_monitor 2>/dev/null
    pkill -f obstacle_injector 2>/dev/null
    
    # Kill Nav2 — must kill all spawned processes
    pkill -f navigation_launch 2>/dev/null
    pkill -f nav2 2>/dev/null
    pkill -f bt_navigator 2>/dev/null
    pkill -f controller_server 2>/dev/null
    pkill -f planner_server 2>/dev/null
    pkill -f behavior_server 2>/dev/null
    pkill -f smoother_server 2>/dev/null
    pkill -f velocity_smoother 2>/dev/null
    pkill -f waypoint_follower 2>/dev/null
    pkill -f lifecycle_manager 2>/dev/null
    pkill -f map_server 2>/dev/null
    
    # Kill SLAM
    pkill -f slam_toolbox 2>/dev/null
    
    # Kill entire process group (catches anything we missed)
    kill -- -$$ 2>/dev/null
    
    # Wait briefly then send zero velocity again
    sleep 0.5
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
      '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' 2>/dev/null
    
    echo -e "${GREEN}✅ All stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "${GREEN}🚗 JetAuto Autonomous Exploration${NC}"
echo -e "${YELLOW}Ctrl+C to emergency stop everything${NC}"
echo ""

# Create log directory
LOG_DIR="$HOME/ros2_ws/logs/explore_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
ln -sfn "$LOG_DIR" "$HOME/ros2_ws/logs/explore_latest"
echo -e "Logs: ${LOG_DIR}"
echo ""

# 1. SLAM
echo -e "${GREEN}[1/4] Starting SLAM...${NC}"
ros2 run slam_toolbox async_slam_toolbox_node --ros-args \
  --params-file "$CONFIG_DIR/slam_params.yaml" \
  --log-level warn \
  > "$LOG_DIR/slam.log" 2>&1 &
SLAM_PID=$!
sleep 3

if ! kill -0 $SLAM_PID 2>/dev/null; then
    echo -e "${RED}SLAM failed to start! Check $LOG_DIR/slam.log${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ SLAM running (PID $SLAM_PID)${NC}"

# 2. Nav2
echo -e "${GREEN}[2/4] Starting Nav2...${NC}"
ros2 launch nav2_bringup navigation_launch.py \
  params_file:="$CONFIG_DIR/nav2_params.yaml" \
  use_sim_time:=false autostart:=true \
  > "$LOG_DIR/nav2.log" 2>&1 &
NAV2_PID=$!
sleep 5

echo -e "${GREEN}  ✓ Nav2 running (PID $NAV2_PID)${NC}"

# 3. Safety monitor
echo -e "${GREEN}[3/5] Starting safety monitor...${NC}"
ros2 run jetauto_autonomy safety_monitor --ros-args --log-level info \
  > "$LOG_DIR/safety.log" 2>&1 &
SAFETY_PID=$!
sleep 1
echo -e "${GREEN}  ✓ Safety monitor running (PID $SAFETY_PID)${NC}"

# 4. Obstacle injector (learns from stuck events → injects into costmap)
echo -e "${GREEN}[4/5] Starting obstacle injector...${NC}"
ros2 run jetauto_autonomy obstacle_injector --ros-args --log-level info \
  > "$LOG_DIR/injector.log" 2>&1 &
INJECTOR_PID=$!
sleep 1
echo -e "${GREEN}  ✓ Obstacle injector running (PID $INJECTOR_PID)${NC}"

# 5. Frontier explorer (foreground — visible output)
echo ""
echo -e "${GREEN}[5/5] Starting frontier explorer...${NC}"
echo -e "${YELLOW}═══════════════════════════════════════${NC}"
echo -e "${YELLOW}  Robot is now exploring autonomously   ${NC}"
echo -e "${YELLOW}  Ctrl+C to stop everything             ${NC}"
echo -e "${YELLOW}═══════════════════════════════════════${NC}"
echo ""

ros2 run jetauto_autonomy frontier_explorer --ros-args --log-level info 2>&1 | tee "$LOG_DIR/frontier.log" &
EXPLORER_PID=$!

# Wait for explorer to finish (or Ctrl+C)
wait $EXPLORER_PID 2>/dev/null

# If explorer exits on its own (exploration complete), clean up
cleanup
