#!/bin/bash
# Emergency stop — kills everything and stops the robot
echo "🛑 EMERGENCY STOP"

# 1. Send zero velocity immediately (multiple times to ensure delivery)
for i in 1 2 3; do
  ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
    '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' 2>/dev/null &
done

# 2. Kill our nodes
pkill -f frontier_explorer 2>/dev/null
pkill -f safety_monitor 2>/dev/null

# 3. Cancel any active Nav2 goals
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{}" --cancel 2>/dev/null &

# 4. Kill Nav2
pkill -f navigation_launch 2>/dev/null
pkill -f nav2 2>/dev/null
pkill -f bt_navigator 2>/dev/null
pkill -f controller_server 2>/dev/null
pkill -f planner_server 2>/dev/null

# 5. Kill SLAM
pkill -f slam_toolbox 2>/dev/null

# 6. One more zero velocity for good measure
sleep 0.5
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' 2>/dev/null

echo "✅ All stopped"
