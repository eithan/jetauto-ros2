#!/bin/bash
source /opt/ros/humble/setup.bash

mv=$(ros2 topic echo /ros_robot_controller/battery --once 2>/dev/null | grep "data:" | awk '{print $2}')

if [ -z "$mv" ]; then
    echo "No battery data received (is the robot running?)"
    exit 1
fi

python3 -c "
v = $mv / 1000.0
pct = max(0, min(100, int((v - 9.0) / (12.6 - 9.0) * 100)))
print(f'{pct}% battery remaining ({v:.2f}V)')
"
