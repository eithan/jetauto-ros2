# JetAuto Autonomous Home Robot — Master Plan

*Created: 2026-03-18*

## Current State

The `jetauto-ros2` repo already has:
- **detector_node** — YOLOv8 real-time object detection → dashboard
- **caption_node** — Florence-2 rich scene captioning → TTS
- **tts_node** — pyttsx3 spoken announcements
- Launch files, YAML configs, working ROS2 ament_python scaffold
- YOLOv8s model downloaded on robot

What's missing: navigation, mapping, exploration behavior, face recognition, autonomous movement.

---

## Phase 1: SLAM — Teach the Robot to See Space
**Goal:** Robot builds a 2D map of its environment in real-time.

**Stack:** `slam_toolbox` (2D SLAM — lighter, battle-tested, sufficient for a ground robot)
- Alternative: RTAB-Map (3D, heavier) if richer spatial understanding is needed later

**Tasks:**
1. Install `ros-humble-slam-toolbox`
2. Configure lidar topic remapping to JetAuto's lidar topic (likely `/scan`)
3. Configure depth camera topics (likely `/camera/depth/image_raw`, `/camera/color/image_raw`)
4. Create SLAM launch file alongside existing nodes
5. Test: teleoperate robot around a room, verify map builds in RViz2

**Critical:** Need a working `tf` tree. Transforms from `base_link` → `laser_frame` → `camera_frame` must be accurate and published. Check if Hiwonder provides a URDF or static transform publisher.

**Estimated effort:** 1-2 sessions

---

## Phase 2: Nav2 — Teach the Robot to Drive Itself
**Goal:** Robot navigates to any map point autonomously, avoiding obstacles.

**Stack:** Nav2 (`ros-humble-navigation2`)
- Path planning (A*, NavFn, Smac), obstacle avoidance (DWB/MPPI), costmap layers, recovery behaviors

**Tasks:**
1. Install: `sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup`
2. Create Nav2 params YAML tuned for JetAuto dimensions (robot radius, max speeds, acceleration limits)
3. Set up costmap — lidar for obstacle layer, depth camera as secondary obstacle source
4. Create nav launch file: SLAM + Nav2 + existing perception nodes
5. Test: send nav goal via RViz2, watch robot drive there avoiding obstacles
6. Tune controller parameters until movement is smooth

**Critical:** TF tree must be correct (from Phase 1). Without correct transforms, Nav2 produces garbage.

**Estimated effort:** 2-3 sessions (TF setup is usually the time sink)

---

## Phase 3: Frontier Exploration — Intelligent Wandering
**Goal:** Robot autonomously explores unknown space.

**Stack:** `m-explore-ros2` (explore_lite port for ROS2)

**Tasks:**
1. Install or clone `m-explore-ros2`
2. Configure to use SLAM costmap
3. Launch: SLAM + Nav2 + explore_lite + perception stack
4. Test: put robot in a room, let it go — should systematically explore until space is mapped
5. Add "search mode" topic — when active, robot explores; when target found, stops exploring

**Addition: Exploration budget.** Configurable max exploration time or area. Robot shouldn't drive forever if it can't find the target. After budget exhausted → return to start, report "couldn't find [target]."

**What this gives you:** Combined with existing YOLO detector, the robot is now autonomously scanning for objects/people while exploring. This is where it starts feeling autonomous.

**Estimated effort:** 1-2 sessions

---

## Phase 4: Search Behavior Controller — The Brain
**Goal:** Orchestrate everything — explore, detect, approach, interact.

**Architecture:**
```
                    ┌──────────────────┐
                    │   search_brain   │
                    │ (state machine)  │
                    └────────┬─────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌───────────┐
    │ explore_lite │   │  detector   │   │ tts_node  │
    │ (Nav2 goals) │   │  (YOLOv8)  │   │ (announce)│
    └─────────────┘   └─────────────┘   └───────────┘
```

**States:**
- `IDLE` — waiting for search command
- `EXPLORING` — frontier exploration active, scanning with YOLO
- `TARGET_DETECTED` — target seen, pausing exploration
- `APPROACHING` — navigating to target's estimated position
- `CONFIRMING` — close to target, verifying detection
- `INTERACTING` — speaking, following, or waiting for command
- `RETURNING` — going back to start position

**Tasks:**
1. New node: `search_brain_node.py` — subscribes to detections, publishes Nav2 goals, controls explore_lite
2. State transitions with hysteresis (require N consecutive frames before switching — no flickering)
3. Target position estimation: depth camera + TF → project bounding box center to map coordinates
4. "Approach" behavior: Nav2 goal ~1m in front of target (not ON target)
5. TTS integration: announce state changes ("Looking for a person...", "I see someone!", "Found you!")

**Estimated effort:** 2-3 sessions

---

## Phase 5: Face Recognition — Know WHO It Found
**Goal:** Identify specific people, not just "a person."

**Stack:** InsightFace (optimized for edge, includes detection + alignment + recognition)

**Pipeline:**
1. YOLO detects person
2. Crop upper portion of bounding box (head region)
3. Face detection → alignment → embedding extraction
4. Compare embedding against known embeddings (cosine similarity)
5. Match > threshold → publish recognized name
6. Robot greets by name: "Hey Eithan!"

**Enrollment flow:**
- Take 5-10 photos per person from different angles
- Run through embedding model
- Average embeddings → store as person's template (`.npy` files)
- Simple CLI script for enrollment

**Privacy:** All embeddings stored locally on robot. No cloud.

**Estimated effort:** 2 sessions

---

## Phase 6: Visual Memory & Room Understanding
**Goal:** Remember where people/objects were seen, search intelligently.

**Visual Memory:**
- SQLite database on robot
- Entries: `{label, type (person/object), map_x, map_y, room, timestamp}`
- Updated on every detection with known map position
- Query: "Where was Eithan last seen?" → check DB → navigate to stored coordinates

**Room Classification (rule-based first):**
- YOLO sees stove/fridge → kitchen
- Sees bed → bedroom
- Sees desk/monitor → office
- Simple, reliable, no extra model needed
- Can upgrade to CLIP-based later if needed

**Person Following (reactive control):**
- Track detected person's bounding box center
- Proportional controller: person left → turn left, right → turn right
- Depth camera maintains ~1.2m distance
- Publish `/cmd_vel` directly (NOT Nav2 — following needs reactive control, not path planning)
- **Safety:** Always check lidar for obstacles even while following

**Estimated effort:** 3-4 sessions

---

## Resource Budget (Orin Nano 8GB)

| Component | GPU Memory | CPU | Notes |
|-----------|-----------|-----|-------|
| YOLOv8s | ~500MB | Low | Always on |
| Florence-2-base | ~1.5GB | Moderate | Time-share, not simultaneous with face recog |
| SLAM (slam_toolbox) | 0 | Moderate | CPU only |
| Nav2 | 0 | Moderate | CPU only |
| InsightFace | ~300MB | Low | Only when YOLO detects person |
| TTS (pyttsx3) | 0 | Low | On-demand |

**Key constraint:** Florence-2 + YOLO + InsightFace simultaneously may push GPU memory. Solution: time-share. During autonomous search, run YOLO + InsightFace. Florence-2 captioning becomes optional/on-demand during search mode.

---

## Safety Behaviors (All Phases)

- **Emergency stop:** lidar detects obstacle < 0.2m → immediate halt
- **Battery monitoring:** return to start if battery drops below threshold
- **Stuck detection:** no movement for 30s despite trying → recovery behavior
- **Velocity limits:** appropriate for indoor use
- **Exploration budget:** max time/area before giving up and returning

---

## Design Decisions

1. **2D SLAM over RTAB-Map** — robot drives on floors, 2D is simpler and faster
2. **Skip emotional states** until Phases 1-5 work reliably — curiosity/boredom floats add complexity for zero functional value at this stage
3. **Use existing ROS2 packages** (explore_lite, slam_toolbox, nav2) — write only glue code
4. **Person following uses direct `/cmd_vel`**, not Nav2 — following a moving person needs reactive control
5. **InsightFace over FaceNet** — optimized for edge, less glue code needed
6. **Rule-based room classification first** — simple, reliable, upgradeable later

---

## Build Timeline

```
Phase 1 (SLAM)          ████░░░░░░  Week 1-2
Phase 2 (Nav2)           ██████░░░░  Week 2-4
Phase 3 (Exploration)    ████░░░░░░  Week 4-5
Phase 4 (Search Brain)   ██████████  Week 5-7
Phase 5 (Face Recog)     ██████░░░░  Week 7-8
Phase 6 (Memory/Rooms)   ████████░░  Week 9+
```

Each phase produces a working, testable robot. Can stop after any phase and have something useful.
