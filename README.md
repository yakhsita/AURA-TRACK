# AURA-TRACK

🚀 **Autonomous Aerial Support for Public Safety & Emergency Response**

## 1. Project Overview
* **Objective:** Build an autonomous, identity-aware drone to assist emergency responders.
* **Core Capabilities:** Locks onto specific identities, recovers from visual occlusions, adapts to crowds, and safely isolates AI control from flight mechanics.

## 2. Hardware Components
1. **Flight Controller:** Pixhawk 2.4.8 (Handles safety-critical flight)
2. **Companion Computer:** NVIDIA Jetson Orin Nano (Handles heavy AI processing)
3. **Vision:** RPI Camera
4. **Environment Sensor:** LIDAR (For elevation and target following)
5. **Propulsion:** 2000kv BLDC Motors with 5-inch propellers
6. **Power:** High energy-density Lithium-Ion Battery
7. **Failsafe:** Manual Radio Transmitter

## 3. Software Stack & Tools
* **Vision & AI:** Python, OpenCV, PyTorch, TensorRT, ByteTrack
* **Autonomy:** ArduPilot / ArduCopter, Mission Planner
* **Middleware & Comm:** ROS 2, MAVProxy, pymavlink
* **Pre-Flight Testing:** Ubuntu/WSL environment running Gazebo and ArduPilot SITL (Highly recommended for safe, simulated testing before deploying to physical hardware).

## 4. Process Flowchart (System Architecture)
```text
[RPI Camera] 
    │
    ▼ (Live Video Feed)
[NVIDIA Jetson Orin Nano]
    │── 1. Visual Processing (Camera Node, Person Detecting Node)
    │── 2. ROS Middleware (Topics, TF Transforms)
    └── 3. Flight Control (Path Planner, Boundary Checks)
    │
    ▼ (MAVLink Serial Protocol)
[Pixhawk 2.4.8 Flight Controller] ◀━━━ [Radio Transmitter] (Emergency Override)
    │── 1. MAVLink Protocol Execution
    │── 2. Altitude Controller
    └── 3. Failsafes 
```

## 5. Execution Plan (Step-by-Step Implementation)
1. **Phase 1: Vision Setup** 
   * Connect the onboard camera to the Jetson Nano to capture live video.
2. **Phase 2: Detection & Tracking** 
   * Run lightweight OpenCV/PyTorch AI models to detect humans in real-time.
3. **Phase 3: Identity Locking** 
   * Implement tracking logic to lock onto a specific identity and re-identify them after occlusions.
4. **Phase 4: Autonomous Flight** 
   * Feed coordinates through the ROS-MAVLink bridge to guide the drone while avoiding obstacles.
5. **Phase 5: Continuous Correction** 
   * Establish a continuous feedback loop between the Jetson's AI predictions and the Pixhawk's stable flight parameters to ensure smooth following.
