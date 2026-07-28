# AURA-TRACK

🚀 **Autonomous Aerial Support for Public Safety & Emergency Response**

AURA-TRACK is an autonomous aerial intelligence system designed to assist public safety and emergency response teams. It provides real-time, identity-aware human tracking in dynamic and high-risk environments, operating independently of static infrastructure.

## 🌟 Key Features
* **Identity-Aware Tracking:** Locks onto a specific target identity, re-identifies the target after visual occlusion, and maintains a crowd-robust following (unlike standard follow-drones that lose targets in crowds).
* **Context-Adaptive Autonomy:** Features motion-aware path prediction, dynamic speed adjustment, adaptive distance regulation, and environment-aware behavior.
* **Safety-First Control:** Employs decoupled AI control with autopilot-enforced safety, collision-aware navigation, and a fail-safe hover mode.
* **Edge-Optimized Intelligence:** Runs on-board AI inference with low latency and energy-efficient execution for real-time responsiveness.

## 🏗️ System Architecture
AURA-TRACK utilizes a layered architecture for real-time perception-driven autonomy on edge hardware:
1. **Vision & Sensing:** An RPI Camera and LIDAR capture live video feeds and elevation data.
2. **Processing (Edge AI):** An NVIDIA Jetson Orin Nano handles heavy visual processing (human detection/locking) and runs ROS middleware.
3. **Flight Control:** A Pixhawk 2.4.8 flight controller receives commands via the MAVLink protocol to execute safety-critical flight maneuvers and altitude control.

## 🛠️ Technology Stack
* **AI & Vision:** OpenCV, PyTorch, TensorRT, ByteTrack
* **Autonomy & Flight Control:** ArduPilot / ArduCopter, Mission Planner, Pixhawk 2.4.8
* **Communication & Middleware:** ROS 2, MAVProxy, pymavlink
* **Hardware:** NVIDIA Jetson Orin Nano, RPI Camera, LIDAR, Lithium-Ion Battery, 2000kv BLDC Motors (5" props)
