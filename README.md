# One-Surgeon-Four-Arm (OSFA) Robotic System

AI-powered robotic surgical system enabling single-surgeon hysterectomy operations. Integrates stereo vision tracking, multi-instrument control, and semi-automatic laparoscope positioning.

---

## 📋 System Overview

**Hardware:**
- UR5 robotic arm with stereo laparoscope
- Uterus manipulator (UDP control)
- Multiple surgical tools (1-4)
- Joystick controller

**Software Stack:**
- ROS Melodic / Python 3.8
- YOLOv5-OBB for oriented bounding box detection
- OpenCV stereo vision
- Real-time multi-tool tracking

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Required: ROS Melodic, Python 3.8+, CUDA 10.2+, MATLAB
# Hardware: UR5 connected at 192.168.3.33
```

### 2. Installation

```bash
# Create Python environment
conda create -n osfa python=3.8
conda activate osfa

# Install dependencies
cd tool_tracking_module/yolov5_obb
pip install -r requirements.txt
pip install rospkg catkin_pkg

# Build ROS packages
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 3. Calibration

**Stereo Camera:**
1. Run camera node and stereo calibration in CLion
2. Target error: < 0.1 pixels
3. Update parameters in tracking scripts

**Hand-Eye:**
1. Collect 20-30 calibration images with static checkerboard
2. Process in MATLAB Camera Calibrator (5mm squares, 3 radial coeffs)
3. Export transformation matrix to control system

### 4. Launch System

```bash
# Terminal 1: ROS core
roscore

# Terminal 2: Robot driver  
roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.3.33

# Terminal 3: Vision tracking
conda activate osfa
cd tool_tracking_module/depthtracker
python3 yolo_final.py --weights path/to/best.pt --conf 0.25

# Terminal 4: Main controller
sudo python3 main.py
```

**Operation:**
- Press `4`: Switch to OSFA mode
- Press `6`: Initialize workspace (required for depth scaling)
- Press `2` + `3`: Set initial position
- Press `1`: Toggle manual/automatic control

---

## 📂 Project Structure

```
OSFA_system/
├── tool_tracking_module/
│   ├── depthtracker/           # Multi-tool tracking scripts
│   └── yolov5_obb/             # OBB detection framework
├── Robot_control_interface/    # Control system & GUI
└── stereo_calibration/         # Camera calibration package
```

---

## 🔧 Training Custom Models

```bash
# 1. Collect training data (~20 mins video)
python3 yolo_final.py --weights best.pt --view-img

# 2. Annotate with LabelMe, convert to YOLO format

# 3. Train model
cd tool_tracking_module/yolov5_obb
python train.py --data cadaver_tools.yaml --cfg yolov5s.yaml --epochs 300
```

---

## 🐛 Troubleshooting

**No /joint_states data:** Restart UR5 control box

**CUDA out of memory:** Use `--device cpu` in tracking script

**Camera swap:** Edit camera indices in stereo node

**Process cleanup:**
```bash
kill -9 $(ps -ef | grep main.py | grep -v grep | awk '{print $2}')
rosnode cleanup
```

---

## 📝 Notes

- Always run workspace initialization (`6`) before operation
- Maintain calibration error < 0.1 pixels for best performance
- Some scripts (main.py, calibration helpers) located in separate repos

---

**System achieves:** 30-50 FPS tracking, <50ms latency, ±2mm depth accuracy