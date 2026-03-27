# One-Surgeon-Four-Arm (OSFA) Robotic System

AI-powered robotic surgical system enabling single-surgeon hysterectomy operations. Integrates stereo vision tracking, multi-instrument control, and semi-automatic laparoscope positioning.

🌐 **Project Page:** [https://lbawmy.github.io/osfa/](https://lbawmy.github.io/osfa/)

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


### 3. Calibration

**Stereo Camera:**
1. Run camera node and stereo calibration in CLion
2. Target error: < 0.1 pixels
3. Update parameters in tracking scripts

**Hand-Eye:**
1. Collect 20-30 calibration images with static checkerboard
2. Process in MATLAB Camera Calibrator (5mm squares, 3 radial coeffs)
3. Export transformation matrix to control system



### 4. AI Subcomponents

#### Tool Tracking

The tool tracking pipeline uses YOLOv5-OBB (Oriented Bounding Box) for real-time multi-tool detection and stereo vision for 3D localization.

**Installation:**
```bash
# Create Python environment from yaml
conda env create -f tool_tracking_module/tracking_env.yaml
conda activate tracking_env

# Install additional ROS dependencies
pip install rospkg catkin_pkg

# Build ROS packages
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

**Model Setup:**
1. Place trained OBB weights (`best.pt`) in `tool_tracking_module/yolov5_obb/runs/train/`
2. To train a custom model, see [Training Custom Models](#-training-custom-models) below

**Run tracking standalone:**
```bash
conda activate tracking_env
cd tool_tracking_module/yolov5_obb/
python match/yolo_track_multi_pure.py --weights runs/train/weights/best.pt --source assets/000.avi --img 640 --device 0 --view-img --classes 1 3 5 --conf 0.1
```

**Stereo Depth Estimation:**
1. Ensure stereo calibration is complete (see Calibration above)
2. Update intrinsic/extrinsic parameters in `tool_tracking_module/depthtracker/yolo_final.py`
3. Kalman filters (`KalmanFilter.py`, `KalmanFilter_multi.py`) smooth 3D position estimates


#### Phase Recognition

Surgical phase recognition classifies the current operative step in real time to enable context-aware automation.

**Setup:**
1. Prepare phase-annotated video data with per-frame labels
2. Train the phase recognition model on annotated surgical recordings
3. Deploy the trained model alongside the tracking pipeline

**Integration:**
- Phase predictions are published via ROS topics for downstream consumption by the control system
- The controller adjusts instrument behavior (e.g., tool selection, motion constraints) based on the recognized phase

#### Main Program Setup

The main control program (`Robot_control_interface/main.py`) requires ROS, custom message packages, and several Python dependencies.

**Installation:**
```bash
# 1. Source ROS environment
source /opt/ros/melodic/setup.bash

# 2. Build custom ROS message packages (yolo_bbox_msg, blackbox_msg)
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# 3. Install Python dependencies for the control interface
pip install pyqt5 opencv-python numpy rospkg catkin_pkg keyboard

# 4. Install UR5 driver
sudo apt install ros-melodic-ur-modern-driver

# 5. Install joystick driver
sudo apt install ros-melodic-joy
```

### 5. Launch System

```bash
# Terminal 1: ROS core
roscore

# Terminal 2: Robot driver for connecting laparoscope manipulator  
roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.3.33

# Terminal 3: Vision tracking
conda activate tracking_env
cd tool_tracking_module/yolov5_obb/
python match/yolo_track_multi_pure.py --weights runs/train/weights/best.pt --source assets/000.avi --img 640 --device 0 --view-img --classes 1 3 5 --conf 0.1

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
│   ├── tracking_env.yaml           # Conda environment for tracking
│   ├── depthtracker/               # Stereo depth tracking & Kalman filters
│   │   ├── yolo_final.py           # Main depth tracking entry point
│   │   └── KalmanFilter*.py        # 3D position smoothing
│   └── yolov5_obb/                 # OBB detection framework
│       ├── train.py                # Model training script
│       ├── detect.py               # Inference script
│       ├── match/                  # Multi-tool tracking scripts
│       ├── data/                   # Dataset configs & hyper-parameters
│       └── runs/train/             # Trained model weights
├── Robot_control_interface/
│   ├── main.py                     # Main entry point (PyQt5 GUI)
│   ├── Mainwindow_ROS_MultiTools.py  # Core controller & ROS integration
│   ├── qt_gui/                     # GUI layouts & joystick widgets
│   ├── robots/                     # UR5 driver & custom ROS messages
│   ├── backend/                    # Robotics utilities
│   ├── stero_camera_node_1080p.py  # Stereo camera publisher
│   └── command.txt                 # Launch command reference
└── stereo_calibration/             # ROS stereo calibration package (C++)
```

---

## 🔧 Training Custom Models

```bash
cd tool_tracking_module/yolov5_obb
python train.py --data ./data/yolov5obb_inhouse_phantom.yaml --epochs 500 --batch-size 32 --img 640 --device 0
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