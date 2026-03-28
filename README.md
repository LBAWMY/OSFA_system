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

## 🚀 Getting Started

### Step 1 · Prerequisites *(~50 min)*

**Software Requirements:**

| Dependency | Install |
|---|---|
| ROS Melodic | [wiki.ros.org/melodic/Installation](https://wiki.ros.org/melodic/Installation/Ubuntu) |
| ROS cv-bridge | `sudo apt install ros-melodic-cv-bridge` |
| UR5 ROS driver | [ur_modern_driver](https://github.com/ros-industrial/ur_modern_driver) — clone into `~/catkin_ws/src` and `catkin_make` |
| Joystick driver | `sudo apt install ros-melodic-joy` |
| PyTorch + CUDA | [pytorch.org/get-started](https://pytorch.org/get-started/locally/) |
| OpenCV | [Build from source](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) — required for ROS Melodic + Python 3 compatibility |
| MATLAB | Required for hand-eye calibration |

**Hardware:**
- UR5 robotic arm connected at `192.168.3.33`
- Stereo laparoscope camera
- USB joystick controller

### Step 2 · Installation *(~30 min)*

#### A. Tool Tracking Environment

```bash
# Create conda environment from yaml
conda env create -f tool_tracking_module/tracking_env.yaml
conda activate tracking_env
pip install rospkg catkin_pkg
```

#### B. Main Control Program

```bash
# Source ROS and build workspace
source /opt/ros/melodic/setup.bash
cd ~/catkin_ws
catkin_make                       # builds custom msgs (yolo_bbox_msg, blackbox_msg)
source devel/setup.bash

# Python dependencies
pip install pyqt5 opencv-python numpy rospkg catkin_pkg keyboard

# ROS packages
sudo apt install ros-melodic-ur-modern-driver ros-melodic-joy
```

### Step 3 · Calibration *(~15 min)*

**Stereo Camera:**
1. Run camera node and stereo calibration in CLion
2. Target error: < 0.1 pixels
3. Update parameters in tracking scripts

**Hand-Eye:**
1. Collect 20-30 calibration images with static checkerboard
2. Process in MATLAB Camera Calibrator (5mm squares, 3 radial coeffs)
3. Export transformation matrix to control system

### Step 4 · Launch System

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

## 🤖 AI Subcomponents

### Tool Tracking *(~10 min setup, ~50 sec inference for demo video clip001.avi)*

The tool tracking pipeline uses YOLOv5-OBB (Oriented Bounding Box) for real-time multi-tool detection and stereo vision for 3D localization.

**Model Setup:**
1. Trained OBB weights (`best.pt`) are located in `tool_tracking_module/yolov5_obb/runs/train/`
2. Download the video assets from [Google Drive](https://drive.google.com/drive/folders/1EdWaH4jMvLIYsDY4C-Mk9EUMAoWp_L82?usp=drive_link) and place them in `tool_tracking_module/yolov5_obb/assets/`
3. To train a custom model, see [Training Custom Models](#-training-custom-models) below

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

### Phase Recognition *(~10 min setup, ~10 sec inference for demo video 020.mp4)*

Surgical phase recognition classifies the current operative step in real time, adapted from [Trans-SVNet](https://github.com/xjgaocs/Trans-SVNet).

**Model Setup:**
1. Pretrained weights (`TeCNO.pth`, `TeCNO_Trans.pth`) are located in `surgical phase_module/model/`
2. Download the [AutoLaparo dataset](https://autolaparo.github.io/) and prepare per-frame label files (see format below)

**Run inference:**
```bash
cd "surgical phase_module"
python run.py \
  --video_path ./data/020.mp4 \
  --label_path ./data/020.txt \
  --save_dir ./data/
```

**Integration:**
- Phase predictions are published via ROS topics for downstream consumption by the control system
- The controller adjusts instrument behavior (e.g., tool selection, motion constraints) based on the recognized phase

📖 For detailed usage, label format, and arguments, see [surgical phase_module/README.md](surgical%20phase_module/README.md)

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
├── surgical phase_module/
│   ├── run.py                      # Main training & evaluation entry point
│   ├── transformer2_3_1.py         # Transformer-based phase model
│   ├── tecno.py                    # TeCNO temporal CNN architecture
│   ├── mstcn.py                    # MS-TCN temporal segmentation
│   ├── trans_SV.py                 # Transformer with surgical video features
│   ├── train_embedding.py          # Feature embedding training
│   ├── generate_LFB.py             # Long-term feature bank generation
│   ├── model/                      # Pretrained weights (TeCNO, TeCNO_Trans)
│   └── data/                       # Sample videos & phase annotations
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

**1. Download the dataset** from Zenodo and create the required folder structure:

```bash
cd tool_tracking_module/yolov5_obb

# Create dataset directory
mkdir -p dataset

# Download dataset from Zenodo and extract into the dataset folder
# https://doi.org/10.5281/zenodo.18875389
# After downloading, place the contents so the structure looks like:
```

```
tool_tracking_module/yolov5_obb/
└── dataset/
    └── dataset_inhouse_phantom/
        ├── train/
        │   └── images/
        └── val/
            └── images/
```

**2. Train the model:**

```bash
cd tool_tracking_module/yolov5_obb
conda activate tracking_env
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

- Always run workspace initialization before operation
- Maintain calibration error < 0.1 pixels for best performance
- Some scripts (main.py, calibration helpers) located in separate repos

---

**System achieves:** 30-50 FPS tracking, <50ms latency, ±2mm depth accuracy