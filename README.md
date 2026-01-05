# OSFA_system

One-Surgical-Four-Arms system for robotic-assisted hysterectomy. This repository contains the workflows for calibration, YOLO-based tool detection, and real-time robotic control used during cadaveric experiments.

---

## 📋 System Overview

### Hardware

* **Robotics:** UR5 Robot Arm, Uterus Manipulator, Surgical Tools (1–4).
* **Vision:** 2x Stereo Cameras, 4x Tripods, 12x9x5 Checkerboard.
* **Control:** Computer (Ubuntu), Joystick, Phantom, Ruler.

### Software

* **Framework:** ROS Melodic / Python 3.
* **Vision:** YOLOv5 (Stereo matching and monocular tracking).
* **Calibration:** OpenCV (Python) and MATLAB Camera Calibrator.

---

## 🛠 1. Calibration Workflow

### 1.1 Stereo Camera Calibration

1. **Initialize Node:**
```bash
python3 stereo_camera_node_for_cadaver.py

```


2. **Verification:**
* Check `/camera1/usb_cam1/image_raw` and `/camera2/usb_cam2/image_raw` (640p).
* Ensure Left/Right sides are correctly mapped. If swapped, modify the video capture ID in the script.


3. **Run Calibration (CLion):**
* Open terminal: `source /opt/ros/melodic/setup.bash && clion`.
* Load project: `/home/trs-server/catkin_ws/src/stereo_calibration`.
* **Action:** Press `SPACE` to collect ~15 samples (Checkerboard must be visible in both views).
* **Result:** Press `a` to compute. Target error should be **< 0.1**. Update `yolo_final.py` (lines 102–115) with new parameters.



### 1.2 Hand-to-Eye Calibration

1. **Data Collection:** Run `hand_eye_calibration_sample.py`.
2. **Process:** Set robot speed to 0.1. Capture 20–30 images of a static checkerboard while moving the robot arm.
3. **MATLAB Processing:**
* Use Camera Calibrator App (Square: 5mm, 3 Radial coefficients).
* Export parameters and update `eTc` in `robots.ur5.py` and `self.Kc` in `Mainwindow_ROS_MultiTools.py`.



---

## 👁 2. Vision & Training

### 2.1 Data Collection

Record ~20 mins of video including different depths, tool orientations, and lighting:

```bash
conda activate yolov5
CUDA_VISIBLE_DEVICES=1 python3 yolo_final.py --weights path/to/best.pt --conf 0.25 --classes 2 4 6 10 14 --view-img --agnostic-nms

```

### 2.2 Network Training

1. **Labeling:** Convert LabelMe JSONs to YOLO format using `labelme2coco.py`.
2. **Training:**
```bash
python train.py --data cadaver_tools.yaml --cfg yolov5s.yaml --batch-size 64 --epochs 300

```



---

## 🚀 3. Execution (Cadaver Experiment)

### Step 1: Initial Settings

1. **ROS Core:** `roscore`
2. **Joystick:** `rosrun joy joy_node` (Ensure `js0` is active).
3. **UR5 Drivers:**
```bash
roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.3.33

```



### Step 2: Launch System

1. **YOLO Node:** Launch `yolo_final.py` with appropriate weights.
2. **Uterus Manipulator:** Run `run_udp.bat` on the robot side; verify with `python3 udp_test.py`.
3. **Main Controller:**
```bash
sudo su
python3 main.py

```



### Step 3: Operation Commands (GUI)

* **'1':** Switch Manual/Automatic mode.
* **'4':** Switch to System Version 2 (OSFA).
* **'6':** Initialize Workspace (Critical for depth/zoom scaling).
* **'2' then '3':** Set Global Initial Position.

---

## 🔍 Troubleshooting & Useful Commands

**Monitor Status:**

```bash
rostopic list
rostopic echo /joint_states
rostopic echo /blackbox_info

```

**Kill Hanging Processes:**

```bash
kill -9 $(ps -ef | grep main.py | grep -v grep | awk '{print $2}')

```

**Note:** If `joint_states` shows no data, restart the UR5 hardware control box and re-run the bringup launch.

---

**Next Step:** Would you like me to generate a specific `cadaver_tools.yaml` template for your YOLO training configuration?