how to use:
- python3 stero_camera_node.py
- cd /detection/depthtracker, python yolo_track_multi_pure.py --weights ../yolov5/runs/train/exp11/weights/best.pt --conf 0.25 --classes 1 3 --view-img
- python3 main.py

click "start" on gui
click "end" to close the gui

udp:
"Mainwindow_ROS.py", line 104-106, line 248-250
send instruction to robot: e.g. "0#1#0#0#0#0#"
