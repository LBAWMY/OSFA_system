#!/usr/bin/env python
import rospy
import roslib
import sys
import cv2
from robots.ur5 import UR5
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from std_msgs.msg import *
from sensor_msgs.msg import *
import os

SAVE_FILE_DIR = './data/videos'
if not os.path.exists(SAVE_FILE_DIR):
    os.makedirs(SAVE_FILE_DIR)

SAVE_FILE_NAME = os.path.join(SAVE_FILE_DIR, 'test005.mp4')
print('saving to ', SAVE_FILE_NAME)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(SAVE_FILE_NAME, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (1920, 1080), True)

bridge = CvBridge()
joint_pos = None
img_right_src = None
img_left_src = None
robot = None
sample_count = 0

def jointStateCallback(jointState):
    global joint_pos
    joint_pos = np.array(jointState.position, dtype=float)

def imageCallback(img):
    global img_right_src, img_left_src
    img_src = bridge.imgmsg_to_cv2(img, "bgr8")
    W, H = img_src.shape[:2]
    img_right_src = img_src[:, int(H / 2):, :]
    img_left_src = img_src[:, :int(H / 2), :]

rospy.init_node('usb_caliberation_node', anonymous=True)
rospy.Subscriber('/camera1_2/usb_cam1_2/image_raw_1080p', Image, imageCallback,
                     queue_size=1)

while not rospy.is_shutdown():
    if img_left_src is not None:
        video.write(img_left_src)
        cv2.imshow('image_left', img_left_src)
        # cv2.imshow('image_right', img_right_src)
        cv2.waitKey(20)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
