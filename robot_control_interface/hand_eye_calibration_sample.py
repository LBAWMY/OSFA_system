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

SAVE_FILE_DIR = './data/hand2eye_calibration_curi_v1/' # hand2eye_calibration

if not os.path.exists(SAVE_FILE_DIR):
    os.makedirs(SAVE_FILE_DIR)

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
    img_left_src = img_src[:, :int(H / 2), :]
    img_right_src = img_src[:, int(H / 2):, :]

rospy.init_node('usb_caliberation_node', anonymous=True)
rospy.Subscriber('/camera1_2/usb_cam1_2/image_raw', Image, imageCallback,
                     queue_size=1)
rospy.Subscriber('joint_states', JointState, jointStateCallback)

while not rospy.is_shutdown():
    if robot is None and joint_pos is not None:
        robot = UR5(init_joint_positions=joint_pos)

    if robot is not None:
        bTe = robot._robotics.MFK(joint_pos)
        print('q: ', joint_pos)
        print('bTe: ', bTe)
        if img_left_src is not None:
            cv2.imshow('image_left', img_left_src)
            # cv2.waitKey(1)

        key_press = cv2.waitKey(1)
        if key_press & 0xFF == ord('s'):
            sample_count = sample_count + 1
            cv2.imwrite(SAVE_FILE_DIR + "RImage" + str(sample_count).zfill(2) + ".jpg", img_right_src)
            cv2.imwrite(SAVE_FILE_DIR + "LImage" + str(sample_count).zfill(2) + ".jpg", img_left_src)

            fo = open(SAVE_FILE_DIR + "RobotPose" + str(sample_count).zfill(2) + ".txt", "w")
            fo.write(str(bTe[0, 0]) + " " + str(bTe[0, 1]) + " " + str(bTe[0, 2]) + " " + str(bTe[0, 3]) + "\n" +
                        str(bTe[1, 0]) + " " + str(bTe[1, 1]) + " " + str(bTe[1, 2]) + " " + str(bTe[1, 3]) + "\n" +
                        str(bTe[2, 0]) + " " + str(bTe[2, 1]) + " " + str(bTe[2, 2]) + " " + str(bTe[2, 3]) + "\n" +
                        str(bTe[3, 0]) + " " + str(bTe[3, 1]) + " " + str(bTe[3, 2]) + " " + str(bTe[3, 3]) + "\n")
            fo.close()

        if key_press & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
