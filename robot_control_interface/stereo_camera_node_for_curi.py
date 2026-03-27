import rospy
import roslib
import sys
import cv2
import time
import numpy as np
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError

# ros communications
rospy.init_node('usb_cam_node', anonymous=True) # , anonymous=True
image_pub_L = rospy.Publisher("/camera1/usb_cam1/image_raw", Image, queue_size=1)
image_pub_R = rospy.Publisher("/camera2/usb_cam2/image_raw", Image, queue_size=1)
image_pub_LR = rospy.Publisher("/camera1_2/usb_cam1_2/image_raw", Image, queue_size=1)

frequency = 200 # 50hz
dt = 1.0 / frequency
loop_rate = rospy.Rate(frequency)

bridge = CvBridge()

# set video id
cap_R = cv2.VideoCapture(3) # 1
cap_R.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# cap_R.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_YUYV) # cv2.CAP_MODE_YUYV
# cap_R.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1920
height = 1080
cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_R.set(cv2.CAP_PROP_FPS, 60)

cap_L = cv2.VideoCapture(1)
cap_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1920
height = 1080
cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_L.set(cv2.CAP_PROP_FPS, 60)

image_L_pre = None
image_R_pre = None

while not rospy.is_shutdown():
    print('opencv version: ', cv2.__version__)
    # get a image
    ret_R, image_R = cap_R.read()
    ret_L, image_L = cap_L.read()

    print('left FrameRate: ', cap_L.get(cv2.CAP_PROP_FPS))
    print('right FrameRate: ', cap_R.get(cv2.CAP_PROP_FPS))
    if image_L is not None and image_R is not None:
        # crop the black-area for curi kal-stroz systems
        Rimg_ROI = image_R[3:1061, 47:1853]
        Limg_ROI = image_L[4:1080, 258:1662]

        rz_R_Img_1080p = cv2.resize(Rimg_ROI, (1920, 1080))
        rz_L_Img_1080p = cv2.resize(Limg_ROI, (1920, 1080))

        rz_R_Img_640p = cv2.resize(Rimg_ROI, (640, 480))
        rz_L_Img_640p = cv2.resize(Limg_ROI, (640, 480))

        rz_LR_Img_1080p = cv2.hconcat([rz_L_Img_1080p, rz_R_Img_1080p])
        rz_LR_Img_640p = cv2.hconcat([rz_L_Img_640p, rz_R_Img_640p])

        cv2.imshow('Img_Right', rz_R_Img_640p)
        cv2.imshow('Img_Left', rz_L_Img_640p)
        cv2.imshow('Img_LR', rz_LR_Img_640p)
        # try:
        ros_LImg = bridge.cv2_to_imgmsg(rz_L_Img_640p, "bgr8") # rz_L_Img_1080p
        ros_LImg.header.stamp = rospy.Time.now()
        ros_RImg = bridge.cv2_to_imgmsg(rz_R_Img_640p, "bgr8") # rz_R_Img_1080p
        ros_RImg.header.stamp = rospy.Time.now()
        ros_LRImg = bridge.cv2_to_imgmsg(rz_LR_Img_640p, "bgr8")
        ros_LRImg.header.stamp = rospy.Time.now()
        ros_LRImg_1080p = bridge.cv2_to_imgmsg(rz_LR_Img_1080p, "bgr8")
        ros_LRImg_1080p.header.stamp = rospy.Time.now()
        image_pub_L.publish(ros_LImg)
        image_pub_R.publish(ros_RImg)
        image_pub_LR.publish(ros_LRImg)
        # image_pub_LR.publish(ros_LRImg_1080p)
        # except CvBridgeError as e:
        #     print(e)
        loop_rate.sleep()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('L-test.png', image_L)
        cv2.imwrite('R-test.png', image_R)
        # cv2.imwrite('LR-test.png', rz_LR_Img_1080p)
    t3 = time.time()

cap_L.release()
cap_R.release()
cv2.destroyAllWindows()
