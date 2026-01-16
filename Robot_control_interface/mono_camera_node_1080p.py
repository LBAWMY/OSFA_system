import rospy
import roslib
import sys
import cv2
import time
import numpy as np
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError

# def publish_image(imgdata):
#     image_temp=Image()
#     header = Header(stamp=rospy.Time.now())
#     header.frame_id = 'map'
#     image_temp.encoding='rgb8'
#     image_temp.data=np.array(imgdata).tostring()
#     #print(imgdata)
#     #image_temp.is_bigendian=True
#     image_temp.header=header
#     image_temp.step=1241*3
#     image_pubulish.publish(image_temp)

# ros communications
rospy.init_node('usb_cam_node', anonymous=True)
image_pub_L = rospy.Publisher("/camera1/usb_cam1/image_raw", Image)
image_pub_R = rospy.Publisher("/camera2/usb_cam2/image_raw", Image)
image_pub_LR = rospy.Publisher("/camera1_2/usb_cam1_2/image_raw", Image)

# frequency = 50 # 50hz
# dt = 1.0 / frequency
# loop_rate = rospy.Rate(frequency)

bridge = CvBridge()

# set video id
cap_L = cv2.VideoCapture(1)
cap_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1920
height = 1080
cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# cap_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('i', 'Y', 'U', 'V'))

# cap_R = cv2.VideoCapture(1)
# cap_L.set(cv2.CAP_PROP_MODE, -1) # cv2.CAP_MODE_YUYV
# width = 1920
# height = 1080
# cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

while(1):
    print('opencv version: ', cv2.__version__)
    # get a image
    t1 = time.time()
    # ret_R, image_R = cap_R.read()
    t1_2 = time.time()
    t2_1 = time.time()
    ret_L, image_L = cap_L.read()
    t2_2 = time.time()

    print('right img read. ', (t1_2 - t1), 'sec')
    print('left img read. ', (t2_2 - t2_1), 'sec')
    if image_L is not None:

        # Rimg_ROI = image_R[5:570, 24:700]
        # Limg_ROI = image_L[4:1080, 26:1894]

        # Rimg_ROI = image_R
        Limg_ROI = image_L

        # rz_R_Img=cv2.resize(Rimg_ROI, (640, 480))
        # rz_L_Img=cv2.resize(Limg_ROI, (640, 480))
        # rz_LR_Img=np.concatenate((rz_L_Img, rz_R_Img), axis=1)
        # rz_R_Img = Rimg_ROI
        rz_L_Img= Limg_ROI

        # print('Image Shape: ', rz_R_Img.shape)
        t2 = time.time()
        print('img manipulation.', (t2 - t1), 'sec')
        # cv2.imshow('Img_Right', rz_R_Img)
        cv2.imshow('Img_Left', rz_L_Img)
        # cv2.imshow('Img_LR', rz_LR_Img)
        try:
            ros_LImg = bridge.cv2_to_imgmsg(rz_L_Img, "bgr8")
            ros_LImg.header.stamp = rospy.Time.now()
            # ros_RImg = bridge.cv2_to_imgmsg(rz_R_Img, "bgr8")
            # ros_RImg.header.stamp = rospy.Time.now()
            # ros_LRImg = bridge.cv2_to_imgmsg(rz_LR_Img, "bgr8")
            # ros_LRImg.header.stamp = rospy.Time.now()
            # image_pub_L.publish(ros_LImg)
            # image_pub_R.publish(ros_RImg)
            # image_pub_LR.publish(ros_LRImg)
        except CvBridgeError as e:
            print(e)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if cv2.waitKey(1) & 0xFF == ord('s'):
        # cv2.imwrite('L-test.png', rz_L_Img)
        # cv2.imwrite('R-test.png', rz_R_Img)
        # cv2.imwrite('LR-test.png', rz_LR_Img)
    t3 = time.time()
    print('Done.', (t3 - t1), 'sec')

cap_L.release() 
# cap_R.release()
cv2.destroyAllWindows()
