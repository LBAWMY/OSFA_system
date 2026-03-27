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
rospy.init_node('usb_cam_node', anonymous=True) # , anonymous=True
image_pub_L = rospy.Publisher("/camera1/usb_cam1/image_raw", Image, queue_size=1)
image_pub_R = rospy.Publisher("/camera2/usb_cam2/image_raw", Image, queue_size=1)
image_pub_LR = rospy.Publisher("/camera1_2/usb_cam1_2/image_raw", Image, queue_size=1)
image_pub_LR_1080p = rospy.Publisher("/camera1_2/usb_cam1_2/image_raw_1080p", Image, queue_size=1)

frequency = 200 # 50hz
dt = 1.0 / frequency
loop_rate = rospy.Rate(frequency)

bridge = CvBridge()

# set video id
cap_R = cv2.VideoCapture(1)
cap_R.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_YUYV) # cv2.CAP_MODE_YUYV
# cap_R.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1920
height = 1080
cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
# cap_R.set(cv2.CAP_PROP_FPS, 50)

cap_L = cv2.VideoCapture(3)
cap_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
width = 1920
height = 1080
cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

image_L_pre = None
image_R_pre = None

while not rospy.is_shutdown():
    print('opencv version: ', cv2.__version__)
    # get a image
    t1 = time.time()
    ret_R, image_R = cap_R.read()
    t1_2 = time.time()
    t2_1 = time.time()
    ret_L, image_L = cap_L.read()
    t2_2 = time.time()

    # image_L_pre, image_R_pre
    # if image_L is None and image_L_pre is not None:
    #     image_L = image_L_pre.copy()
    # if image_R is None and image_R_pre is not None:
    #     image_R = image_R_pre.copy()

    print('right img read. ', (t1_2 - t1), 'sec')
    print('left img read. ', (t2_2 - t2_1), 'sec')
    print('left FrameRate: ', cap_L.get(cv2.CAP_PROP_FPS))
    print('right FrameRate: ', cap_R.get(cv2.CAP_PROP_FPS))
    if image_L is not None and image_R is not None:
        # image_L_pre = image_L
        # image_R_pre = image_R
        t2_3 = time.time()
        Rimg_ROI = image_R[0:1061, 47:1853]
        Limg_ROI = image_L[3:1080, 258:1661]

        # Rimg_ROI[:,:,0] = Rimg_ROI[:,:,0]/1.1
        # Rimg_ROI[:,:,1] = Rimg_ROI[:,:,1]/1.4
        # Rimg_ROI[:,:,2] = Rimg_ROI[:,:,2]/1.05
        # Rimg_ROI[:,:,0][Rimg_ROI[:,:,0]>15] -= 5
        # Rimg_ROI[:,:,1][Rimg_ROI[:,:,1]>10] -= 5
        # Rimg_ROI[:,:,2][Rimg_ROI[:,:,2]<240] += 15
        # Rimg_ROI[:,:,2][Rimg_ROI[:,:,2]<100] -= 15
        rz_R_Img_1080p = cv2.resize(Rimg_ROI, (1920, 1080))
        rz_L_Img_1080p = cv2.resize(Limg_ROI, (1920, 1080))

        # Here crop the region area for 3d visualization in image
        shift = 0
        rz_L_Img_1080p_shift = rz_L_Img_1080p[:, shift:, :]
        rz_R_Img_1080p_shift = rz_R_Img_1080p[:, :1920-shift, :]

        rz_L_Img_shift = cv2.resize(rz_L_Img_1080p_shift, (640, 480))
        rz_R_Img_shift = cv2.resize(rz_R_Img_1080p_shift, (640, 480))
        rz_LR_Img_shift = cv2.hconcat([rz_L_Img_shift, rz_R_Img_shift])

        rz_L_Img_1080p_shift = cv2.resize(rz_L_Img_1080p_shift, (1920, 1080))
        rz_R_Img_1080p_shift = cv2.resize(rz_R_Img_1080p_shift, (1920, 1080))

        # rz_LR_Img_1080p = cv2.hconcat([rz_L_Img_1080p, rz_R_Img_1080p])
        rz_LR_Img_1080p = cv2.hconcat([rz_L_Img_1080p_shift, rz_R_Img_1080p_shift])

        t2 = time.time()
        print('img manipulation.', (t2 - t2_3), 'sec')
        cv2.imshow('Img_Right_Shift', rz_R_Img_shift)
        cv2.imshow('Img_Left_Shift', rz_L_Img_shift)
        cv2.imshow('Img_LR', rz_LR_Img_shift)
        # try:
        ros_LImg = bridge.cv2_to_imgmsg(rz_L_Img_shift, "bgr8")
        ros_LImg.header.stamp = rospy.Time.now()
        ros_RImg = bridge.cv2_to_imgmsg(rz_R_Img_shift, "bgr8")
        ros_RImg.header.stamp = rospy.Time.now()
        ros_LRImg = bridge.cv2_to_imgmsg(rz_LR_Img_shift, "bgr8")
        ros_LRImg.header.stamp = rospy.Time.now()
        ros_LRImg_1080p = bridge.cv2_to_imgmsg(rz_LR_Img_1080p, "bgr8")
        ros_LRImg_1080p.header.stamp = rospy.Time.now()
        image_pub_L.publish(ros_LImg)
        image_pub_R.publish(ros_RImg)
        image_pub_LR.publish(ros_LRImg)
        image_pub_LR_1080p.publish(ros_LRImg_1080p)
        # except CvBridgeError as e:
        #     print(e)
        loop_rate.sleep()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('L-test.png', image_L)
        cv2.imwrite('R-test.png', image_R)
        # cv2.imwrite('LR-test.png', rz_LR_Img)
    t3 = time.time()
    print('Done.', (t3 - t1), 'sec')

cap_L.release()
cap_R.release()
cv2.destroyAllWindows()
