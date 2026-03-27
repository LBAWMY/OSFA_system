import rospy
import sys
import roslib
# roslib.load_manifest('ur5_endoscope_arm')
import cv2
import numpy as np
import copy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image, Joy, JointState
# from ur5_endoscope_arm.msg import *
# from robots.ur5_vision_msg import ur5_vision_msg

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap

from functools import wraps
from Ui_MainWindow import Ui_MainWindow

# from robots.ur5 import UR5
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64

import socket


def debug_class(calss_name='MainWindow'):
    # print('in debug_class')
    def debug(f):
        # print('in debug')
        @wraps(f)
        def print_debug(*args, **kwargs):
            # print('in print_debug')
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print(calss_name + '.' + f.__name__ + '() Error！')
                print('Error：', e)
        return print_debug
    return debug


class MainWindow_ROS(QMainWindow, Ui_MainWindow):
    @debug_class('MainWindow')
    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow_ROS, self).__init__(parent)
        self.setupUi(self)
        # Initialization: UI
        # self.label_img_show.setScaledContents(True)  # Show image with adaptive scale
        # self.img_none = np.ones((480, 640, 3), dtype=np.uint8)*255
        self.img_none = np.ones((810, 1440, 3), dtype=np.uint8)*255
        self.img_right_src = None
        self.show_img(self.img_none)
        # Initialization: Network
        # ToDO
        # Other settings
        # self.camera_left_index = 1
        self.video_flag = False
        # self.vision_params = ur5_vision_msg()
        # self.vision_params = {}
        # self.Kc = np.array([[719.778534468158, 0.0, 379.165423468819],
        #                     [0.0, 787.884348579813, 273.066509594866],
        #                     [0.0, 0.0, 1.0]])

        # Initialization: ROS
        rospy.init_node("Mainwindow_node")
        # capture image
        self.bridge = CvBridge()
        # rospy.Subscriber("/camera2/usb_cam2/image_raw", Image, self.imageCallback)
        rospy.Subscriber("/camera/usb_cam/image_raw", Image, self.imageCallback)
        # capture joystick
        # self.joy_src = None
        # rospy.Subscriber("joy", Joy, self.joyCallback)
        # capture the ur5 joint state
        # self.joint_state = None
        # rospy.Subscriber("joint_states", JointState, self.jointStateCallback)
        # publish the vision params
        # self.vision_params_pub = rospy.Publisher("vision_params", ur5_vision_msg, queue_size=10)
        # publish command to control ur5
        # self.ur5_crtl_script_pub = rospy.Publisher("ur_driver/URScript", String, queue_size=10)
        self.loop_rate = rospy.Rate(50) # 50hz
        # subscribe the predicted action
        # rospy.Subscriber("/pred_action", Vector3, self.predActionCallback)
        # subscribe the estimated depth
        # rospy.Subscriber("/tool_depth", Float64, self.toolDepthCallback)
        # subscribe the estimated position
        rospy.Subscriber("/pred_position_aux", Vector3, self.predPositionAuxCallback)
        # subscribe the estimated position
        rospy.Subscriber("/pred_position_main", Vector3, self.predPositionMainCallback)

        # robot
        # self.robot = None

        # load gui sources
        self.load_icons()

        # udp connection
        self.addr = ('198.168.0.1', 1234)
        self.UDPSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def imageCallback(self, img):
        img_src = self.bridge.imgmsg_to_cv2(img, "bgr8")
        W, H = img_src.shape[:2]
        # self.img_left_src = cv2.resize(img_src[:, :int(H/2), :], (1440, 810))
        self.img_right_src = cv2.resize(img_src, (1440, 810))

    # def joyCallback(self, Joy):
    #     self.joy_src = Joy
    #
    # def jointStateCallback(self, jointState):
    #     self.joint_state = jointState
    #     self.joint_pos = np.array(self.joint_state.position,dtype=float)
    
    # def predActionCallback(self, vector3):
    #     self.pred_action = vector3

    def predPositionAuxCallback(self, vector3):
        self.pred_position_aux = vector3

    def predPositionMainCallback(self, vector3):
        self.pred_position_main = vector3

    # def toolDepthCallback(self, depth):
    #     self.tool_depth = depth.data

    @debug_class('MainWindow')
    def show_img(self, img):
        """
        show the numpy format image with the QLabel datatype
        :param img:
        :return:
        """
        show_img = QImage(img.data, img.shape[1], img.shape[0],
                          QImage.Format_RGB888)
        # cv2.imshow('Img', img)
        # if cv2.waitKey(20):
        # print(img.shape)
        self.label_img_show.setPixmap(QPixmap.fromImage(show_img))

    @debug_class('MainWindow')
    @pyqtSlot()
    def pushButton_start_clicked(self):
        """
        open camera if the button was clicked
        :return:
        """
        if self.img_right_src is not None:
            # Disable this button
            self.pushButton_start.setEnabled(False)

            # acquire the image from the camera
            self.video_flag = True

            # speed_level setting
            # self.image_xy_speed_coef = 1.0
            xy_flag_add = True
            xy_flag_dec = True
            # self.depth_zz_speed_coef = 1.0
            z_flag_add = True
            z_flag_dec = True

            # Manual/Automatic Mode
            '''Manual_Mode = True
            Mode_flag = True
            # Predefined action
            self.Return_Mode = False
            Return_flag = True # used for detect the button only once, when pressed and released
            self.Zoomin_Mode = False
            Zoomin_flag = True # used for detect the button only once, when pressed and released
            self.zoomin_action_once_flag = True # used for remember the previous position
            self.Zoomout_Mode = False
            Zoomout_flag = True # used for detect the button only once, when pressed and released
            self.zoomout_action_once_flag = True # used for remember the previous position
            self.Track_xy_Mode = False
            Track_xy_flag = True # used for detect the button only once, when pressed and released
            self.Track_z_Mode = False
            Track_z_flag = True
            self.Emergency_Mode = False
            Emergency_flag = True
            self.Return_Track_Mode = False
            Return_Track_flag = True
            # status for the predefined action: False means the target not finished; True means that the target finished
            self.target_status = True
            # flag for control the depth using predefined interval or no depth control at this moment
            self.user_depth_flag = False'''

            # ------------- uterus manipulator setting ---------------------
            self.UterusUp_Mode = False
            self.UterusDown_Mode = False
            self.UterusLeft_Mode = False
            self.UterusRight_Mode = False
            self.UterusInsert_Mode = False
            self.UterusRetract_Mode = False

            # left-right view flag
            self.gui_side_left = True

            while self.video_flag:
                self.img_right_src_RGB = cv2.cvtColor(self.img_right_src, cv2.COLOR_BGR2RGB)

                actions_img = [0, 0, 0]
                self.action_uterus = [0, 0, 0, 0, 0, 0]   # up, down, left, right, insert, retract
                if self.UterusUp_Mode is True:   # y  # todo emergency mode of the uterus manipulator
                    # self.uterus_up()
                    self.action_uterus = [1, 0, 0, 0, 0, 0]
                    actions_img = [0, 1, 0] # may need to multiply -1 in actions[0 or 1]
                    print('UterusUp mode')
                    self.UterusUp_Mode = False
                elif self.UterusDown_Mode is True:
                    # self.uterus_down()
                    self.action_uterus = [0, 1, 0, 0, 0, 0]
                    actions_img = [0, -1, 0] # may need to multiply -1 in actions[0 or 1]
                    print('UterusDown mode')
                    self.UterusDown_Mode = False
                elif self.UterusLeft_Mode is True:    # x
                    # self.uterus_left()
                    self.action_uterus = [0, 0, 1, 0, 0, 0]
                    actions_img = [1, 0, 0] # may need to multiply -1 in actions[0 or 1]
                    print('UterusLeft mode')
                    self.UterusLeft_Mode = False
                elif self.UterusRight_Mode is True:
                    # self.uterus_right()
                    self.action_uterus = [0, 0, 0, 1, 0, 0]
                    actions_img = [-1, 0, 0] # may need to multiply -1 in actions[0 or 1]
                    print('UterusRight mode')
                    self.UterusRight_Mode = False
                elif self.UterusInsert_Mode is True:     # z
                    # self.uterus_insert()
                    self.action_uterus = [0, 0, 0, 0, 1, 0]
                    actions_img = [0, 0, 1] # may need to multiply -1 in actions[0 or 1]
                    print('UterusInsert mode')
                    self.UterusInsert_Mode = False
                elif self.UterusRetract_Mode is True:
                    # self.uterus_retract()
                    self.action_uterus = [0, 0, 0, 0, 0, 1]
                    actions_img = [0, 0, -1] # may need to multiply -1 in actions[0 or 1]
                    print('UterusRetract mode')
                    self.UterusRetract_Mode = False
                print(self.action_uterus)

                # action_uterus: up down left right insert retract  (image)
                action_uterus_data = str(self.action_uterus[0]) + "#" + str(self.action_uterus[1]) + "#" + str(self.action_uterus[2]) + "#" + str(self.action_uterus[3]) + "#" + str(self.action_uterus[4]) + "#" + str(self.action_uterus[5]) + "#"
                self.UDPSock.sendto(action_uterus_data.encode('utf-8'), self.addr)

                action_uterus_print = f'uterus action: {self.action_uterus}'
                cv2.putText(self.img_right_src_RGB, action_uterus_print, (150, 80), 0, 0.5, [225, 255, 255], thickness=2,
                                    lineType=cv2.LINE_AA)
                # uterus_panel_count_print = f'uterus count: {self.uterus_panel_count[0,:]}'
                # cv2.putText(self.img_right_src_RGB, uterus_panel_count_print, (150, 120), 0, 0.5, [225, 255, 255], thickness=2,
                                    # lineType=cv2.LINE_AA)

                '''xy_speed = f'V_xy speed: {self.image_xy_speed_coef:.2f}'
                z_speed = f'V_z speed: {self.depth_zz_speed_coef:.2f}'
                if self.Emergency_Mode is True:
                    Mode = f'Mode: Emergency'
                elif self.Return_Mode is True:
                    Mode = f'Mode: Return'
                elif self.Zoomin_Mode is True:
                    Mode= f'Mode: Zoom_in'
                elif self.Zoomout_Mode is True:
                    Mode = f'Mode: Zoom_out'
                elif self.Track_xy_Mode is True:
                    Mode = f'Mode: Track_XY'
                elif self.Track_z_Mode is True:
                    Mode = f'Mode: Track_Z'
                elif self.Return_Track_Mode is True:
                    Mode = f'Mode: ReturnTrack'
                else:
                    Mode = f'Mode: Manual' if Manual_Mode is True else f'Mode: Automatic'''

                if self.mode == 'Camera':
                    gui_mode = f'Mode: Camera'
                elif self.mode =='Uterus':
                    gui_mode = f'Mode: Uterus'
                else:
                    gui_mode = f'Mode:'

                # cv2.putText(self.img_right_src_RGB, Mode, (150, 20), 0, 0.5, [225, 255, 255], thickness=2,
                #                     lineType=cv2.LINE_AA)
                # cv2.putText(self.img_right_src_RGB, xy_speed, (320, 20), 0, 0.5, [225, 255, 255], thickness=1,
                #                     lineType=cv2.LINE_AA)
                # cv2.putText(self.img_right_src_RGB, z_speed, (480, 20), 0, 0.5, [225, 255, 255], thickness=1,
                #                     lineType=cv2.LINE_AA)
                cv2.putText(self.img_right_src_RGB, gui_mode, (150, 60), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)


                # self.img_left_src_RGB = self.draw_img(self.img_left_src_RGB, actions_img)
                if self.mode == 'Uterus':
                    self.img_right_src_RGB = self.draw_img_seperate(self.img_right_src_RGB, actions_img, self.mode)

                # draw the position
                scale_x = 1440 / 640
                scale_y = 810 / 480
                if (not np.isnan(self.pred_position_aux.x)) and (not np.isnan(self.pred_position_aux.y)):
                    positions_aux_img = [int(self.pred_position_aux.x * scale_x), int(self.pred_position_aux.y * scale_y)]
                else:
                    positions_aux_img = [-1, -1]

                if (not np.isnan(self.pred_position_main.x)) and (not np.isnan(self.pred_position_main.y)):
                    positions_main_img = [int(self.pred_position_main.x * scale_x), int(self.pred_position_main.y * scale_y)]
                else:
                    positions_main_img = [-1, -1]

                positions_aux_img_x = f'x pos: {positions_aux_img[0]:d}'
                positions_aux_img_y = f'y pos: {positions_aux_img[1]:d}'
                cv2.putText(self.img_right_src_RGB, positions_aux_img_x, (280, 60), 0, 0.5, [225, 255, 255], thickness=1,
                    lineType=cv2.LINE_AA)
                cv2.putText(self.img_right_src_RGB, positions_aux_img_y, (480, 60), 0, 0.5, [225, 255, 255], thickness=1,
                    lineType=cv2.LINE_AA)

                if self.gui_side_left is True:
                    self.img_right_src_RGB = self.draw_img_GUI_left_side(self.img_right_src_RGB, position=positions_aux_img, position_main=positions_main_img)
                else:
                    self.img_right_src_RGB = self.draw_img_GUI_right_side(self.img_right_src_RGB, position=positions_aux_img, position_main=positions_main_img)

                if positions_main_img[0] >= 0 and positions_main_img[1] >= 0:
                    self.img_right_src_RGB = cv2.circle(self.img_right_src_RGB, (positions_main_img[0], positions_main_img[1]), 4, (251, 33,  194), 8)

                self.show_img(self.img_right_src_RGB)

                self.loop_rate.sleep()
                #
                # UI update
                QApplication.processEvents()
        else:
            # self.textEdit.setText('No camera connected!')
            msg = QMessageBox.warning(self, 'warning', 'No camera connected!', buttons=QMessageBox.Ok)
            print('No camera connected!')

    # @debug_class('MainWindow')
    # @pyqtSlot()
    def pushButton_Test_clicked(self):
        pass
        """
        Test for control the laparoscope by joystick
        :return:
        """
        # calculate vision parameters(cVc/Errors) and publish the control signals
        # self.actions = [-self.joy_src.axes[0], -self.joy_src.axes[1], self.joy_src.axes[4]]
        '''self.actions = [-self.joy_src.axes[3] * self.image_xy_speed_coef, -self.joy_src.axes[4] * self.image_xy_speed_coef, self.joy_src.axes[1] * self.depth_zz_speed_coef]
        self.vision_params = self.cal_vision_params(self.Kc, self.actions, depth=0.1, flag=0)
        # self.vision_params_pub.publish(self.vision_params)
        cVc = self.vision_params.Parameter
        homo_delta = self.vision_params.ImageError[0:2]

        # drive the ur5 robot
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)
        print("step1: cVc: ", cVc)
        print("homo_delta: ", homo_delta)
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        # print('delta_q: ', delta_q)
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")'''


    def uterus_up(self):
        self.UterusUp_Mode = False

    def index2uterus_motion(self, index, img):
        if index == 0:
            img = merge_bbox_img(roi_bbox=self.uterus_panel_up_bbox, template=self.uterus_panel_up_click_icon, img=img, alpha=0.3, beta=0.7)
            self.UterusUp_Mode = True
        elif index == 1:
            img = merge_bbox_img(roi_bbox=self.uterus_panel_down_bbox, template=self.uterus_panel_down_click_icon, img=img, alpha=0.3, beta=0.7)
            self.UterusDown_Mode = True
        elif index == 2:
            img = merge_bbox_img(roi_bbox=self.uterus_panel_left_bbox, template=self.uterus_panel_left_click_icon, img=img, alpha=0.3, beta=0.7)
            self.UterusLeft_Mode = True
        elif index == 3:
            img = merge_bbox_img(roi_bbox=self.uterus_panel_right_bbox, template=self.uterus_panel_right_click_icon, img=img, alpha=0.3, beta=0.7)
            self.UterusRight_Mode = True
        elif index == 4:
            img = merge_bbox_img(roi_bbox=self.uterus_panel_insert_bbox, template=self.uterus_panel_insert_click_icon, img=img, alpha=0.3, beta=0.7)
            self.UterusInsert_Mode = True
        elif index == 5:
            img = merge_bbox_img(roi_bbox=self.uterus_panel_retract_bbox, template=self.uterus_panel_retract_click_icon, img=img, alpha=0.3, beta=0.7)
            self.UterusRetract_Mode = True
        return img

    @debug_class('MainWindow')
    @pyqtSlot()
    def pushButton_end_clicked(self):
        """
        terminate the process
        :return:
        """
        self.video_flag = False
        # Here we show the blank image instead
        self.show_img(self.img_none)
        # Clear the text
        # TODO
        self.pushButton_start.setEnabled(True)

    @debug_class('MainWindow')
    def draw_img_seperate(self, img, actions, mode):
        """
        draw the action results in the image
        :param img:
        :param actions:
        :return:
        """
        h, w, c = img.shape
        # print('w, h: ', w, h)
        rect_h = 10
        rect_w = 40
        #
        refer_h = 7/8 * h
        refer_w = 5/6 * w

        if mode == 'Uterus':
            refer_h = 1 / 2 * h
            refer_w = 1 / 2 * w

        # reference
        pt_left = (int(refer_w - 1 * rect_w), int(refer_h))
        pt_right = (int(refer_w + 1 * rect_w), int(refer_h))
        pt_up = (int(refer_w), int(refer_h  - 1 * rect_w))
        pt_down = (int(refer_w), int(refer_h  + 1 * rect_w))
        pt_in = (int(refer_w + 1.5 * rect_w), int(refer_h  - 1 * rect_w))
        pt_out = (int(refer_w + 1.5 * rect_w), int(refer_h  + 1 * rect_w))

        # left/right action
        pt1 = (int(refer_w), int(refer_h))
        pt2 = (int(refer_w - actions[0] * rect_w), int(refer_h))

        # up/down action
        pt3 = (int(refer_w), int(refer_h))
        pt4 = (int(refer_w), int(refer_h  - actions[1] * rect_w))

        # zoom in/out action
        pt5 = (int(refer_w + 1.5 * rect_w), int(refer_h))
        pt6 = (int(refer_w + 1.5 * rect_w), int(refer_h  - actions[2] * rect_w))

        # draw the reference
        gray_value = 200
        img = cv2.rectangle(img, pt_left, pt_right, (gray_value, gray_value, gray_value), thickness=4)
        img = cv2.rectangle(img, pt_up, pt_down, (gray_value, gray_value, gray_value), thickness=4)
        img = cv2.rectangle(img, pt_in, pt_out, (gray_value, gray_value, gray_value), thickness=4)

        # draw the rect in image
        img = cv2.rectangle(img, pt1, pt2, (0, 255, 0), thickness=4)
        img = cv2.rectangle(img, pt3, pt4, (255, 0, 0), thickness=4)
        img = cv2.rectangle(img, pt5, pt6, (0, 0, 255), thickness=4)
        return img

    @debug_class('MainWindow')
    def draw_img_GUI_left_side(self, ori_img, position, position_main):
        resize_W = 1440
        resize_H = 810
        part_W = round(resize_W/16)
        part_H = round(resize_H/8)
        # draw the tool center in image
        img = ori_img.copy()
        count_threshold = 50
        # draw the GUI interface for manipulation
        if ((position[0] < part_W * 2 and position[0] > 0 and position[1] < part_H * 2 and position[1] > 0)
            or (position_main[0] < part_W * 2 and position_main[0] > 0 and position_main[1] < part_H * 2 and position_main[1] > 0)) \
                and self.mode == 'None':
            # self.mode = 'Main' # need once flag to set this variable
            print('uterus shape: ', self.uterus_panel_up_icon.shape)
            print('uterus bbox: ', self.main_dominant_bbox_left_small)
            # self.main_panel_small_count[0,0] = self.main_panel_small_count[0, 0] + 1
            # if is_pos_in_bbox(position, self.main_uterus_bbox_left_small):
            if is_pos_in_bbox(position, self.main_uterus_bbox_left_small) or is_pos_in_bbox(position_main, self.main_uterus_bbox_left_small):
                # img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.main_uterus_icon_large, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 0] = self.main_panel_count[0, 0] + 1
                self.main_panel_count[0, 1] = self.main_panel_count[0, 2] = self.main_panel_count[0, 3] = 0
                if self.main_panel_count[0, 0] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.uterus_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    self.mode = 'Uterus'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.main_uterus_icon_large, img=img, alpha=0.3, beta=0.7)
            else:
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count = np.zeros((1,4))
                # self.main_panel_small_count = np.zeros((1,4))

            '''elif is_pos_in_bbox(position, self.main_dominant_bbox_left_small):
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_large, template=self.main_domain_icon_large, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 1] = self.main_panel_count[0, 1] + 1
                self.main_panel_count[0, 0] = self.main_panel_count[0, 2] = self.main_panel_count[0, 3] = 0
                if self.main_panel_count[0, 1] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_large, template=self.main_domain_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    # self.mode = 'Dominant'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_large, template=self.main_domain_icon_large, img=img, alpha=0.3, beta=0.7)
            elif is_pos_in_bbox(position, self.main_switch_bbox_left_small):
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_large, template=self.main_switch_left_icon_large, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 2] = self.main_panel_count[0, 2] + 1
                self.main_panel_count[0, 0] = self.main_panel_count[0, 1] = self.main_panel_count[0, 3] = 0
                if self.main_panel_count[0, 2] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_large, template=self.main_switch_left_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    self.gui_side_left = not self.gui_side_left
                    # self.mode = 'Switch'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_large, template=self.main_switch_left_icon_large, img=img, alpha=0.3, beta=0.7)
            elif is_pos_in_bbox(position, self.main_camera_bbox_left_small):
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_large, template=self.main_camera_icon_large, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 3] = self.main_panel_count[0, 3] + 1
                self.main_panel_count[0, 0] = self.main_panel_count[0, 1] = self.main_panel_count[0, 2] = 0
                if self.main_panel_count[0, 3] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_large, template=self.camera_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    self.main_panel_count[0, 3] = 0
                    self.mode = 'Camera'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_large, template=self.main_camera_icon_large, img=img, alpha=0.3, beta=0.7)'''
        else:
            self.main_panel_count = np.zeros((1,4))
            # self.main_panel_small_count = np.zeros((1,4))

        # Uterus panel
        if self.mode == 'Uterus':
            img = ori_img.copy()
            img = merge_bbox_img(roi_bbox=self.uterus_panel_up_bbox, template=self.uterus_panel_up_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_down_bbox, template=self.uterus_panel_down_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_left_bbox, template=self.uterus_panel_left_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_right_bbox, template=self.uterus_panel_right_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_retract_bbox, template=self.uterus_panel_retract_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_insert_bbox, template=self.uterus_panel_insert_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.uterus_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
            uterus_panel_W = round(resize_W/16)
            uterus_panel_H = round(resize_H/8)

            if is_pos_in_bbox(position, self.uterus_panel_up_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_up_bbox):
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 0] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 0] = 0
            if is_pos_in_bbox(position, self.uterus_panel_down_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_down_bbox):
                self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 1] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 1] = 0
            if is_pos_in_bbox(position, self.uterus_panel_left_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_left_bbox):
                self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 2] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 2] = 0
            if is_pos_in_bbox(position, self.uterus_panel_right_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_right_bbox):
                self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 3] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 3] = 0
            if is_pos_in_bbox(position, self.uterus_panel_insert_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_insert_bbox):
                self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 4] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 4] = 0
            if is_pos_in_bbox(position, self.uterus_panel_retract_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_retract_bbox):
                self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 5] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 5] = 0
            if (self.uterus_panel_count[0,:6] == self.uterus_panel_count_old[0,:6]).all():
                self.uterus_panel_count[0,:6] = 0
                self.uterus_panel_count[0, 6] = self.uterus_panel_count[0, 6] + 1
                # self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = 0

            uterus_panel_count_sort = np.sort(self.uterus_panel_count[0,:-1])
            if self.uterus_panel_count[0, 6] > count_threshold*5:
                img = img.copy()
                self.uterus_panel_count[0, :] = 0
                self.mode = 'None'
            elif uterus_panel_count_sort[-1] < count_threshold:
                # max_index = np.where(self.uterus_panel_count==uterus_panel_count_sort[0][-1])
                pass
            elif uterus_panel_count_sort[-1] > count_threshold and uterus_panel_count_sort[-2] < count_threshold:
                select_index = np.where(self.uterus_panel_count==uterus_panel_count_sort[-1])
                img = self.index2uterus_motion(select_index[1], img)
                # self.uterus_panel_count[0,:6] = 0
                # self.uterus_panel_count[0,select_index[1]] = uterus_panel_count_sort[-1]
            elif uterus_panel_count_sort[-1] > count_threshold and uterus_panel_count_sort[-2] > count_threshold:
                # discard_index = np.where(self.uterus_panel_count==uterus_panel_count_sort[-1])
                # self.uterus_panel_count[0, discard_index] = 0
                select_index = np.where(self.uterus_panel_count==uterus_panel_count_sort[-2])
                img = self.index2uterus_motion(select_index[1], img)
                # self.uterus_panel_count[0,:6] = 0
                # self.uterus_panel_count[0,select_index[1]] = uterus_panel_count_sort[-2]
            self.uterus_panel_count_old = copy.deepcopy(self.uterus_panel_count)

        # Camera panel
        if self.mode == 'Camera':
            self.mode = 'None'
            '''img = ori_img.copy()
            # TODO: Draw the position again, sloved
            # TODO: xiaochu chanyu tuan
            img = merge_bbox_img(roi_bbox=self.camera_panel_zoomin_bbox_left, template=self.camera_panel_zoomin_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.camera_panel_zoomout_bbox_left, template=self.camera_panel_zoomout_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.camera_panel_zoomauto_bbox_left, template=self.camera_panel_zoomauto_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.camera_panel_setzero_bbox_left, template=self.camera_panel_setzero_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_large, template=self.camera_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
            camera_panel_H = round(3*resize_H/4/4)
            if is_pos_in_bbox(position, self.camera_panel_zoomin_bbox_left):
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 0] + 1
                self.camera_panel_count[0, 1] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 4] = self.camera_panel_count[0, 5] = 0
            elif is_pos_in_bbox(position, self.camera_panel_zoomout_bbox_left):
                self.camera_panel_count[0, 1] = self.camera_panel_count[0, 1] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 4] = self.camera_panel_count[0, 5] = 0
            elif is_pos_in_bbox(position, self.camera_panel_zoomauto_bbox_left):
                self.camera_panel_count[0, 2] = self.camera_panel_count[0, 2] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 1] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 4] = self.camera_panel_count[0, 5] = 0
            elif is_pos_in_bbox(position, self.camera_panel_setzero_bbox_left):
                self.camera_panel_count[0, 3] = self.camera_panel_count[0, 3] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 1] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 4] = self.camera_panel_count[0, 5] = 0
            elif is_pos_in_bbox(position, self.main_camera_bbox_left_large):
                self.camera_panel_count[0, 4] = self.camera_panel_count[0, 4] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 1] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 5] = 0
            else:
                self.camera_panel_count[0, 5] = self.camera_panel_count[0, 5] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 1] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 4] = 0

            if self.camera_panel_count[0, 0] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.camera_panel_zoomin_bbox_left, template=self.camera_panel_zoomin_click_icon, img=img, alpha=0.3, beta=0.7)
                self.Return_Track_Mode = False # only adjust the depth
                self.Zoomin_Mode = True
                self.zoomin_action_once_flag = True
            elif self.camera_panel_count[0, 1] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.camera_panel_zoomout_bbox_left, template=self.camera_panel_zoomout_click_icon, img=img, alpha=0.3, beta=0.7)
                self.Return_Track_Mode = False # only adjust the depth
                self.Zoomout_Mode = True
                self.zoomout_action_once_flag = True
            elif self.camera_panel_count[0, 2] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.camera_panel_zoomauto_bbox_left, template=self.camera_panel_zoomauto_click_icon, img=img, alpha=0.3, beta=0.7)
                self.Return_Track_Mode = False # only adjust the depth
                self.Track_z_Mode = True
            elif self.camera_panel_count[0, 3] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.camera_panel_setzero_bbox_left, template=self.camera_panel_setzero_click_icon, img=img, alpha=0.3, beta=0.7)
                # self.Return_Mode = True
                self.target_status = False # necessary: to be used to back to initial position
                self.Return_Track_Mode = True
            elif self.camera_panel_count[0, 5] > count_threshold*3:
                img = img.copy()
                # self.camera_panel_count[0, 5] = 0
                self.mode = 'None'''
        if position[0] >= 0 and position[1] >= 0:
            img = cv2.circle(img, (position[0], position[1]), 4, (123, 238, 253), 8)

        return img

    @debug_class('MainWindow')
    def draw_img_GUI_right_side(self, ori_img, position, position_main):
        resize_W = 1440
        resize_H = 810
        part_W = round(resize_W/16)
        part_H = round(resize_H/8)
        # draw the tool center in image
        img = ori_img.copy()
        count_threshold = 50

        # draw the GUI interface for manipulation
        if ((position[0] > resize_W - part_W * 2 and position[0] > 0 and position[1] < part_H * 2 and position[1] > 0)
            or (position_main[0] > resize_W - part_W * 2 and position_main[0] > 0 and position_main[1] < part_H * 2 and position_main[1] > 0)) \
                and self.mode == 'None':
            # self.mode = 'Main' # need once flag to set this variable
            print('uterus shape: ', self.uterus_panel_up_icon.shape)
            print('uterus bbox: ', self.main_dominant_bbox_right_small)
            # self.main_panel_small_count[0,0] = self.main_panel_small_count[0, 0] + 1
            if is_pos_in_bbox(position, self.main_uterus_bbox_right_small) or is_pos_in_bbox(position_main, self.main_uterus_bbox_right_small):
                # img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.main_uterus_icon_large, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_right_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_switch_bbox_right_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_camera_bbox_right_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 0] = self.main_panel_count[0, 0] + 1
                self.main_panel_count[0, 1] = self.main_panel_count[0, 2] = self.main_panel_count[0, 3] = 0
                if self.main_panel_count[0, 0] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_large, template=self.uterus_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    self.mode = 'Uterus'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_large, template=self.main_uterus_icon_large, img=img, alpha=0.3, beta=0.7)

            else:
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_right_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_switch_bbox_right_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_camera_bbox_right_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count = np.zeros((1,4))
                # self.main_panel_small_count = np.zeros((1,4))
        else:
            self.main_panel_count = np.zeros((1,4))
            # self.main_panel_small_count = np.zeros((1,4))


        # if self.main_panel_count[0, 1] > count_threshold:
        #     img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_large, template=self.main_domain_click_icon_large, img=img, alpha=0.3, beta=0.7)
        # if self.main_panel_count[0, 2] > count_threshold:
        #     img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_large, template=self.main_switch_left_click_icon_large, img=img, alpha=0.3, beta=0.7)
        # if self.main_panel_count[0, 3] > count_threshold:
        #     img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_large, template=self.camera_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
        #     self.mode = 'Camera'

        # Uterus panel
        if self.mode == 'Uterus':
            img = ori_img.copy()
            img = merge_bbox_img(roi_bbox=self.uterus_panel_up_bbox, template=self.uterus_panel_up_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_down_bbox, template=self.uterus_panel_down_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_left_bbox, template=self.uterus_panel_left_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_right_bbox, template=self.uterus_panel_right_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_retract_bbox, template=self.uterus_panel_retract_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.uterus_panel_insert_bbox, template=self.uterus_panel_insert_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_large, template=self.uterus_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
            uterus_panel_W = round(resize_W/16)
            uterus_panel_H = round(resize_H/8)
            if is_pos_in_bbox(position, self.uterus_panel_up_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_up_bbox):
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 0] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 0] = 0
            if is_pos_in_bbox(position, self.uterus_panel_down_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_down_bbox):
                self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 1] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 1] = 0
            if is_pos_in_bbox(position, self.uterus_panel_left_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_left_bbox):
                self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 2] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 2] = 0
            if is_pos_in_bbox(position, self.uterus_panel_right_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_right_bbox):
                self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 3] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 3] = 0
            if is_pos_in_bbox(position, self.uterus_panel_insert_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_insert_bbox):
                self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 4] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 4] = 0
            if is_pos_in_bbox(position, self.uterus_panel_retract_bbox) or is_pos_in_bbox(position_main, self.uterus_panel_retract_bbox):
                self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 5] + 1
                self.uterus_panel_count[0, 6] = 0
            else: self.uterus_panel_count[0, 5] = 0
            if (self.uterus_panel_count[0,:6] == self.uterus_panel_count_old[0,:6]).all():
                self.uterus_panel_count[0,:6] = 0
                self.uterus_panel_count[0, 6] = self.uterus_panel_count[0, 6] + 1
                # self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = 0

            uterus_panel_count_sort = np.sort(self.uterus_panel_count[0,:-1])
            if self.uterus_panel_count[0, 6] > count_threshold*5:
                img = img.copy()
                self.uterus_panel_count[0, :] = 0
                self.mode = 'None'
            elif uterus_panel_count_sort[-1] < count_threshold:
                # max_index = np.where(self.uterus_panel_count==uterus_panel_count_sort[0][-1])
                pass
            elif uterus_panel_count_sort[-1] > count_threshold and uterus_panel_count_sort[-2] < count_threshold:
                select_index = np.where(self.uterus_panel_count==uterus_panel_count_sort[-1])
                img = self.index2uterus_motion(select_index[1], img)
                # self.uterus_panel_count[0,:6] = 0
                # self.uterus_panel_count[0,select_index[1]] = uterus_panel_count_sort[-1]
            elif uterus_panel_count_sort[-1] > count_threshold and uterus_panel_count_sort[-2] > count_threshold:
                # discard_index = np.where(self.uterus_panel_count==uterus_panel_count_sort[-1])
                # self.uterus_panel_count[0, discard_index] = 0
                select_index = np.where(self.uterus_panel_count==uterus_panel_count_sort[-2])
                img = self.index2uterus_motion(select_index[1], img)
                # self.uterus_panel_count[0,:6] = 0
                # self.uterus_panel_count[0,select_index[1]] = uterus_panel_count_sort[-2]
            self.uterus_panel_count_old = copy.deepcopy(self.uterus_panel_count)

        # Camera panel
        if self.mode == 'Camera':
            self.mode = 'None'
            '''img = ori_img.copy()
            # TODO: Draw the position again, sloved
            # TODO: xiaochu chanyu tuan
            img = merge_bbox_img(roi_bbox=self.camera_panel_zoomin_bbox_right, template=self.camera_panel_zoomin_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.camera_panel_zoomout_bbox_right, template=self.camera_panel_zoomout_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.camera_panel_zoomauto_bbox_right, template=self.camera_panel_zoomauto_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.camera_panel_setzero_bbox_right, template=self.camera_panel_setzero_icon, img=img, alpha=0.3, beta=0.7)
            img = merge_bbox_img(roi_bbox=self.main_camera_bbox_right_large, template=self.camera_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
            camera_panel_H = round(3*resize_H/4/4)
            if is_pos_in_bbox(position, self.camera_panel_zoomin_bbox_right):
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 0] + 1
                self.camera_panel_count[0, 1] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 4] = self.camera_panel_count[0, 5] = 0
            elif is_pos_in_bbox(position, self.camera_panel_zoomout_bbox_right):
                self.camera_panel_count[0, 1] = self.camera_panel_count[0, 1] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 4] = self.camera_panel_count[0, 5] = 0
            elif is_pos_in_bbox(position, self.camera_panel_zoomauto_bbox_right):
                self.camera_panel_count[0, 2] = self.camera_panel_count[0, 2] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 1] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 4] = self.camera_panel_count[0, 5] = 0
            elif is_pos_in_bbox(position, self.camera_panel_setzero_bbox_right):
                self.camera_panel_count[0, 3] = self.camera_panel_count[0, 3] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 1] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 4] = self.camera_panel_count[0, 5] = 0
            elif is_pos_in_bbox(position, self.main_camera_bbox_right_large):
                self.camera_panel_count[0, 4] = self.camera_panel_count[0, 4] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 1] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 5] = 0
            else:
                self.camera_panel_count[0, 5] = self.camera_panel_count[0, 5] + 1
                self.camera_panel_count[0, 0] = self.camera_panel_count[0, 1] = self.camera_panel_count[0, 2] = self.camera_panel_count[0, 3] = self.camera_panel_count[0, 4] = 0

            if self.camera_panel_count[0, 0] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.camera_panel_zoomin_bbox_right, template=self.camera_panel_zoomin_click_icon, img=img, alpha=0.3, beta=0.7)
                self.Return_Track_Mode = False # only adjust the depth
                self.Zoomin_Mode = True
                self.zoomin_action_once_flag = True
            elif self.camera_panel_count[0, 1] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.camera_panel_zoomout_bbox_right, template=self.camera_panel_zoomout_click_icon, img=img, alpha=0.3, beta=0.7)
                self.Return_Track_Mode = False # only adjust the depth
                self.Zoomout_Mode = True
                self.zoomout_action_once_flag = True
            elif self.camera_panel_count[0, 2] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.camera_panel_zoomauto_bbox_right, template=self.camera_panel_zoomauto_click_icon, img=img, alpha=0.3, beta=0.7)
                self.Return_Track_Mode = False # only adjust the depth
                self.Track_z_Mode = True
            elif self.camera_panel_count[0, 3] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.camera_panel_setzero_bbox_right, template=self.camera_panel_setzero_click_icon, img=img, alpha=0.3, beta=0.7)
                # self.Return_Mode = True
                self.target_status = False # necessary: to be used to back to initial position
                self.Return_Track_Mode = True
            elif self.camera_panel_count[0, 5] > count_threshold*3:
                img = img.copy()
                self.camera_panel_count[0, 5] = 0
                self.mode = 'None'''

        if position[0] >= 0 and position[1] >= 0:
            img = cv2.circle(img, (position[0], position[1]), 4, (123, 238, 253), 8)

        return img

    def load_icons(self):
        resize_W = 1440
        resize_H = 810
        ####################################### Load the icons
        ####################################### main interface buttons
        part_W = round(resize_W/16)
        part_W_l = round(resize_W/16*1.3)
        part_H = round(resize_H/8)
        main_uterus_icon_small = cv2.imread('./sources/GUI/main/uterus_1.png')
        main_uterus_icon_large = cv2.imread('./sources/GUI/main/uterus_2.png')
        self.main_uterus_icon_small = cv2.resize(main_uterus_icon_small, (part_W_l, part_H*2))
        self.main_uterus_icon_large = cv2.resize(main_uterus_icon_large, (part_W*2, part_H*2))

        main_domain_icon_small = cv2.imread('./sources/GUI/main/main_1.png')
        main_domain_icon_large = cv2.imread('./sources/GUI/main/main_2.png')
        main_domain_click_icon_large = cv2.imread('./sources/GUI/main/main_3.png')
        self.main_domain_icon_small = cv2.resize(main_domain_icon_small, (part_W_l, part_H*2))
        self.main_domain_icon_large = cv2.resize(main_domain_icon_large, (part_W*2, part_H*2))
        self.main_domain_click_icon_large = cv2.resize(main_domain_click_icon_large, (part_W*2, part_H*2))

        main_switch_icon_small = cv2.imread('./sources/GUI/main/switch_1.png')
        main_switch_left_icon_large = cv2.imread('./sources/GUI/main/switch_2_left.png')
        main_switch_right_icon_large = cv2.imread('./sources/GUI/main/switch_2_right.png')
        main_switch_left_click_icon_large = cv2.imread('./sources/GUI/main/switch_3_left.png')
        main_switch_right_click_icon_large = cv2.imread('./sources/GUI/main/switch_3_right.png')
        self.main_switch_icon_small = cv2.resize(main_switch_icon_small, (part_W_l, part_H*2))
        self.main_switch_left_icon_large = cv2.resize(main_switch_left_icon_large, (part_W*2, part_H*2))
        self.main_switch_right_icon_large = cv2.resize(main_switch_right_icon_large, (part_W*2, part_H*2))
        self.main_switch_left_click_icon_large = cv2.resize(main_switch_left_click_icon_large, (part_W*2, part_H*2))
        self.main_switch_right_click_icon_large = cv2.resize(main_switch_right_click_icon_large, (part_W*2, part_H*2))

        main_camera_icon_small = cv2.imread('./sources/GUI/main/camera_1.png')
        main_camera_icon_large = cv2.imread('./sources/GUI/main/camera_2.png')
        self.main_camera_icon_small = cv2.resize(main_camera_icon_small, (part_W_l, part_H*2))
        self.main_camera_icon_large = cv2.resize(main_camera_icon_large, (part_W*2, part_H*2))

        ####################################### camera interface buttons
        camera_panel_W = round(resize_W/8)
        camera_panel_H = round(3*resize_H/4/4)
        camera_panel_click_icon_large = cv2.imread('./sources/GUI/camera/camera_3.png')
        self.camera_panel_click_icon_large = cv2.resize(camera_panel_click_icon_large,(part_W*2, part_H*2))

        camera_panel_zoomin_icon = cv2.imread('./sources/GUI/camera/in_2.png')
        camera_panel_zoomin_click_icon = cv2.imread('./sources/GUI/camera/in_3.png')
        self.camera_panel_zoomin_icon = cv2.resize(camera_panel_zoomin_icon, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomin_click_icon = cv2.resize(camera_panel_zoomin_click_icon, (camera_panel_W, camera_panel_H))

        camera_panel_zoomout_icon = cv2.imread('./sources/GUI/camera/out_2.png')
        camera_panel_zoomout_click_icon = cv2.imread('./sources/GUI/camera/out_3.png')
        self.camera_panel_zoomout_icon = cv2.resize(camera_panel_zoomout_icon, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomout_click_icon = cv2.resize(camera_panel_zoomout_click_icon, (camera_panel_W, camera_panel_H))

        camera_panel_zoomauto_icon = cv2.imread('./sources/GUI/camera/track_2.png')
        camera_panel_zoomauto_click_icon = cv2.imread('./sources/GUI/camera/track_3.png')
        self.camera_panel_zoomauto_icon = cv2.resize(camera_panel_zoomauto_icon, (camera_panel_W, camera_panel_H))
        self.camera_panel_zoomauto_click_icon = cv2.resize(camera_panel_zoomauto_click_icon, (camera_panel_W, camera_panel_H))

        camera_panel_setzero_icon = cv2.imread('./sources/GUI/camera/zero_2.png')
        camera_panel_setzero_click_icon = cv2.imread('./sources/GUI/camera/zero_3.png')
        self.camera_panel_setzero_icon = cv2.resize(camera_panel_setzero_icon, (camera_panel_W, camera_panel_H))
        self.camera_panel_setzero_click_icon = cv2.resize(camera_panel_setzero_click_icon, (camera_panel_W, camera_panel_H))

        ####################################### uterus interface buttons
        uterus_panel_W = round(resize_W/16)
        uterus_panel_H = round(resize_H/8)
        uterus_panel_click_icon_large = cv2.imread('./sources/GUI/uterus/uterus_3.png')
        self.uterus_panel_click_icon_large = cv2.resize(uterus_panel_click_icon_large,(part_W*2, part_H*2))

        uterus_panel_down_icon = cv2.imread('./sources/GUI/uterus/down_2.png')
        uterus_panel_down_click_icon = cv2.imread('./sources/GUI/uterus/down_3.png')
        self.uterus_panel_down_icon = cv2.resize(uterus_panel_down_icon,(uterus_panel_W*4, uterus_panel_H))
        self.uterus_panel_down_click_icon = cv2.resize(uterus_panel_down_click_icon, (uterus_panel_W*4, uterus_panel_H))

        uterus_panel_up_icon = cv2.imread('./sources/GUI/uterus/up_2.png')
        uterus_panel_up_click_icon = cv2.imread('./sources/GUI/uterus/up_3.png')
        self.uterus_panel_up_icon = cv2.resize(uterus_panel_up_icon,(uterus_panel_W*4, uterus_panel_H))
        self.uterus_panel_up_click_icon = cv2.resize(uterus_panel_up_click_icon, (uterus_panel_W*4, uterus_panel_H))

        uterus_panel_left_icon = cv2.imread('./sources/GUI/uterus/left_2.png')
        uterus_panel_left_click_icon = cv2.imread('./sources/GUI/uterus/left_3.png')
        self.uterus_panel_left_icon = cv2.resize(uterus_panel_left_icon,(uterus_panel_W*2, uterus_panel_H*2))
        self.uterus_panel_left_click_icon = cv2.resize(uterus_panel_left_click_icon, (uterus_panel_W*2, uterus_panel_H*2))

        uterus_panel_right_icon = cv2.imread('./sources/GUI/uterus/right_2.png')
        uterus_panel_right_click_icon = cv2.imread('./sources/GUI/uterus/right_3.png')
        self.uterus_panel_right_icon = cv2.resize(uterus_panel_right_icon,(uterus_panel_W*2, uterus_panel_H*2))
        self.uterus_panel_right_click_icon = cv2.resize(uterus_panel_right_click_icon, (uterus_panel_W*2, uterus_panel_H*2))

        uterus_panel_insert_icon = cv2.imread('./sources/GUI/uterus/insert_2.png')
        uterus_panel_insert_click_icon = cv2.imread('./sources/GUI/uterus/insert_3.png')
        self.uterus_panel_insert_icon = cv2.resize(uterus_panel_insert_icon,(uterus_panel_W*2, uterus_panel_H*2))
        self.uterus_panel_insert_click_icon = cv2.resize(uterus_panel_insert_click_icon, (uterus_panel_W*2, uterus_panel_H*2))

        uterus_panel_retract_icon = cv2.imread('./sources/GUI/uterus/retract_2.png')
        uterus_panel_retract_click_icon = cv2.imread('./sources/GUI/uterus/retract_3.png')
        self.uterus_panel_retract_icon = cv2.resize(uterus_panel_retract_icon,(uterus_panel_W*2, uterus_panel_H*2))
        self.uterus_panel_retract_click_icon = cv2.resize(uterus_panel_retract_click_icon, (uterus_panel_W*2, uterus_panel_H*2))

        # Area setting for the icons
        # Main panel small bbox left
        self.main_uterus_bbox_left_small = [0, 0, part_W_l, 2*part_H] # x1, y1, x2, y2
        self.main_dominant_bbox_left_small = [0, 2*part_H, part_W_l, 4*part_H] # x1, y1, x2, y2
        self.main_switch_bbox_left_small = [0, 4*part_H, part_W_l, 6*part_H] # x1, y1, x2, y2
        self.main_camera_bbox_left_small = [0, 6*part_H, part_W_l, 8*part_H] # x1, y1, x2, y2
        # self.main_uterus_bbox_left_small = [part_W, 0, 2*part_W, 2*part_H] # x1, y1, x2, y2
        # self.main_dominant_bbox_left_small = [part_W, 2*part_H, 2*part_W, 4*part_H] # x1, y1, x2, y2
        # self.main_switch_bbox_left_small = [part_W, 4*part_H, 2*part_W, 6*part_H] # x1, y1, x2, y2
        # self.main_camera_bbox_left_small = [part_W, 6*part_H, 2*part_W, 8*part_H] # x1, y1, x2, y2
        # large bbox left
        self.main_uterus_bbox_left_large = [0, 0, 2*part_W, 2*part_H] # x1, y1, x2, y2
        self.main_dominant_bbox_left_large = [0, 2*part_H, 2*part_W, 4*part_H] # x1, y1, x2, y2
        self.main_switch_bbox_left_large = [0, 4*part_H, 2*part_W, 6*part_H] # x1, y1, x2, y2
        self.main_camera_bbox_left_large = [0, 6*part_H, 2*part_W, 8*part_H] # x1, y1, x2, y2
        # small bbox right
        self.main_uterus_bbox_right_small = [resize_W - part_W_l, 0, resize_W, 2*part_H] # x1, y1, x2, y2
        self.main_dominant_bbox_right_small = [resize_W - part_W_l, 2*part_H, resize_W, 4*part_H] # x1, y1, x2, y2
        self.main_switch_bbox_right_small = [resize_W - part_W_l, 4*part_H, resize_W, 6*part_H] # x1, y1, x2, y2
        self.main_camera_bbox_right_small = [resize_W - part_W_l, 6*part_H, resize_W, 8*part_H] # x1, y1, x2, y2
        # large bbox right
        self.main_uterus_bbox_right_large = [resize_W - 2*part_W, 0, resize_W, 2*part_H] # x1, y1, x2, y2
        self.main_dominant_bbox_right_large = [resize_W - 2*part_W, 2*part_H, resize_W, 4*part_H] # x1, y1, x2, y2
        self.main_switch_bbox_right_large = [resize_W - 2*part_W, 4*part_H, resize_W, 6*part_H] # x1, y1, x2, y2
        self.main_camera_bbox_right_large = [resize_W - 2*part_W, 6*part_H, resize_W, 8*part_H] # x1, y1, x2, y2

        ####################################### Camera panel
        self.camera_panel_zoomin_bbox_left = [0, 0, camera_panel_W, camera_panel_H] # x1, y1, x2, y2
        self.camera_panel_zoomout_bbox_left = [0, camera_panel_H, camera_panel_W, 2*camera_panel_H] # x1, y1, x2, y2
        self.camera_panel_zoomauto_bbox_left = [0, 2*camera_panel_H, camera_panel_W, 3*camera_panel_H] # x1, y1, x2, y2
        self.camera_panel_setzero_bbox_left = [0, 3*camera_panel_H, camera_panel_W, 4*camera_panel_H] # x1, y1, x2, y2
        #
        self.camera_panel_zoomin_bbox_right = [resize_W - camera_panel_W, 0, resize_W, camera_panel_H] # x1, y1, x2, y2
        self.camera_panel_zoomout_bbox_right = [resize_W - camera_panel_W, camera_panel_H, resize_W, 2*camera_panel_H] # x1, y1, x2, y2
        self.camera_panel_zoomauto_bbox_right = [resize_W - camera_panel_W, 2*camera_panel_H, resize_W, 3*camera_panel_H] # x1, y1, x2, y2
        self.camera_panel_setzero_bbox_right = [resize_W - camera_panel_W, 3*camera_panel_H, resize_W, 4*camera_panel_H] # x1, y1, x2, y2

        ####################################### Uterus panel   # no difference left or right
        self.uterus_panel_up_bbox = [round(resize_W/2-uterus_panel_W*2), 0, round(resize_W/2+uterus_panel_W*2), uterus_panel_H] # x1, y1, x2, y2
        self.uterus_panel_down_bbox = [round(resize_W/2-uterus_panel_W*2), resize_H-uterus_panel_H, round(resize_W/2+uterus_panel_W*2), resize_H] # x1, y1, x2, y2
        self.uterus_panel_left_bbox = [0, round(resize_H/2-uterus_panel_H), uterus_panel_W*2, round(resize_H/2+uterus_panel_H)] # x1, y1, x2, y2
        self.uterus_panel_right_bbox = [round(resize_W-uterus_panel_W*2), round(resize_H/2-uterus_panel_H), resize_W, round(resize_H/2+uterus_panel_H)] # x1, y1, x2, y2
        self.uterus_panel_insert_bbox = self.main_camera_bbox_left_large # x1, y1, x2, y2
        self.uterus_panel_retract_bbox = self.main_camera_bbox_right_large # x1, y1, x2, y2

        ####################################### panel count
        self.main_panel_count = np.zeros((1,5)) # extra one for disappear counting
        self.camera_panel_count = np.zeros((1,6)) # extra one for disappear counting
        self.uterus_panel_count = np.zeros((1,7)) # extra one for disappear counting
        self.uterus_panel_count_old = np.zeros((1,7)) # extra one for disappear counting

        ####################################### gui mode
        self.mode = 'None'


def merge_bbox_img(roi_bbox, template, img, alpha=0.5, beta=0.5):
    """
    bbox: x1, y1, x2, y2
    img: original image
    """
    img[roi_bbox[1]:roi_bbox[3],roi_bbox[0]:roi_bbox[2],::-1] = cv2.addWeighted(img[roi_bbox[1]:roi_bbox[3],roi_bbox[0]:roi_bbox[2],::-1], alpha, template, beta, 0)
    return img

def is_pos_in_bbox(pos, bbox):
    """
    :param pos: list [x, y]
    :param bbox: list [x1, y1, x2, y2], x1, y1 is the top left. x2, y2 is the bottom right
    :return: bool, True or False
    """
    if (pos[0] > bbox[0] and pos[0] < bbox[2]) and (pos[1] > bbox[1] and pos[1] < bbox[3]):
        return True
    else:
        return False
