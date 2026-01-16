import rospy
import sys
import roslib
# roslib.load_manifest('ur5_endoscope_arm')
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image, Joy, JointState
# from ur5_endoscope_arm.msg import *
from robots.ur5_vision_msg import ur5_vision_msg

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap

from functools import wraps
from Ui_MainWindow import Ui_MainWindow

from robots.ur5 import UR5
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64

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
        self.camera_left_index = 1
        self.video_flag = False
        # self.vision_params = ur5_vision_msg()
        self.vision_params = {}
        self.Kc = np.array([[719.778534468158, 0.0, 379.165423468819],
                            [0.0, 787.884348579813, 273.066509594866],
                            [0.0, 0.0, 1.0]])

        # Initialization: ROS
        rospy.init_node("Mainwindow_node")
        # capture image
        self.bridge = CvBridge()
        # rospy.Subscriber("/camera2/usb_cam2/image_raw", Image, self.imageCallback)
        rospy.Subscriber("/camera1_2/usb_cam1_2/image_raw", Image, self.imageCallback)
        # capture joystick
        self.joy_src = None
        rospy.Subscriber("joy", Joy, self.joyCallback)
        # capture the ur5 joint state
        self.joint_state = None
        rospy.Subscriber("joint_states", JointState, self.jointStateCallback)
        # publish the vision params
        # self.vision_params_pub = rospy.Publisher("vision_params", ur5_vision_msg, queue_size=10)
        # publish command to control ur5
        self.ur5_crtl_script_pub = rospy.Publisher("ur_driver/URScript", String, queue_size=10)
        self.loop_rate = rospy.Rate(50) # 50hz
        # subscribe the predicted action
        rospy.Subscriber("/pred_action", Vector3, self.predActionCallback)
        # subscribe the estimated depth
        rospy.Subscriber("/tool_depth", Float64, self.toolDepthCallback)
        # subscribe the estimated position
        rospy.Subscriber("/pred_position", Vector3, self.predPositionCallback)

        # robot
        self.robot = None

        # load gui sources
        self.load_icons()

    def imageCallback(self, img):
        img_src = self.bridge.imgmsg_to_cv2(img, "bgr8")
        W, H = img_src.shape[:2]
        # self.img_left_src = cv2.resize(img_src[:, :int(H/2), :], (1440, 810))
        self.img_right_src = cv2.resize(img_src[:, int(H/2):, :], (1440, 810))

    def joyCallback(self, Joy):
        self.joy_src = Joy

    def jointStateCallback(self, jointState):
        self.joint_state = jointState
        self.joint_pos = np.array(self.joint_state.position,dtype=float)
    
    def predActionCallback(self, vector3):
        self.pred_action = vector3

    def predPositionCallback(self, vector3):
        self.pred_position = vector3

    def toolDepthCallback(self, depth):
        self.tool_depth = depth.data

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
            self.image_xy_speed_coef = 1.0
            xy_flag_add = True
            xy_flag_dec = True
            self.depth_zz_speed_coef = 1.0
            z_flag_add = True
            z_flag_dec = True

            # Manual/Automatic Mode
            Manual_Mode = True
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
            self.user_depth_flag = False

            # left-right view flag
            self.gui_side_left = True

            while self.video_flag:
                self.img_right_src_RGB = cv2.cvtColor(self.img_right_src, cv2.COLOR_BGR2RGB)

                # Use the button A,B,X,Y to adjuest the zoom-in/out and x/y axis speed along the image
                if self.joy_src.buttons[1] == 1 and xy_flag_add is True:
                    self.image_xy_speed_coef += 0.25
                    xy_flag_add = False
                if self.joy_src.buttons[1] == 0 and xy_flag_add is False:
                    xy_flag_add = True
                if self.joy_src.buttons[2] == 1 and xy_flag_dec is True:
                    self.image_xy_speed_coef -= 0.25
                    xy_flag_dec = False
                if self.joy_src.buttons[2] == 0 and xy_flag_dec is False:
                    xy_flag_dec = True

                if self.joy_src.buttons[3] == 1 and z_flag_add is True:
                    self.depth_zz_speed_coef += 0.25
                    z_flag_add = False
                if self.joy_src.buttons[3] == 0 and z_flag_add is False:
                    z_flag_add = True
                if self.joy_src.buttons[0] == 1 and z_flag_dec is True:
                    self.depth_zz_speed_coef -= 0.25
                    z_flag_dec = False
                if self.joy_src.buttons[0] == 0 and z_flag_dec is False:
                    z_flag_dec = True
                self.image_xy_speed_coef = np.clip(self.image_xy_speed_coef, a_min=0.25, a_max=3)
                self.depth_zz_speed_coef = np.clip(self.depth_zz_speed_coef, a_min=0.25, a_max=3)

                # adjust the left-right panel

                if self.joy_src.buttons[6] == 1 and Mode_flag is True:
                    Manual_Mode = not Manual_Mode # switch to other mode
                    Mode_flag = False
                if self.joy_src.buttons[6] == 0 and Mode_flag is False:
                    Mode_flag = True
                print('Mode flag: ', Mode_flag)

                if self.joy_src.buttons[7] == 1 and Return_flag is True and self.target_status is True:
                    self.Return_Mode = True
                    Return_flag = False
                    self.target_status = False
                if self.joy_src.buttons[7] == 0 and Return_flag is False:
                    Return_flag = True
                print('Return flag: ', Return_flag)

                if self.joy_src.buttons[4] == 1 and Zoomin_flag is True and self.target_status is True:
                    self.Zoomin_Mode = True
                    self.zoomin_action_once_flag = True
                    Zoomin_flag = False
                    self.target_status = False
                if self.joy_src.buttons[4] == 0 and Zoomin_flag is False:
                    Zoomin_flag = True
                print('Zoomin flag: ', Zoomin_flag)

                if self.joy_src.buttons[5] == 1 and Zoomout_flag is True and self.target_status is True:
                    self.Zoomout_Mode = True
                    self.zoomout_action_once_flag = True
                    Zoomout_flag = False
                    self.target_status = False
                if self.joy_src.buttons[5] == 0 and Zoomout_flag is False:
                    Zoomout_flag = True
                print('Zoomout flag: ', Zoomout_flag)

                if self.joy_src.axes[7] == 1 and Track_xy_flag is True:
                    self.Track_xy_Mode = not self.Track_xy_Mode
                    Track_xy_flag = False
                    # self.target_status = False
                if self.joy_src.axes[7] == 0 and Track_xy_flag is False:
                    Track_xy_flag = True
                print('Track_xy flag: ', Track_xy_flag)

                if self.joy_src.axes[7] == -1 and Track_z_flag is True:
                    self.Track_z_Mode = not self.Track_z_Mode
                    Track_z_flag = False
                if self.joy_src.axes[7] == 0 and Track_z_flag is False:
                    Track_z_flag = True
                print('Track_z flag: ', Track_z_flag)

                if self.joy_src.axes[6] == 1 and Emergency_flag is True:
                    self.Emergency_Mode = not self.Emergency_Mode
                    Emergency_flag = False
                if self.joy_src.axes[6] == 0 and Emergency_flag is False:
                    Emergency_flag = True
                print('Emergency flag: ', Emergency_flag)

                if self.joy_src.axes[6] == -1 and Return_Track_flag is True and self.target_status is True:
                    self.Return_Track_Mode = not self.Return_Track_Mode
                    Return_Track_flag = False
                    self.target_status = False
                if self.joy_src.axes[6] == 0 and Return_Track_flag is False:
                    Return_Track_flag = True
                print('Return_Track flag: ', Return_Track_flag)

                if self.Emergency_Mode is True:
                    self.Return_Mode = self.Zoomin_Mode = self.Zoomout_Mode = self.Track_xy_Mode = self.Track_z_Mode = False
                    Manual_Mode = True
                    self.target_status = True
                    self.user_depth_flag = False
                elif self.Return_Mode is True:
                    self.move_init()
                    actions_img = [0, 0, 0] # may need to multiply -1 in actions[0 or 1]
                    print('Return mode')
                elif self.Zoomin_Mode is True:
                    self.zoom_in()
                    actions_img = [0, 0, 0] # may need to multiply -1 in actions[0 or 1]
                    self.user_depth_flag = True # No automatic depth control this time
                    print('Zoomin mode')
                elif self.Zoomout_Mode is True:
                    self.zoom_out()
                    actions_img = [0, 0, 0] # may need to multiply -1 in actions[0 or 1]
                    self.user_depth_flag = True # No automatic depth control this time
                    print('Zoomout mode')
                elif self.Track_xy_Mode is True:
                    self.semi_track_xy()
                    actions_img = [-self.actions[0]/self.image_xy_speed_coef, -self.actions[1]/self.image_xy_speed_coef, 0.0] # may need to multiply -1 in actions[0 or 1]
                    print('--------------------------------------------------------------------------------semi-xy mode')
                elif self.Track_z_Mode is True:
                    self.Return_Track_Mode = False # init: always track the tool in xy axis; click the track z will zoom in/out in predefined, once finished, back to manual/automatic mode
                    self.user_depth_flag = False # use automatic depth control this time
                    self.semi_track_depth()
                    actions_img = [0.0, 0.0, self.actions[2]/self.depth_zz_speed_coef] # may need to multiply -1 in actions[0 or 1]
                    print('--------------------------------------------------------------------------------semi-z mode')
                elif self.Return_Track_Mode is True:
                    if self.target_status is False:
                        # first step
                        self.move_init()
                        actions_img = [0, 0, 0] # may need to multiply -1 in actions[0 or 1]
                    else:
                        # second step
                        self.semi_track_xy()
                        actions_img = [-self.actions[0]/self.image_xy_speed_coef, -self.actions[1]/self.image_xy_speed_coef, 0.0] # may need to multiply -1 in actions[0 or 1]
                else:
                    if Manual_Mode is True: # LT button
                        # Using the network to detect the image
                        # TODO
                        # Using the actions from the joystick
                        # self.actions = [self.joy_src.axes[0], self.joy_src.axes[1], self.joy_src.axes[4]] # [left(1)/right(-1), up(1)/down(-1), zoom in(1)/out(-1)]
                        # self.img_left_src_RGB = self.draw_img(self.img_left_src_RGB, self.actions)
                        # self.show_img(self.img_left_src_RGB)
                        #
                        # # calculate vision parameters(cVc/Errors) and publish the control signals
                        # self.vision_params = self.cal_vision_params(self.Kc, self.actions)
                        # self.vision_params_pub.publish(self.vision_params)
                        # cVc = self.vision_params.Parameter
                        # homo_delta = self.vision_params.ImageError[0:2]
                        #
                        # # drive the ur5 robot
                        self.pushButton_Test_clicked()
                        actions_img = [-self.actions[0]/self.image_xy_speed_coef, -self.actions[1]/self.image_xy_speed_coef, self.actions[2]/self.depth_zz_speed_coef] # may need to multiply -1 in actions[0 or 1]
                        print('Joystick mode')
                    else:
                        self.automatic_ctrl()
                        actions_img = [-self.actions[0]/self.image_xy_speed_coef, -self.actions[1]/self.image_xy_speed_coef, self.actions[2]/self.depth_zz_speed_coef] # may need to multiply -1 in actions[0 or 1]
                        print('Automatic mode')
                print('need to atttach the joystick')
                # self.actions = [self.joy_src.axes[0], self.joy_src.axes[1], self.joy_src.axes[4]] # [left(1)/right(-1), up(1)/down(-1), zoom in(1)/out(-1)]
                # some notions show in image
                xy_speed = f'V_xy speed: {self.image_xy_speed_coef:.2f}'
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
                    Mode = f'Mode: Manual' if Manual_Mode is True else f'Mode: Automatic'

                if self.mode == 'Camera':
                    gui_mode = f'Mode: Camera'
                elif self.mode =='Uterus':
                    gui_mode = f'Mode: Uterus'
                else:
                    gui_mode = f'Mode:'

                cv2.putText(self.img_right_src_RGB, Mode, (150, 20), 0, 0.5, [225, 255, 255], thickness=2,
                                    lineType=cv2.LINE_AA)
                cv2.putText(self.img_right_src_RGB, xy_speed, (320, 20), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)
                cv2.putText(self.img_right_src_RGB, z_speed, (480, 20), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)
                cv2.putText(self.img_right_src_RGB, gui_mode, (150, 60), 0, 0.5, [225, 255, 255], thickness=1,
                                    lineType=cv2.LINE_AA)


                # self.img_left_src_RGB = self.draw_img(self.img_left_src_RGB, actions_img)
                self.img_right_src_RGB = self.draw_img_seperate(self.img_right_src_RGB, actions_img)
                # draw the position
                scale_x = 1440 / 640
                scale_y = 810 / 480
                if (not np.isnan(self.pred_position.x)) and (not np.isnan(self.pred_position.y)):
                    positions_img = [int(self.pred_position.x * scale_x), int(self.pred_position.y * scale_y)]
                else:
                    positions_img = [-1, -1]

                positions_img_x = f'x pos: {positions_img[0]:d}'
                positions_img_y = f'y pos: {positions_img[1]:d}'
                cv2.putText(self.img_right_src_RGB, positions_img_x, (280, 60), 0, 0.5, [225, 255, 255], thickness=1,
                    lineType=cv2.LINE_AA)
                cv2.putText(self.img_right_src_RGB, positions_img_y, (480, 60), 0, 0.5, [225, 255, 255], thickness=1,
                    lineType=cv2.LINE_AA)

                if self.gui_side_left is True:
                    self.img_right_src_RGB = self.draw_img_GUI_left_side(self.img_right_src_RGB, positions_img)
                else:
                    self.img_right_src_RGB = self.draw_img_GUI_right_side(self.img_right_src_RGB, positions_img)
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
        """
        Test for control the laparoscope by joystick
        :return:
        """
        # calculate vision parameters(cVc/Errors) and publish the control signals
        # self.actions = [-self.joy_src.axes[0], -self.joy_src.axes[1], self.joy_src.axes[4]]
        self.actions = [-self.joy_src.axes[3] * self.image_xy_speed_coef, -self.joy_src.axes[4] * self.image_xy_speed_coef, self.joy_src.axes[1] * self.depth_zz_speed_coef]
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
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")
    
    def automatic_ctrl(self):
        """
        Control signals from predicted action
        :return:
        """
        # pass
        # calculate vision parameters(cVc/Errors) and publish the control signals
        if self.user_depth_flag is False: # automatic depth control
            self.actions = [-self.pred_action.x * self.image_xy_speed_coef, -self.pred_action.y * self.image_xy_speed_coef, self.pred_action.z * self.depth_zz_speed_coef]
        else: # no automatic depth control
            self.actions = [-self.pred_action.x * self.image_xy_speed_coef, -self.pred_action.y * self.image_xy_speed_coef, 0.0]
        self.vision_params = self.cal_vision_params(self.Kc, self.actions, depth=self.tool_depth, flag=1)
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
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def semi_track_xy(self):
        """
        Control the FOV only use the 2D position information
        :return:
        """
        # pass
        # calculate vision parameters(cVc/Errors) and publish the control signals
        self.actions = [-self.pred_action.x * self.image_xy_speed_coef, -self.pred_action.y * self.image_xy_speed_coef, 0.0]
        self.vision_params = self.cal_vision_params(self.Kc, self.actions, depth=self.tool_depth, flag=1)
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
        # if np.max(delta_q) <= 1e-6:
        #     self.target_status = True # False means the target has already achieved
        #     self.Track_xy_Mode = False
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def semi_track_depth(self):
        """
        Control FOV only consider the depth information
        :return:
        """
        # pass
        # calculate vision parameters(cVc/Errors) and publish the control signals
        self.actions = [0.0, 0.0, self.pred_action.z * self.depth_zz_speed_coef]
        self.vision_params = self.cal_vision_params(self.Kc, self.actions, depth=self.tool_depth, flag=1)
        # self.vision_params_pub.publish(self.vision_params)
        cVc = self.vision_params.Parameter
        homo_delta = self.vision_params.ImageError[0:2]

        # drive the ur5 robot
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)
        print("step1: cVc: ", cVc)
        print("homo_delta: ", homo_delta)
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        if np.max(delta_q) <= 1e-6:
            self.target_status = True # False means the target has already achieved
            self.Track_z_Mode = False
        # print('delta_q: ', delta_q)
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def move_init(self):
        """
        Moving to the init camera position
        :return:
        """
        # drive the ur5 robot, avoid init for multiple times
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)

        cVc = self.robot.bVc_to_cVc(self.joint_pos)
        homo_delta = np.array([0, 0]) / 2.0
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        if np.max(delta_q) <= 1e-6:
            self.target_status = True # False means the target has already achieved
            self.Return_Mode = False
        # print('delta_q: ', delta_q)
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def zoom_in(self):
        if self.robot is None:
            self.robot = UR5(init_joint_positions=self.joint_pos)

        cVc, self.zoomin_action_once_flag = self.robot.rVc_to_cVc(self.joint_pos, self.zoomin_action_once_flag, 1)
        print('zoomin_action_once_flag: ', self.zoomin_action_once_flag)
        homo_delta = np.array([0, 0]) / 2.0
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        if np.max(delta_q) <= 1e-6:
            self.target_status = True # False means the target has already achieved
            self.Zoomin_Mode = False
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def zoom_out(self):
        cVc, self.zoomout_action_once_flag = self.robot.rVc_to_cVc(self.joint_pos, self.zoomout_action_once_flag, -1)
        print('zoomout_action_once_flag: ', self.zoomout_action_once_flag)
        homo_delta = np.array([0, 0]) / 2.0
        delta_q = self.robot.cVc_to_deltaq(cVc, homo_delta, self.joint_pos)
        if np.max(delta_q) <= 1e-6:
            self.target_status = True # False means the target has already achieved
            self.Zoomout_Mode = False
        ur5_crtl_script = 'speedj([' + str(delta_q[0, 0]) + ', ' + str(delta_q[1, 0]) + ', ' + str(delta_q[2, 0]) + ', ' \
                          + str(delta_q[3, 0]) + ', ' + str(delta_q[4, 0]) + ', ' + str(delta_q[5, 0]) + '], 50, 0.02)'
        self.ur5_crtl_script_pub.publish(ur5_crtl_script + "\n")

    def cal_vision_params(self, Kc, actions, depth, flag):
        if flag == 0: # mannually control: constant depth
            c_z = depth
        elif flag == 1: # automatic control: use the tool depth
            print('depth: ', depth)
            if depth > 0.01 and depth < 0.30:
                print('success :', depth)
                c_z = depth
            else:
                c_z = 0.1 # use the default depth when the depth is invalid
        k_image = 0.05 # velocity of movement
        homo_delta = np.array([actions[0], actions[1]]) / 2.0
        u_ = actions[0] * Kc[0, 0] / 2.0
        v_ = actions[1] * Kc[1, 1] / 2.0
        # Lmatrix = np.array([[-Kc[0, 0]/c_z, 0, u_/c_z], [0, -Kc[1, 1]/c_z, v_/c_z]])
        # uvn = np.dot(np.linalg.pinv(Kc), np.array([240, 320, 1.0]))
        # # print(uvn)
        # Lmatrix = np.array([[-1/c_z, 0, uvn[0]/c_z], [0, -1/c_z, uvn[1]/c_z]])
        Lmatrix = np.array([[-1/c_z, 0, 0.0/c_z], [0, -1/c_z, 0.0/c_z]])
        cVc = -1 * k_image * np.dot(np.linalg.pinv(Lmatrix), homo_delta)
        d_error = actions[2]
        # cVc[2] += -1 * k_image * 0.0015 * d_error
        cVc[2] += -1 * k_image * 0.10 * d_error
        print("cVc = \n", cVc)

        #set boundary of the position cmd
        step_limitation = 0.10
        if np.linalg.norm(cVc) > step_limitation:
            cVc = cVc * step_limitation / np.linalg.norm(cVc)
        # vision_params = ur5_vision_msg()
        vision_params = ur5_vision_msg
        vision_params.Parameter = cVc[0:3] # new cmd
        vision_params.ImageError = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        vision_params.ImageError[0:2] = homo_delta[0:2] # publish img errors
        # vision_params.ImageError[2:4] = np.array([0.0, 0.0]) # publish target current positions
        # vision_params.ImageError[4:6] = h_dy[0:2] # publish desired img features
        # vision_params.ImageError[6] = 0.0 # d0 here
        # vision_params.ImageError[7] = d_error # publish desired img features
        return vision_params

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
    def draw_img(self, img, actions):
        """
        draw the action results in the image
        :param img:
        :param actions:
        :return:
        """
        h, w, c = img.shape
        # print('w, h: ', w, h)
        rect_wh = 5
        # left/right action
        if actions[0] > 0: # left case
            pt1 = (0, int((0.5 - actions[0] / 2.0) * h))
            pt2 = (rect_wh, int((0.5 + actions[0] / 2.0) * h))
        elif actions[0] < 0: # right case
            pt1 = (w - rect_wh, int((0.5 + actions[0] / 2.0) * h))
            pt2 = (w, int((0.5 - actions[0] / 2.0) * h))
        else:
            pt1 = (0, 0)
            pt2 = (0, 0)

        # up/down action
        if actions[1] > 0: # up case
            pt3 = (int((0.5 - actions[1] / 2.0) * w), 0)
            pt4 = (int((0.5 + actions[1] / 2.0) * w), rect_wh)
        elif actions[1] < 0: # down case
            pt3 = (int((0.5 + actions[1] / 2.0) * w), h - rect_wh)
            pt4 = (int((0.5 - actions[1] / 2.0) * w), h)
        else:
            pt3 = (0, 0)
            pt4 = (0, 0)

        # draw the rect in image
        img = cv2.rectangle(img, pt1, pt2, (255, 255, 0), thickness=-1)
        img = cv2.rectangle(img, pt3, pt4, (255, 255, 0), thickness=-1)
        return img

    @debug_class('MainWindow')
    def draw_img_seperate(self, img, actions):
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

        refer_h = 7/8 * h
        refer_w = 5/6 * w

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
    def draw_img_GUI_left_side(self, ori_img, position):
        resize_W = 1440
        resize_H = 810
        part_W = round(resize_W/16)
        part_H = round(resize_H/8)
        # draw the tool center in image
        img = ori_img.copy()
        count_threshold = 50
        # draw the GUI interface for manipulation
        if position[0] < part_W * 2 and position[0] > 0 and self.mode == 'None':
            # self.mode = 'Main' # need once flag to set this variable
            print('uterus shape: ', self.uterus_panel_up_icon.shape)
            print('uterus bbox: ', self.main_dominant_bbox_left_small)
            if is_pos_in_bbox(position, self.main_uterus_bbox_left_small):
                # img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.main_uterus_icon_large, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 0] = self.main_panel_count[0, 0] + 1
                self.main_panel_count[0, 1] = self.main_panel_count[0, 2] = self.main_panel_count[0, 3] = 0
                if self.main_panel_count[0, 0] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.uterus_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    self.mode = 'Uterus'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.main_uterus_icon_large, img=img, alpha=0.3, beta=0.7)
            elif is_pos_in_bbox(position, self.main_dominant_bbox_left_small):
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
                    img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_large, template=self.main_camera_icon_large, img=img, alpha=0.3, beta=0.7)
            else:
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count = np.zeros((1,4))
        else:
            self.main_panel_count = np.zeros((1,4))


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
            img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.uterus_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
            uterus_panel_W = round(resize_W/16)
            uterus_panel_H = round(resize_H/8)
            if is_pos_in_bbox(position, self.uterus_panel_up_bbox):
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 0] + 1
                self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_down_bbox):
                self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 1] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_left_bbox):
                self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 2] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_right_bbox):
                self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 3] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_insert_bbox):
                self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 4] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_retract_bbox):
                self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 5] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 6] = 0
            else:
                self.uterus_panel_count[0, 6] = self.uterus_panel_count[0, 6] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = 0

            if self.uterus_panel_count[0, 0] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_up_bbox, template=self.uterus_panel_up_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 1] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_down_bbox, template=self.uterus_panel_down_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 2] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_left_bbox, template=self.uterus_panel_left_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 3] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_right_bbox, template=self.uterus_panel_right_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 4] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_insert_bbox, template=self.uterus_panel_insert_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 5] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_retract_bbox, template=self.uterus_panel_retract_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 6] > count_threshold*5:
                img = img.copy()
                self.uterus_panel_count[0, 6] = 0
                self.mode = 'None'
        # Camera panel
        if self.mode == 'Camera':
            img = ori_img.copy()
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
                self.mode = 'None'
        if position[0] >= 0 and position[1] >= 0:
            img = cv2.circle(img, (position[0], position[1]), 4, (123, 238, 253), 8)

        return img

    @debug_class('MainWindow')
    def draw_img_GUI_right_side(self, ori_img, position):
        resize_W = 1440
        resize_H = 810
        part_W = round(resize_W/16)
        part_H = round(resize_H/8)
        # draw the tool center in image
        img = ori_img.copy()
        count_threshold = 50

        # draw the GUI interface for manipulation
        if position[0] > resize_W - part_W * 2 and position[0] > 0 and self.mode == 'None':
            # self.mode = 'Main' # need once flag to set this variable
            print('uterus shape: ', self.uterus_panel_up_icon.shape)
            print('uterus bbox: ', self.main_dominant_bbox_right_small)
            if is_pos_in_bbox(position, self.main_uterus_bbox_right_small):
                # img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_left_large, template=self.main_uterus_icon_large, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_right_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_switch_bbox_right_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_camera_bbox_right_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 0] = self.main_panel_count[0, 0] + 1
                self.main_panel_count[0, 1] = self.main_panel_count[0, 2] = self.main_panel_count[0, 3] = 0
                if self.main_panel_count[0, 0] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_large, template=self.uterus_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    self.mode = 'Uterus'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_large, template=self.main_uterus_icon_large, img=img, alpha=0.3, beta=0.7)
            elif is_pos_in_bbox(position, self.main_dominant_bbox_right_small):
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_left_large, template=self.main_domain_icon_large, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_switch_bbox_right_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_camera_bbox_right_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 1] = self.main_panel_count[0, 1] + 1
                self.main_panel_count[0, 0] = self.main_panel_count[0, 2] = self.main_panel_count[0, 3] = 0
                if self.main_panel_count[0, 1] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_right_large, template=self.main_domain_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    # self.mode = 'Dominant'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_right_large, template=self.main_domain_icon_large, img=img, alpha=0.3, beta=0.7)
            elif is_pos_in_bbox(position, self.main_switch_bbox_right_small):
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_right_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_switch_bbox_left_large, template=self.main_switch_left_icon_large, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_camera_bbox_right_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 2] = self.main_panel_count[0, 2] + 1
                self.main_panel_count[0, 0] = self.main_panel_count[0, 1] = self.main_panel_count[0, 3] = 0
                if self.main_panel_count[0, 2] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_switch_bbox_right_large, template=self.main_switch_left_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    self.gui_side_left = not self.gui_side_left
                    # self.mode = 'Switch'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_switch_bbox_right_large, template=self.main_switch_left_icon_large, img=img, alpha=0.3, beta=0.7)
            elif is_pos_in_bbox(position, self.main_camera_bbox_right_small):
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_right_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_switch_bbox_right_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                # img = merge_bbox_img(roi_bbox=self.main_camera_bbox_left_large, template=self.main_camera_icon_large, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count[0, 3] = self.main_panel_count[0, 3] + 1
                self.main_panel_count[0, 0] = self.main_panel_count[0, 1] = self.main_panel_count[0, 2] = 0
                if self.main_panel_count[0, 3] > count_threshold:
                    img = merge_bbox_img(roi_bbox=self.main_camera_bbox_right_large, template=self.camera_panel_click_icon_large, img=img, alpha=0.3, beta=0.7)
                    self.main_panel_count[0, 3] = 0
                    self.mode = 'Camera'
                else:
                    img = merge_bbox_img(roi_bbox=self.main_camera_bbox_right_large, template=self.main_camera_icon_large, img=img, alpha=0.3, beta=0.7)
            else:
                img = merge_bbox_img(roi_bbox=self.main_uterus_bbox_right_small, template=self.main_uterus_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_dominant_bbox_right_small, template=self.main_domain_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_switch_bbox_right_small, template=self.main_switch_icon_small, img=img, alpha=0.3, beta=0.7)
                img = merge_bbox_img(roi_bbox=self.main_camera_bbox_right_small, template=self.main_camera_icon_small, img=img, alpha=0.3, beta=0.7)
                self.main_panel_count = np.zeros((1,4))
        else:
            self.main_panel_count = np.zeros((1,4))


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
            if is_pos_in_bbox(position, self.uterus_panel_up_bbox):
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 0] + 1
                self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_down_bbox):
                self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 1] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_left_bbox):
                self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 2] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_right_bbox):
                self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 3] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_insert_bbox):
                self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 4] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 6] = 0
            elif is_pos_in_bbox(position, self.uterus_panel_retract_bbox):
                self.uterus_panel_count[0, 5] = self.uterus_panel_count[0, 5] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 6] = 0
            else:
                self.uterus_panel_count[0, 6] = self.uterus_panel_count[0, 6] + 1
                self.uterus_panel_count[0, 0] = self.uterus_panel_count[0, 1] = self.uterus_panel_count[0, 2] = self.uterus_panel_count[0, 3] = self.uterus_panel_count[0, 4] = self.uterus_panel_count[0, 5] = 0

            if self.uterus_panel_count[0, 0] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_up_bbox, template=self.uterus_panel_up_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 1] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_down_bbox, template=self.uterus_panel_down_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 2] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_left_bbox, template=self.uterus_panel_left_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 3] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_right_bbox, template=self.uterus_panel_right_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 4] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_insert_bbox, template=self.uterus_panel_insert_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 5] > count_threshold:
                img = merge_bbox_img(roi_bbox=self.uterus_panel_retract_bbox, template=self.uterus_panel_retract_click_icon, img=img, alpha=0.3, beta=0.7)
            elif self.uterus_panel_count[0, 6] > count_threshold*5:
                img = img.copy()
                self.uterus_panel_count[0, 6] = 0
                self.mode = 'None'
        # Camera panel
        if self.mode == 'Camera':
            img = ori_img.copy()
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
                self.mode = 'None'

        if position[0] >= 0 and position[1] >= 0:
            img = cv2.circle(img, (position[0], position[1]), 4, (123, 238, 253), 8)

        return img

    def load_icons(self):
        resize_W = 1440
        resize_H = 810
        ####################################### Load the icons
        ####################################### main interface buttons
        part_W = round(resize_W/16)
        part_H = round(resize_H/8)
        main_uterus_icon_small = cv2.imread('./sources/GUI/main/uterus_1.png')
        main_uterus_icon_large = cv2.imread('./sources/GUI/main/uterus_2.png')
        self.main_uterus_icon_small = cv2.resize(main_uterus_icon_small, (part_W, part_H*2))
        self.main_uterus_icon_large = cv2.resize(main_uterus_icon_large, (part_W*2, part_H*2))

        main_domain_icon_small = cv2.imread('./sources/GUI/main/main_1.png')
        main_domain_icon_large = cv2.imread('./sources/GUI/main/main_2.png')
        main_domain_click_icon_large = cv2.imread('./sources/GUI/main/main_3.png')
        self.main_domain_icon_small = cv2.resize(main_domain_icon_small, (part_W, part_H*2))
        self.main_domain_icon_large = cv2.resize(main_domain_icon_large, (part_W*2, part_H*2))
        self.main_domain_click_icon_large = cv2.resize(main_domain_click_icon_large, (part_W*2, part_H*2))

        main_switch_icon_small = cv2.imread('./sources/GUI/main/switch_1.png')
        main_switch_left_icon_large = cv2.imread('./sources/GUI/main/switch_2_left.png')
        main_switch_right_icon_large = cv2.imread('./sources/GUI/main/switch_2_right.png')
        main_switch_left_click_icon_large = cv2.imread('./sources/GUI/main/switch_3_left.png')
        main_switch_right_click_icon_large = cv2.imread('./sources/GUI/main/switch_3_right.png')
        self.main_switch_icon_small = cv2.resize(main_switch_icon_small, (part_W, part_H*2))
        self.main_switch_left_icon_large = cv2.resize(main_switch_left_icon_large, (part_W*2, part_H*2))
        self.main_switch_right_icon_large = cv2.resize(main_switch_right_icon_large, (part_W*2, part_H*2))
        self.main_switch_left_click_icon_large = cv2.resize(main_switch_left_click_icon_large, (part_W*2, part_H*2))
        self.main_switch_right_click_icon_large = cv2.resize(main_switch_right_click_icon_large, (part_W*2, part_H*2))

        main_camera_icon_small = cv2.imread('./sources/GUI/main/camera_1.png')
        main_camera_icon_large = cv2.imread('./sources/GUI/main/camera_2.png')
        self.main_camera_icon_small = cv2.resize(main_camera_icon_small, (part_W, part_H*2))
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
        self.main_uterus_bbox_left_small = [0, 0, part_W, 2*part_H] # x1, y1, x2, y2
        self.main_dominant_bbox_left_small = [0, 2*part_H, part_W, 4*part_H] # x1, y1, x2, y2
        self.main_switch_bbox_left_small = [0, 4*part_H, part_W, 6*part_H] # x1, y1, x2, y2
        self.main_camera_bbox_left_small = [0, 6*part_H, part_W, 8*part_H] # x1, y1, x2, y2
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
        self.main_uterus_bbox_right_small = [resize_W - part_W, 0, resize_W, 2*part_H] # x1, y1, x2, y2
        self.main_dominant_bbox_right_small = [resize_W - part_W, 2*part_H, resize_W, 4*part_H] # x1, y1, x2, y2
        self.main_switch_bbox_right_small = [resize_W - part_W, 4*part_H, resize_W, 6*part_H] # x1, y1, x2, y2
        self.main_camera_bbox_right_small = [resize_W - part_W, 6*part_H, resize_W, 8*part_H] # x1, y1, x2, y2
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
        # TODO: need to be check!
        self.camera_panel_zoomin_bbox_right = [resize_W - camera_panel_W, 0, resize_W, camera_panel_H] # x1, y1, x2, y2
        self.camera_panel_zoomout_bbox_right = [resize_W - camera_panel_W, camera_panel_H, resize_W, 2*camera_panel_H] # x1, y1, x2, y2
        self.camera_panel_zoomauto_bbox_right = [resize_W - camera_panel_W, 2*camera_panel_H, resize_W, 3*camera_panel_H] # x1, y1, x2, y2
        self.camera_panel_setzero_bbox_right = [resize_W - camera_panel_W, 3*camera_panel_H, resize_W, 4*camera_panel_H] # x1, y1, x2, y2

        ####################################### Uterus panel
        self.uterus_panel_up_bbox = [round(resize_W/2-uterus_panel_W*2), 0, round(resize_W/2+uterus_panel_W*2), uterus_panel_H] # x1, y1, x2, y2
        self.uterus_panel_down_bbox = [round(resize_W/2-uterus_panel_W*2), resize_H-uterus_panel_H, round(resize_W/2+uterus_panel_W*2), resize_H] # x1, y1, x2, y2
        self.uterus_panel_left_bbox = [0, round(resize_H/2-uterus_panel_H), uterus_panel_W*2, round(resize_H/2+uterus_panel_H)] # x1, y1, x2, y2
        self.uterus_panel_right_bbox = [round(resize_W-uterus_panel_W*2), round(resize_H/2-uterus_panel_H), resize_W, round(resize_H/2+uterus_panel_H)] # x1, y1, x2, y2
        self.uterus_panel_insert_bbox = self.main_camera_bbox_left_large # x1, y1, x2, y2
        self.uterus_panel_retract_bbox = self.main_camera_bbox_right_large # x1, y1, x2, y2

        ####################################### panel count
        self.main_panel_count = np.zeros((1,5)) # extra one for disappear counting
        self.camera_panel_count = np.zeros((1,6)) # extra one for disappear counting
        self.uterus_panel_count = np.zeros((1,8)) # extra one for disappear counting

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


# img_show = img_resize.copy()
# img_show = merge_bbox_img(roi_bbox=uterus_bbox_left_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=dominant_bbox_left_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=switch_bbox_left_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=camera_bbox_left_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# plt.imshow(img_show[:,:,::-1])
#
# img_show = img_resize.copy()
# img_show = merge_bbox_img(roi_bbox=uterus_bbox_left_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=dominant_bbox_left_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=switch_bbox_left_large, template=test_l, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=camera_bbox_left_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# plt.imshow(img_show[:,:,::-1])
#
# # swith other side
# img_show = img_resize.copy()
# img_show = merge_bbox_img(roi_bbox=uterus_bbox_right_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=dominant_bbox_right_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=switch_bbox_right_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=camera_bbox_right_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# plt.imshow(img_show[:,:,::-1])
#
# # swith other side
# img_show = img_resize.copy()
# img_show = merge_bbox_img(roi_bbox=uterus_bbox_right_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=dominant_bbox_right_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=switch_bbox_right_large, template=test_l, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=camera_bbox_right_small, template=test_s, img=img_show, alpha=0.3, beta=0.7)
# plt.imshow(img_show[:,:,::-1])
#
# # camera panel
# resize_W = 1920
# resize_H = 1080
# camera_panel_W = round(resize_W/8)
# camera_panel_H = round(3*resize_H/4/4)
# test_camera_panel = cv2.resize(test,(camera_panel_W, camera_panel_H))
# # small bbox left
# camera_panel_zoomin_bbox_left_small = [0, 0, camera_panel_W, camera_panel_H] # x1, y1, x2, y2
# camera_panel_zoomout_bbox_left_small = [0, camera_panel_H, camera_panel_W, 2*camera_panel_H] # x1, y1, x2, y2
# camera_panel_zoomauto_bbox_left_small = [0, 2*camera_panel_H, camera_panel_W, 3*camera_panel_H] # x1, y1, x2, y2
# camera_panel_setzero_bbox_left_small = [0, 3*camera_panel_H, camera_panel_W, 4*camera_panel_H] # x1, y1, x2, y2
#
# # camera control panel
# img_show = img_resize.copy()
# img_show = merge_bbox_img(roi_bbox=camera_panel_zoomin_bbox_left_small, template=test_camera_panel, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=camera_panel_zoomout_bbox_left_small, template=test_camera_panel, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=camera_panel_zoomauto_bbox_left_small, template=test_camera_panel, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=camera_panel_setzero_bbox_left_small, template=test_camera_panel, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=camera_bbox_left_large, template=test_l, img=img_show, alpha=0.3, beta=0.7)
# plt.imshow(img_show[:,:,::-1])
#
# # uterus control panel
# resize_W = 1920
# resize_H = 1080
# uterus_panel_W = round(resize_W/16)
# uterus_panel_H = round(resize_H/8)
# test_uterus_panel_up_down = cv2.resize(test,(uterus_panel_W*4, uterus_panel_H))
# test_uterus_panel_others = cv2.resize(test,(uterus_panel_W*2, uterus_panel_H*2))
# # small bbox left
# uterus_panel_up_bbox = [round(resize_W/2-uterus_panel_W*2), 0, round(resize_W/2+uterus_panel_W*2), uterus_panel_H] # x1, y1, x2, y2
# uterus_panel_down_bbox = [round(resize_W/2-uterus_panel_W*2), resize_H-uterus_panel_H, round(resize_W/2+uterus_panel_W*2), resize_H] # x1, y1, x2, y2
# uterus_panel_left_bbox = [0, round(resize_H/2-uterus_panel_H), uterus_panel_W*2, round(resize_H/2+uterus_panel_H)] # x1, y1, x2, y2
# uterus_panel_right_bbox = [round(resize_W-uterus_panel_W*2), round(resize_H/2-uterus_panel_H), resize_W, round(resize_H/2+uterus_panel_H)] # x1, y1, x2, y2
# uterus_panel_insert_bbox = camera_bbox_left_large # x1, y1, x2, y2
# uterus_panel_retract_bbox = camera_bbox_right_large # x1, y1, x2, y2
#
# img_show = img_resize.copy()
# img_show = merge_bbox_img(roi_bbox=uterus_panel_up_bbox, template=test_uterus_panel_up_down, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_down_bbox, template=test_uterus_panel_up_down, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_left_bbox, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_right_bbox, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_insert_bbox, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_retract_bbox, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_bbox_left_large, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7)
# plt.imshow(img_show[:,:,::-1])
#
# img_show = img_resize.copy()
# img_show = merge_bbox_img(roi_bbox=uterus_panel_up_bbox, template=test_uterus_panel_up_down, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_down_bbox, template=test_uterus_panel_up_down, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_left_bbox, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_right_bbox, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_insert_bbox, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_panel_retract_bbox, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7)
# img_show = merge_bbox_img(roi_bbox=uterus_bbox_right_large, template=test_uterus_panel_others, img=img_show, alpha=0.3, beta=0.7) # other side
# plt.imshow(img_show[:,:,::-1])
#
#
#
#
# point_color = (255, 255, 255) # BGR
# # L_Img_color = cv2.circle(L_Img, (random_point_x, random_point_y), 2, point_color, 2)
# # L_Img_Undist_color = cv2.circle(L_Img_Undist, (map1_1[random_point_y, random_point_x, 0], map1_1[random_point_y, random_point_x, 1]), 2, point_color, 2)
# L_Img_color = cv2.circle(L_Img, (map1_1[random_point_y, random_point_x, 0], map1_1[random_point_y, random_point_x, 1]), 2, point_color, 2)
# L_Img_Undist_color = cv2.circle(L_Img_Undist, (random_point_x, random_point_y), 2, point_color, 2)
