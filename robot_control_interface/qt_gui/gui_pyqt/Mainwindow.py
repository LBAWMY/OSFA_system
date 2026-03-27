import numpy as np
import cv2

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap

from functools import wraps
from Ui_MainWindow import Ui_MainWindow

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

class MainWindow(QMainWindow, Ui_MainWindow):
    @debug_class('MainWindow')
    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # Initialization: UI
        # self.label_img_show.setScaledContents(True)  # Show image with adaptive scale
        self.img_none = np.ones((480, 640, 3), dtype=np.uint8)*255
        self.img_green = np.zeros([480, 640, 3], np.uint8)
        self.img_green[:, :, 0] = np.zeros([480, 640]) + 255

        self.show_img(self.img_none)
        # Initialization: Network
        # ToDO
        # Other settings
        self.camera_left_index = 1
        self.video_flag = False

    @debug_class('MainWindow')
    def show_img(self, img):
        """
        show the numpy format image with the QLabel datatype
        :param img:
        :return:
        """
        show_img = QImage(img.data, img.shape[1], img.shape[0],
                          img.shape[1] * 3,
                          QImage.Format_RGB888)
        # cv2.imshow('Img', img)
        # if cv2.waitKey(20):
        print(img.shape)
        self.label_img_show.setPixmap(QPixmap.fromImage(show_img))

    @debug_class('MainWindow')
    @pyqtSlot()
    def pushButton_start_clicked(self):
        """
        open camera if the button was clicked
        :return:
        """
        self.cap_left = cv2.VideoCapture(self.camera_left_index)
        if self.cap_left.isOpened():
            # Disable this button
            self.pushButton_start.setEnabled(False)

            # acquire the image from the camera
            self.video_flag = True
            while self.video_flag:
                ret, self.img_left_src = self.cap_left.read()
                self.img_left_src = cv2.cvtColor(self.img_left_src, cv2.COLOR_BGR2RGB)
                # Using the network to detect the image
                # TODO
                # Using the actions from thr joystick
                self.actions = [0, 0]
                # self.img_left_src = self.draw_img(self.img_left_src, self.actions)
                self.show_img(self.img_left_src)

                # UI flash
                QApplication.processEvents()
        else:
            # self.textEdit.setText('No camera connected!')
            msg = QMessageBox.warning(self, 'warning', 'No camera connected!', buttons=QMessageBox.Ok)
            print('No camera connected!')

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
        # release the camera
        if hasattr(self, 'cap_left'):
            self.cap_left.release()
        self.pushButton_start.setEnabled(True)

    @debug_class('MainWindow')
    def draw_img(self, img, preds):
        """
        draw the action results in the image
        :param img:
        :param preds:
        :return:
        """
        #TODO
        pass

