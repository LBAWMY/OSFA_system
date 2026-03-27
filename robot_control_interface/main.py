from PyQt5 import QtWidgets
from Mainwindow_ROS_MultiTools import MainWindow_ROS
import cv2
import time

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow_ROS()
    MainWindow.show()
    sys.exit(app.exec_())
