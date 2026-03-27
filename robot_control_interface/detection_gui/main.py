from PyQt5 import QtWidgets
from Mainwindow_ROS_new_gui import MainWindow_ROS

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow_ROS()
    MainWindow.show()
    sys.exit(app.exec_())
