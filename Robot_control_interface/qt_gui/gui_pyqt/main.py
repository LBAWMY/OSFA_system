from PyQt5 import QtWidgets
from Mainwindow import MainWindow
from Mainwindow_ROS import MainWindow_ROS

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow_ROS()
    MainWindow.show()
    sys.exit(app.exec_())
