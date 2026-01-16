from PyQt5 import QtWidgets
from GUI_interface import Ui_MainWindow
from Left_1080p import Ui_Left1080p
from Right_1080p import Ui_Right1080p

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())