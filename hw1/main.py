import sys
from ui import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
import Q1
import Q2
import Q3
import Q4

class MainWindow(QtWidgets.QMainWindow):
     def __init__(self):
        super(MainWindow, self).__init__(None)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(Q1.find_corner)
        self.ui.pushButton_2.clicked.connect(Q1.find_intrinsic)
        self.ui.pushButton_4.clicked.connect(Q1.find_distortion)
        self.ui.pushButton_5.clicked.connect(Q1.show_result)
        self.ui.pushButton_3.clicked.connect(lambda:Q1.find_extrinsic(self.ui.lineEdit_2.text()))

        self.ui.pushButton_9.clicked.connect(Q4.keypoint)
        self.ui.pushButton_10.clicked.connect(Q4.matched_keypoint)
        self.ui.pushButton_11.clicked.connect(Q4.wrap_image)
    
        self.ui.pushButton_8.clicked.connect(Q3.Stereo_Disparity_Map)
        
        self.ui.pushButton_6.clicked.connect(lambda:Q2.show_word(self.ui.lineEdit.text()))
        self.ui.pushButton_7.clicked.connect(lambda:Q2.show_vertically(self.ui.lineEdit.text()))
    

if __name__=="__main__":
    Application = QtWidgets.QApplication(sys.argv)
    window=MainWindow()
    window.show()
    sys.exit(Application.exec_())




    
