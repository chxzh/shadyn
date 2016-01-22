from PySide.QtCore import *
from PySide.QtGui import *
import sys

class W1(QWidget):
    def __init__(self, parent=None):
        super(W1, self).__init__(parent)
        self.btn = QPushButton('Click1')

        vb = QVBoxLayout()
        vb.addWidget(self.btn)
        self.setLayout(vb)

        self.btn.clicked.connect(self.fireupWindows2)

    def fireupWindows2(self):
        w2 = W2()
        print w2.exec_()
        print w2.k
        print "end"
#         if w2.exec_():
#             self.w3 = W3()    
#             self.w3.show()

class W2(QDialog):
    def __init__(self, parent=None):
        super(W2, self).__init__(parent)

        self.btn = QPushButton('Click2')

        vb = QVBoxLayout()
        vb.addWidget(self.btn)
        self.setLayout(vb)

        self.btn.clicked.connect(self.fireupWindows3)

    def fireupWindows3(self):
        self.accept()
        self.k = 1
        self.w3 = W3()    
        self.w3.show()
        

class W3(QWidget):
    def __init__(self, parent=None):
        super(W3, self).__init__(parent)
        self.resize(300, 300)
        self.btn = QLabel('The Last Window')

        vb = QVBoxLayout()
        vb.addWidget(self.btn)
        self.setLayout(vb)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = W1()
    w.show()
    sys.exit(app.exec_())