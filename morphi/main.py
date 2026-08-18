from UI.GUI import Main
from PyQt6 import QtWidgets
import sys


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Main()
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec())
