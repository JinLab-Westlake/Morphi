from PyQt6 import QtWidgets, QtGui, QtCore


class InfoWindow(QtWidgets.QWidget):
    """
    Lightweight independent information window: Display Process Info text + log text.
    """
    def __init__(self, parent=None):
        super().__init__(parent, flags=QtCore.Qt.WindowType.Tool)  # The widget window does not occupy the taskbar
        self.setWindowTitle("Process Info & Log")
        self.resize(520, 360)

        lay = QtWidgets.QVBoxLayout(self)

        # A row of status + Clear buttons
        top = QtWidgets.QHBoxLayout()
        self.lab_info = QtWidgets.QLabel("Process Info: -")
        self.lab_info.setStyleSheet("font-weight:600;")
        btn_clear = QtWidgets.QToolButton()
        btn_clear.setText("Clear log")
        btn_clear.clicked.connect(self._clear_log)
        top.addWidget(self.lab_info, 1)
        top.addWidget(btn_clear)

        # Log edit box (Read only)
        self.text_log = QtWidgets.QTextEdit()
        self.text_log.setReadOnly(True)

        lay.addLayout(top)
        lay.addWidget(self.text_log, 1)

        # Top switch (optional)
        row = QtWidgets.QHBoxLayout()
        self.chk_top = QtWidgets.QCheckBox("Always on top")
        self.chk_top.toggled.connect(self._toggle_on_top)
        row.addStretch(1)
        row.addWidget(self.chk_top)
        lay.addLayout(row)

    def set_info(self, text: str):
        self.lab_info.setText(text)

    def append_log(self, text: str):
        if not text:
            return
        self.text_log.append(text)

    def _clear_log(self):
        self.text_log.clear()

    def _toggle_on_top(self, on: bool):
        flags = self.windowFlags()
        if on:
            flags |= QtCore.Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~QtCore.Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()


