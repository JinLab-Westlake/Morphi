from PyQt6 import QtWidgets, QtGui, QtCore


class InfoWindow(QtWidgets.QWidget):
    """
    轻量的独立信息窗：显示 Process Info 文本 + 日志文本。
    """
    def __init__(self, parent=None):
        super().__init__(parent, flags=QtCore.Qt.WindowType.Tool)  # 小工具窗，不占任务栏
        self.setWindowTitle("Process Info & Log")
        self.resize(520, 360)

        lay = QtWidgets.QVBoxLayout(self)

        # 一行状态 + 清空按钮
        top = QtWidgets.QHBoxLayout()
        self.lab_info = QtWidgets.QLabel("Process Info: -")
        self.lab_info.setStyleSheet("font-weight:600;")
        btn_clear = QtWidgets.QToolButton()
        btn_clear.setText("Clear log")
        btn_clear.clicked.connect(self._clear_log)
        top.addWidget(self.lab_info, 1)
        top.addWidget(btn_clear)

        # 日志编辑框（只读）
        self.text_log = QtWidgets.QTextEdit()
        self.text_log.setReadOnly(True)

        lay.addLayout(top)
        lay.addWidget(self.text_log, 1)

        # 置顶开关（可选）
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


