# UI/Layout/display.py
from typing import Optional
from PyQt6 import QtWidgets,QtCore, QtGui
import cv2
import os
from Display_tool.zoom_label import ZoomableLabel

def to_qimage(img) -> QtGui.QImage:
    """支持 uint8/uint16, 灰度/BGR/BGRA -> QImage。优先保留16位灰度显示。"""
    import numpy as np
    if img is None:
        return None
    if img.ndim == 2:
        h, w = img.shape
        if img.dtype == np.uint16:
            # Qt6 支持 16位灰度
            qimg = QtGui.QImage(
                img.data, w, h, w * 2, QtGui.QImage.Format.Format_Grayscale16
            )
            return qimg.copy()
        elif img.dtype == np.uint8:
            qimg = QtGui.QImage(
                img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8
            )
            return qimg.copy()
        else:
            # 其他dtype做可视化归一化到8位
            vis8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            qimg = QtGui.QImage(
                vis8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8
            )
            return qimg.copy()
    elif img.ndim == 3:
        h, w, c = img.shape
        if c == 3:
            # BGR -> RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QtGui.QImage(
                rgb.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888
            )
            return qimg.copy()
        elif c == 4:
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            qimg = QtGui.QImage(
                rgba.data, w, h, w * 4, QtGui.QImage.Format.Format_RGBA8888
            )
            return qimg.copy()
        else:
            raise ValueError(f"Unsupported channels: {c}")
    else:
        raise ValueError(f"Unsupported ndim: {img.ndim}")

# def show_numpy_image(main, img, path: Optional[str] = None):
#     """将 ndarray 显示到 main.image_label，自动等比例缩放。"""
#     qimg = to_qimage(img)
#     if qimg is None:
#         return
#     main._qimage_ref = qimg  # 防止底层buffer被回收
#     pm = QtGui.QPixmap.fromImage(qimg).scaled(
#         main.image_label.size(),
#         QtCore.Qt.AspectRatioMode.KeepAspectRatio,
#         QtCore.Qt.TransformationMode.SmoothTransformation
#     )
#     main._pixmap_ref = pm
#     main.image_label.setPixmap(pm)
#
#     # 更新状态
#     main.last_img = img
#     if path is not None:
#         main.last_img_path = path
#         main.path_label.setText(path)
#     main.btn_save.setEnabled(True)

def clear_image_display(self):
    """清空显示区与状态；供删除/重置时调用。"""
    # 清掉 pixmap 与文本
    self.image_label.clear()
    self.image_label.setText("Please choose an image")
    self.path_label.setText("No file selected")

    # 状态复位
    self.last_img = None
    self.last_img_path = None
    self.last_pred = None
    self.last_mode = None

    # 保存按钮自动禁用（建议用统一更新函数）
    if hasattr(self, "_update_save_button"):
        self._update_save_button()
    else:
        self.btn_save.setEnabled(False)

    # 跑按钮状态联动
    self._update_run_buttons()


def install_autoresize(main):
    """让 label 大小变化时自动重绘以适配。需要在构建界面后调用一次。"""
    class _Filter(QtCore.QObject):
        def eventFilter(self, obj, ev):
            if obj is main.image_label and ev.type() == QtCore.QEvent.Type.Resize:
                if getattr(main, "_qimage_ref", None) is not None:
                    pm = QtGui.QPixmap.fromImage(main._qimage_ref).scaled(
                        main.image_label.size(),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation
                    )
                    main._pixmap_ref = pm
                    main.image_label.setPixmap(pm)
            return super().eventFilter(obj, ev)

    main._resize_filter = _Filter(main.image_label)
    main.image_label.installEventFilter(main._resize_filter)



def _build_center_panel(main) -> QtWidgets.QWidget:
    wrapper = QtWidgets.QWidget()
    w_lay = QtWidgets.QVBoxLayout(wrapper)
    w_lay.setContentsMargins(0, 0, 0, 0)
    # w_lay.setSpacing(6)

    # 顶部按钮行
    btn_row = QtWidgets.QHBoxLayout()
    btn_row.setContentsMargins(8,0,8, 20)
    # btn_row.setSpacing(10)

    # 路径行（新建 path_label 并挂到 main）
    path_row = QtWidgets.QHBoxLayout()
    path_row.setContentsMargins(8, 8, 8, 4)
    # path_row.setSpacing(6)

    path_caption = QtWidgets.QLabel("Path:")
    path_caption.setStyleSheet("color:#666;")

    main.path_label = QtWidgets.QLabel("No file selected")   # ★ 关键：创建并挂到 main
    main.path_label.setWordWrap(True)
    main.path_label.setStyleSheet("color: grey;")
    main.path_label.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )

    path_row.addWidget(path_caption)
    path_row.addWidget(main.path_label, 1)
    w_lay.addLayout(path_row)

    # 图像显示区
    center = QtWidgets.QScrollArea()
    center.setWidgetResizable(True)
    inner = QtWidgets.QWidget()
    c_lay = QtWidgets.QVBoxLayout(inner)
    c_lay.setContentsMargins(0, 0, 0, 0)


    main.image_label = ZoomableLabel("Tag or Select an image")
    main.image_label.zoom_always = False

    main.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    main.image_label.setBackgroundRole(QtGui.QPalette.ColorRole.Base)
    main.image_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                   QtWidgets.QSizePolicy.Policy.Ignored)
    main.image_label.setScaledContents(False)

    main.image_label.filesDropped.connect(main._on_files_dropped)


    main.btn_save = QtWidgets.QPushButton("Save current result")
    main.btn_save.setEnabled(False)
    main.btn_save.clicked.connect(main._save_overlay)



    c_lay.addWidget(main.image_label, 1)
    center.setWidget(inner)
    w_lay.addWidget(center, 1)


    btn_row.addWidget(main.btn_save)
    btn_row.addStretch(1)
    w_lay.addLayout(btn_row)

    return wrapper


class ZoomLabel(QtWidgets.QLabel):

    imageClicked = QtCore.pyqtSignal(int, int)  # (x, y) in image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._pixmap = None

    def set_zoom_pixmap(self, pm: QtGui.QPixmap, reset=True):
        self._pixmap = pm
        self.setPixmap(pm)  # 你已有的逻辑继续保留

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if not self._pixmap:
            return super().mousePressEvent(ev)

        # —— 将 label 点坐标映射到图像像素坐标 ——
        label_w = self.width()
        label_h = self.height()
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()

        # 居中等比缩放的常见映射
        scale = min(label_w / img_w, label_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        off_x = (label_w - disp_w) // 2
        off_y = (label_h - disp_h) // 2

        px = ev.position().x()
        py = ev.position().y()
        if not (off_x <= px < off_x + disp_w and off_y <= py < off_y + disp_h):
            return super().mousePressEvent(ev)

        ix = (px - off_x) / scale
        iy = (py - off_y) / scale
        ix = max(0, min(img_w - 1, int(round(ix))))
        iy = max(0, min(img_h - 1, int(round(iy))))

        self.imageClicked.emit(ix, iy)
        return super().mousePressEvent(ev)

