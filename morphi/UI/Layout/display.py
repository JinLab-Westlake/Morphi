# UI/Layout/display.py
from typing import Optional
from PyQt6 import QtWidgets,QtCore, QtGui
import cv2
import os
from Display_tool.zoom_label import ZoomableLabel

def to_qimage(img) -> QtGui.QImage:
    """Supports uint8/uint16, grayscale /BGR/BGRA -> QImage. Priority is reserved for 16-bit grayscale display."""
    import numpy as np
    if img is None:
        return None
    if img.ndim == 2:
        h, w = img.shape
        if img.dtype == np.uint16:
            # Qt6 supports 16-bit grayscale
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
            # Other Dtypes do visualization normalization to 8 bits
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


def clear_image_display(self):
    """Clear the display area and status; Called for deletion/reset."""
    # Clear the pixmap and text
    self.image_label.clear()
    self.image_label.setText("Please choose an image")
    self.path_label.setText("No file selected")

    # State reset
    self.last_img = None
    self.last_img_path = None
    self.last_pred = None
    self.last_mode = None

    # The Save button is automatically disabled
    if hasattr(self, "_update_save_button"):
        self._update_save_button()
    else:
        self.btn_save.setEnabled(False)

    # Run button state linkage
    self._update_run_buttons()


def install_autoresize(main):
    """Make the label redraw automatically when it changes size.
        It needs to be called once after building the interface."""
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

    # Top button row
    btn_row = QtWidgets.QHBoxLayout()
    btn_row.setContentsMargins(8,0,8, 20)
    # btn_row.setSpacing(10)

    # Path line
    path_row = QtWidgets.QHBoxLayout()
    path_row.setContentsMargins(8, 8, 8, 4)
    # path_row.setSpacing(6)

    path_caption = QtWidgets.QLabel("Path:")
    path_caption.setStyleSheet("color:#666;")

    main.path_label = QtWidgets.QLabel("No file selected")   # create path_label and attach to main
    main.path_label.setWordWrap(True)
    main.path_label.setStyleSheet("color: grey;")
    main.path_label.setTextInteractionFlags(
        QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
    )

    path_row.addWidget(path_caption)
    path_row.addWidget(main.path_label, 1)
    w_lay.addLayout(path_row)

    # Image display area
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
        self.setPixmap(pm)

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if not self._pixmap:
            return super().mousePressEvent(ev)

        # Map label point coordinates to image pixel coordinates ——
        label_w = self.width()
        label_h = self.height()
        img_w = self._pixmap.width()
        img_h = self._pixmap.height()

        # Common mappings for centered proportional scaling
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

