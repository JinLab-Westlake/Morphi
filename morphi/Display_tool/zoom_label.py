from typing import Optional
from PyQt6 import QtWidgets, QtCore, QtGui


class ZoomableLabel(QtWidgets.QLabel):
    """
    带缩放/平移的图像显示控件，并提供 imageClicked(x, y)（图像像素坐标）
    和 filesDropped(paths)（拖拽文件/文件夹路径列表）信号。
    """
    imageClicked = QtCore.pyqtSignal(int, int)
    filesDropped = QtCore.pyqtSignal(list)  # 拖拽文件信号
    imagePaint = QtCore.pyqtSignal(int, int, bool) #画笔

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setBackgroundRole(QtGui.QPalette.ColorRole.Base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)

        self._pixmap_orig: Optional[QtGui.QPixmap] = None
        self._scale = 1.0
        self._offset = QtCore.QPointF(0.0, 0.0)
        self._keep_aspect = True
        self._dragging = False
        self._drag_last_pos = QtCore.QPointF(0.0, 0.0)

        self.setAcceptDrops(True)  # 允许拖拽图像进display
        self.brush_mode = False #画笔开关，默认关

    # ===== 公共接口 =====
    def set_zoom_pixmap(self, pm: QtGui.QPixmap, reset: bool = True):
        self._pixmap_orig = pm
        if reset:
            self.fit_to_view()
        else:
            self.update()

    def fit_to_view(self):
        if not self._pixmap_orig or self._pixmap_orig.isNull():
            self._scale = 1.0
            self._offset = QtCore.QPointF(0.0, 0.0)
            self.update()
            return
        iw, ih = self._pixmap_orig.width(), self._pixmap_orig.height()
        lw, lh = max(1, self.width()), max(1, self.height())
        s = min(lw / iw, lh / ih) if self._keep_aspect else (lw / iw)
        self._scale = float(max(s, 1e-6))
        self._offset = QtCore.QPointF(0.0, 0.0)
        self.update()

    def has_image(self) -> bool:
        return bool(self._pixmap_orig and not self._pixmap_orig.isNull())

    # ===== 事件 =====
    def resizeEvent(self, e: QtGui.QResizeEvent):
        if self.has_image():
            iw, ih = self._pixmap_orig.width(), self._pixmap_orig.height()
            lw, lh = max(1, self.width()), max(1, self.height())
            fit_s = min(lw / iw, lh / ih) if self._keep_aspect else (lw / iw)
            if abs(self._scale - fit_s) < 1e-3:
                self.fit_to_view()
            else:
                self.update()
        super().resizeEvent(e)

    def wheelEvent(self, ev: QtGui.QWheelEvent):
        if not self.has_image():
            return super().wheelEvent(ev)
        angle = ev.angleDelta().y()
        if angle == 0:
            return
        factor = 1.25 if angle > 0 else 0.8
        new_scale = max(min(self._scale * factor, 100.0), 1e-4)
        before = self._map_widget_to_image(ev.position())
        self._scale = new_scale
        after = self._map_widget_to_image(ev.position())
        if before and after:
            self._offset -= (after - before)
        self.update()

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if not self.has_image():
            return super().mousePressEvent(ev)
        # if ev.button() == QtCore.Qt.MouseButton.LeftButton:
        #     img_pt = self._map_widget_to_image(ev.position())
        #     if img_pt:
        #         ix = int(round(img_pt.x()))
        #         iy = int(round(img_pt.y()))
        #         iw, ih = self._pixmap_orig.width(), self._pixmap_orig.height()
        #         ix = max(0, min(iw - 1, ix))
        #         iy = max(0, min(ih - 1, iy))
        #         self.imageClicked.emit(ix, iy)
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            img_pt = self._map_widget_to_image(ev.position())
            if img_pt:
                ix, iy = int(round(img_pt.x())), int(round(img_pt.y()))
                iw, ih = self._pixmap_orig.width(), self._pixmap_orig.height()
                ix = max(0, min(iw - 1, ix))
                iy = max(0, min(ih - 1, iy))
                # 画笔模式：左键绘制；右键/Alt作为擦除（见下）
                if getattr(self, "brush_mode", False):
                    is_erase = (ev.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier) or \
                               (ev.buttons() & QtCore.Qt.MouseButton.RightButton)
                    self.imagePaint.emit(ix, iy, bool(is_erase))
                    # 画笔模式下不进入平移拖拽
                    return
                # 非画笔：发点击选中
                self.imageClicked.emit(ix, iy)
            self._dragging = True
            self._drag_last_pos = ev.position()
        elif ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self._dragging = True
            self._drag_last_pos = ev.position()
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor if self._dragging
                       else QtCore.Qt.CursorShape.ArrowCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if getattr(self, "brush_mode", False) and (ev.buttons() & QtCore.Qt.MouseButton.LeftButton):
            img_pt = self._map_widget_to_image(ev.position())
            if img_pt and self._pixmap_orig:
                ix, iy = int(round(img_pt.x())), int(round(img_pt.y()))
                ix = max(0, min(self._pixmap_orig.width() - 1, ix))
                iy = max(0, min(self._pixmap_orig.height() - 1, iy))
                is_erase = (ev.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier) or \
                           (ev.buttons() & QtCore.Qt.MouseButton.RightButton)
                self.imagePaint.emit(ix, iy, bool(is_erase))
                return
        if self._dragging and self.has_image():
            delta_w = ev.position() - self._drag_last_pos
            if not delta_w.isNull():
                self._offset -= QtCore.QPointF(delta_w.x() / self._scale,
                                               delta_w.y() / self._scale)
                self._drag_last_pos = ev.position()
                self.update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if ev.button() in (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton):
            self._dragging = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(ev)

    def paintEvent(self, ev: QtGui.QPaintEvent):
        super().paintEvent(ev)
        if not self.has_image():
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        pm = self._pixmap_orig
        iw, ih = pm.width(), pm.height()
        lw, lh = self.width(), self.height()
        disp_w = iw * self._scale
        disp_h = ih * self._scale
        cx = lw * 0.5 - (self._offset.x() * self._scale)
        cy = lh * 0.5 - (self._offset.y() * self._scale)
        target = QtCore.QRectF(cx - disp_w * 0.5, cy - disp_h * 0.5, disp_w, disp_h)
        source = QtCore.QRectF(0.0, 0.0, float(iw), float(ih))
        painter.drawPixmap(target, pm, source)
        painter.end()

    # 拖拽
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        md = e.mimeData()
        if md.hasUrls() or md.hasText():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e: QtGui.QDragMoveEvent):
        md = e.mimeData()
        if md.hasUrls() or md.hasText():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent):
        paths = []
        md = e.mimeData()
        if md.hasUrls():
            for u in md.urls():
                p = u.toLocalFile()
                if p:
                    paths.append(p)
        elif md.hasText():
            for line in md.text().strip().splitlines():
                line = line.strip()
                if line:
                    paths.append(line)
        if paths:
            self.filesDropped.emit(paths)  # 触发信号，交给 Main 处理
            e.acceptProposedAction()
        else:
            e.ignore()

    # 坐标换算
    def _map_widget_to_image(self, pos: QtCore.QPointF) -> Optional[QtCore.QPointF]:
        if not self.has_image():
            return None
        pm = self._pixmap_orig
        iw, ih = pm.width(), pm.height()
        lw, lh = self.width(), self.height()
        disp_w = iw * self._scale
        disp_h = ih * self._scale
        cx = lw * 0.5 - (self._offset.x() * self._scale)
        cy = lh * 0.5 - (self._offset.y() * self._scale)
        left = cx - disp_w * 0.5
        top  = cy - disp_h * 0.5
        x = (pos.x() - left) / self._scale
        y = (pos.y() - top) / self._scale
        return QtCore.QPointF(x, y)
