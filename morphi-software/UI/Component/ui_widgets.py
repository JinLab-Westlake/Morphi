from typing import Optional
from PyQt6 import QtWidgets, QtGui, QtCore
import numpy as np

class HistogramLevelsWidget(QtWidgets.QWidget):
    """
    直方图 + 三手柄（黑、γ、白），范围 0..255。
    公共API：
      - set_histogram(hist) / set_hist(hist)
      - set_levels(black, gamma, white, emit_signal=True)
      - levels() -> (b, g, w)
      - 信号：levelsChanged(int black, float gamma, int white)
    """
    levelsChanged = QtCore.pyqtSignal(int, float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # ==== 数据 ====
        self._hist = np.zeros(256, dtype=np.float32)
        self._hist_norm = None

        # ==== 等级参数 ====
        self._b = 0
        self._w = 255
        self._g = 1.0

        # ==== 交互状态 ====
        self._drag = None            # "b" | "w" | "g" | None
        self._handle_radius = 12

        # ==== 绘制边距（统一）====
        self._pad_left = 12
        self._pad_right = 12
        self._pad_top = 8
        self._pad_bottom = 40   # 给底部手柄/文字留足空间

        # 命中区域缓存
        self._hit_b = QtCore.QRect()
        self._hit_w = QtCore.QRect()
        self._hit_g = QtCore.QRect()

        self._pending_update = False
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(16)  # ~60 FPS；想更稳可设 33
        self._debounce.timeout.connect(self._flush_update)

        # 减少背景擦除抖动
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)

        self._suppress_emit = False

    def _schedule_update(self):
        if not self._pending_update:
            self._pending_update = True
            self._debounce.start()

    def _flush_update(self):
        self._pending_update = False
        self.update()

    # ---------- 工具：统一的绘图区域 ----------
    def _plot_rect(self) -> QtCore.QRect:
        r = self.rect()
        return r.adjusted(self._pad_left, self._pad_top, -self._pad_right, -self._pad_bottom)

    def clear_histogram(self):
        self._hist = np.zeros(256, dtype=np.float32)
        self._hist_norm = None
        self._schedule_update()

    def set_histogram(self, hist: Optional[np.ndarray]):
        if hist is None or len(hist) != 256:
            self._hist = np.zeros(256, dtype=np.float32)
            self._hist_norm = None
        else:
            h = np.asarray(hist, dtype=np.float32)
            h = np.maximum(h, 0)
            m = float(h.max())
            # 避免每帧做除法：保存归一化结果
            self._hist = h
            self._hist_norm = (h / m) if m > 0 else h
        self._schedule_update()

    def clear_histogram(self):
        self._hist = np.zeros(256, dtype=np.float32)
        self._hist_norm = None
        self._schedule_update()

    def set_histogram(self, hist: Optional[np.ndarray]):
        if hist is None or len(hist) != 256:
            self._hist = np.zeros(256, dtype=np.float32)
            self._hist_norm = None
        else:
            h = np.asarray(hist, dtype=np.float32)
            h = np.maximum(h, 0)
            m = float(h.max())
            # 避免每帧做除法：保存归一化结果
            self._hist = h
            self._hist_norm = (h / m) if m > 0 else h
        self._schedule_update()

    # 兼容旧调用
    def set_hist(self, hist: Optional[np.ndarray]):
        self.set_histogram(hist)

    def set_levels(self, black: int, gamma: float, white: int, emit_signal=True):
        black = int(max(0, min(255, black)))
        white = int(max(0, min(255, white)))
        if white <= black:
            white = black + 1
        gamma = float(max(0.05, min(5.0, gamma)))

        self._b, self._w, self._g = black, white, gamma
        self._schedule_update()
        if emit_signal and not self._suppress_emit:
            self.levelsChanged.emit(self._b, self._g, self._w)

    def levels(self):
        return self._b, self._g, self._w

    # ---------- 绘制 ----------
    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        full = self.rect()
        plot = self._plot_rect()

        # 背景
        p.fillRect(full, QtGui.QColor("#1e1f22"))

        # plot边框
        p.setPen(QtGui.QPen(QtGui.QColor(230, 230, 230, 200), 1))
        p.drawRect(plot)

        # 直方图
        if self._hist_norm is not None and plot.width() > 0:
            poly = QtGui.QPolygonF()
            left, top = plot.left(), plot.top()
            w, h = plot.width(), plot.height()

            poly.append(QtCore.QPointF(left, top + h))
            for i in range(256):
                x = left + (i / 255.0) * w
                y = top + h * (1.0 - float(self._hist_norm[i]))
                poly.append(QtCore.QPointF(x, y))
            poly.append(QtCore.QPointF(left + w, top + h))

            p.setBrush(QtGui.QColor(76, 139, 245, 150))
            p.setPen(QtGui.QPen(QtGui.QColor(76, 139, 245, 220), 2))
            p.drawPolygon(poly)

            # —— 彩色条（Jet风格）——
            ramp_h = 12
            ramp_rect = QtCore.QRect(plot.left(), plot.bottom() + 4, plot.width(), ramp_h)
            if ramp_rect.width() > 0:
                grad = QtGui.QLinearGradient(ramp_rect.left(), ramp_rect.center().y(),
                                             ramp_rect.right(), ramp_rect.center().y())
                # Jet-ish 颜色节点：蓝→青→黄→红
                grad.setColorAt(0.00, QtGui.QColor(0, 0, 131))
                grad.setColorAt(0.25, QtGui.QColor(0, 60, 255))
                grad.setColorAt(0.50, QtGui.QColor(170, 255, 195))
                grad.setColorAt(0.75, QtGui.QColor(255, 220, 0))
                grad.setColorAt(1.00, QtGui.QColor(128, 0, 0))
                p.fillRect(ramp_rect, QtGui.QBrush(grad))

        # 手柄位置换算
        def x_from_value(v):
            return plot.left() + (v / 255.0) * max(1, plot.width())

        xb = int(x_from_value(self._b))
        xw = int(x_from_value(self._w))
        xg = int(x_from_value(self._gamma_to_x()))

        # 三角手柄（黑、白）
        tri_w, tri_h = 10, 8
        y_tri = plot.bottom() + 12

        def draw_triangle(cx, color):
            path = QtGui.QPainterPath()
            path.moveTo(cx - tri_w, y_tri)
            path.lineTo(cx + tri_w, y_tri)
            path.lineTo(cx, y_tri - tri_h)
            path.closeSubpath()
            p.setBrush(QtGui.QBrush(color))
            p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 1))
            p.drawPath(path)

        draw_triangle(xb, QtGui.QColor("#cfd2d8"))
        draw_triangle(xw, QtGui.QColor("#cfd2d8"))

        # γ 水滴
        drop_r = 9.0
        drop_c = QtCore.QPointF(float(xg), float(plot.bottom() + 10))
        path = QtGui.QPainterPath()
        path.addEllipse(drop_c, drop_r, drop_r)
        p.setBrush(QtGui.QColor("#8ab4ff"))
        p.setPen(QtGui.QPen(QtGui.QColor(30, 30, 30), 1))
        p.drawPath(path)

        # 命中区域
        self._hit_b = QtCore.QRect(int(xb) - 14, int(y_tri) - 14, 28, 18)
        self._hit_w = QtCore.QRect(int(xw) - 14, int(y_tri) - 14, 28, 18)
        self._hit_g = QtCore.QRect(int(drop_c.x()) - 16, int(drop_c.y()) - 16, 32, 32)

        # 文本
        info = f"B:{self._b}  G:{self._g:.2f}  W:{self._w}"
        p.setPen(QtGui.QColor(220, 220, 220))
        p.drawText(full.adjusted(8, full.height() - 22, -8, -4),
                   QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                   info)

    # ---------- 鼠标交互 ----------
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        pos = e.position().toPoint()
        if self._hit_b.contains(pos):
            self._drag = "b"
        elif self._hit_w.contains(pos):
            self._drag = "w"
        elif self._hit_g.contains(pos):
            self._drag = "g"
        else:
            self._drag = self._pick_nearest_handle(pos.x())
        self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if not self._drag:
            return super().mouseMoveEvent(e)
        plot = self._plot_rect()

        x = int(e.position().x())
        x = max(plot.left(), min(plot.right(), x))
        v = int(round((x - plot.left()) * 255.0 / max(1, plot.width())))

        if self._drag == "b":
            self.set_levels(min(v, self._w - 1), self._g, self._w)
        elif self._drag == "w":
            self.set_levels(self._b, self._g, max(v, self._b + 1))
        elif self._drag == "g":
            rel = (v - self._b) / max(1.0, (self._w - self._b))
            rel = float(max(0.01, min(0.99, rel)))
            gamma = self._rel_to_gamma(rel)
            self.set_levels(self._b, gamma, self._w)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self._drag = None
        self.unsetCursor()
        return super().mouseReleaseEvent(e)

    def leaveEvent(self, e: QtCore.QEvent):
        if not self._drag:
            self.unsetCursor()
        return super().leaveEvent(e)

    def _pick_nearest_handle(self, x):
        plot = self._plot_rect()
        def pos_from_val(v):
            return plot.left() + (v / 255.0) * max(1, plot.width())
        xb = pos_from_val(self._b)
        xw = pos_from_val(self._w)
        xg = pos_from_val(self._gamma_to_x())
        d = [(abs(x - xb), "b"), (abs(x - xg), "g"), (abs(x - xw), "w")]
        d.sort()
        return d[0][1]

    # ---------- γ 映射 ----------
    def _gamma_to_x(self):
        g = float(self._g)
        rel = g**0.5 / (g**0.5 + 1.0)  # 单调映射，rel≈0.5时γ≈1
        x = int(round(self._b + rel * (self._w - self._b)))
        return max(0, min(255, x))

    @staticmethod
    def _rel_to_gamma(rel: float):
        rel = max(1e-3, min(1-1e-3, rel))
        t = rel / (1.0 - rel)
        g = max(0.05, min(5.0, t**2))
        return g


# if __name__ == "__main__":
#     import sys
#
#     app = QtWidgets.QApplication(sys.argv)
#     w = HistogramLevelsWidget()
#     w.resize(400, 150)
#     w.show()
#
#     # 测试用的随机直方图（让蓝色区域有高度）
#     w.set_histogram(np.random.randint(0, 100, 256))
#
#     sys.exit(app.exec())
