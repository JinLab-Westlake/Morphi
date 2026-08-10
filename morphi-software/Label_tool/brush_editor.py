from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2
from PyQt6 import QtCore, QtGui
import copy


from Segmentation.seg_infer import draw_overlay

class InstanceBrushEditor(QtCore.QObject):
    """
    让所有实例都可“涂/擦”的画笔编辑器：
      - enable()/disable() 开关
      - 左键涂抹（把像素加入命中实例，并从其它实例里清掉同区域，保证互斥）
      - 右键擦除（从命中实例移除像素）
      - 未命中实例时左键起笔自动新建实例
      - '[' / ']' 调整半径；E 切换涂/擦
    """

    def __init__(self, main, draw_overlay_func, *, coords_mode: str = "orig"):
        """
        :param main: 你的 Main 窗口对象，要求包含：
            - last_img: np.ndarray (H,W,3) BGR
            - last_pred: dict or None. 需包含 "masks"(N,H,W或ROI)、"boxes"(N,4)、"scores"、"classes"
            - last_mode: str, 需为 "seg" 才可编辑
            - image_label: 有 _map_widget_to_image(posF) 方法的控件
            - append_log(str), set_process_info(str), show_image(img, reset_zoom=False)
            - _recompute_boxes_from_full_masks(masks) 方法（若无，会使用类内置回退实现）
            - slider_alpha(0..100) 和 seg_score_thr(阈值)（若没有会用默认值）
        :param draw_overlay_func: 你项目里的 draw_overlay 函数
        :param coords_mode: 传给 draw_overlay 的 coords 参数，通常 "orig"
        """
        super().__init__(main)
        self.main = main
        self.draw_overlay = draw_overlay_func
        self.coords_mode = coords_mode

        # 状态
        self.enabled = False
        self._brush_radius = 8
        self._erase = False
        self._is_down = False
        self._last_xy: Optional[Tuple[int, int]] = None
        self._edit_idx: Optional[int] = None

        # 进入画笔前的 last_pred 快照，用于 Clear 撤销
        self._orig_pred = None

        # 安装事件过滤
        self.main.image_label.installEventFilter(self)

        self._paint_dirty = False
        self._paint_timer = QtCore.QTimer(self)
        self._paint_timer.setInterval(33)  # ~30 FPS
        self._paint_timer.timeout.connect(self._flush_paint_preview)
        self._paint_timer.start()

    def _flush_paint_preview(self):
        if not self.enabled or not self._paint_dirty:
            return
        self._paint_dirty = False
        self._render_overlay_fast()

    # ====== 对外接口 ======
    def enable(self):
        if self.enabled:
            return
        if not self._is_seg_ready():
            self._log("[Brush] 请先完成一次 Seg 推理再开启画笔。")
            return

        lp = self.main.last_pred or {}
        m = lp.get("masks", None)
        self._orig_pred = {**lp, "masks": m.copy() if m is not None else None}


        self._ensure_all_masks_fullsize()  # 统一成 (N,H,W) 整图
        self.enabled = True
        self._set_cursor(True)
        self._log("[Brush] 画笔模式 ON：左键=涂抹，右键=擦除；[ / ] 调半径；E 切换涂/擦。")

    def disable(self):
        if not self.enabled:
            return
        self.enabled = False
        self._is_down = False
        self._last_xy = None
        self._edit_idx = None
        self._set_cursor(False)
        self._log("[Brush] 画笔模式 OFF")

    def set_radius(self, r: int):
        self._brush_radius = max(1, int(r))
        self._log(f"[Brush] 半径={self._brush_radius}")

    def toggle_erase(self):
        self._erase = not self._erase
        self._log("[Brush] 模式=" + ("擦除" if self._erase else "涂抹"))

    # ====== 事件处理 ======
    def eventFilter(self, obj, ev):
        if not self.enabled or obj is not self.main.image_label:
            return super().eventFilter(obj, ev)

        t = ev.type()
        if t == QtCore.QEvent.Type.MouseButtonPress:
            if ev.button() in (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton):
                if not self._is_seg_ready():  # 防抖
                    return True
                self._erase = (ev.button() == QtCore.Qt.MouseButton.RightButton)
                self._is_down = True
                xy = self._w2i(ev.position())
                if xy:
                    self._pick_or_create_instance(xy)
                    self._apply_brush(xy, first_dot=True)
                return True

        elif t == QtCore.QEvent.Type.MouseMove and self._is_down:
            xy = self._w2i(ev.position())
            if xy:
                self._apply_brush(xy)
            return True

        elif t == QtCore.QEvent.Type.MouseButtonRelease:
            if ev.button() in (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton):
                self._is_down = False
                self._last_xy = None
                self._post_brush_refresh()
                self.main.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)
                return True

        elif t == QtCore.QEvent.Type.KeyPress:
            key = ev.key()
            if key == QtCore.Qt.Key.Key_BracketLeft:   # '['
                self.set_radius(self._brush_radius - 1)
                return True
            if key == QtCore.Qt.Key.Key_BracketRight:  # ']'
                self.set_radius(self._brush_radius + 1)
                return True
            if key == QtCore.Qt.Key.Key_E:
                self.toggle_erase()
                return True

        return super().eventFilter(obj, ev)

    # ====== 内部：命中/编辑对象选取 ======
    def _pick_or_create_instance(self, xy: Tuple[int, int]):
        ix, iy = xy
        masks = self._masks()
        if masks is None or masks.size == 0:
            self._create_empty_instance()
            self._edit_idx = 0
            return
        hit = None
        for i in range(masks.shape[0]):
            if masks[i, iy, ix] > 0:
                hit = i; break
        if hit is None:
            # 新建一个空实例，从空白开始“涂”
            self._create_empty_instance()
            self._edit_idx = masks.shape[0]  # 新实例追加到末尾
        else:
            self._edit_idx = int(hit)

    def _create_empty_instance(self):
        H, W = self.main.last_img.shape[:2]
        masks = self._masks()
        boxes = self._boxes()
        if masks is None or masks.size == 0:
            masks_new = np.zeros((1, H, W), dtype=np.uint8)
            boxes_new = np.zeros((1, 4), dtype=np.float32)
            scores    = np.array([1.0], dtype=np.float32)
            classes   = np.array([0], dtype=np.int32)
        else:
            masks_new = np.concatenate([masks, np.zeros((1, H, W), dtype=np.uint8)], axis=0)
            boxes_new = np.concatenate([boxes if boxes is not None else np.zeros((masks.shape[0], 4), np.float32),
                                        np.zeros((1, 4), np.float32)], axis=0)
            sc = self._scores(); cl = self._classes()
            scores  = np.concatenate([np.asarray(sc, np.float32) if sc is not None else np.ones((masks.shape[0],), np.float32),
                                      np.array([1.0], np.float32)])
            classes = np.concatenate([np.asarray(cl, np.int32) if cl is not None else np.zeros((masks.shape[0],), np.int32),
                                      np.array([0], np.int32)])
        self._set("masks", masks_new)
        self._set("boxes", boxes_new)
        self._set("scores", scores)
        self._set("classes", classes)
        self._set("masks_disp", masks_new.copy())
        self._set("disp_size", (H, W))

    # ====== 内部：画笔应用 ======
    def _apply_brush(self, xy: Tuple[int, int], first_dot=False):
        if self._edit_idx is None:
            return
        ix, iy = xy
        masks = self._masks()  # (N,H,W)
        if masks is None or masks.size == 0:
            return
        N = masks.shape[0]
        r = int(max(1, self._brush_radius))

        if self._last_xy is None or first_dot:
            if self._erase:
                cv2.circle(masks[self._edit_idx], (ix, iy), r, 0, -1)
            else:
                for j in range(N):
                    if j == self._edit_idx: continue
                    cv2.circle(masks[j], (ix, iy), r, 0, -1)
                cv2.circle(masks[self._edit_idx], (ix, iy), r, 255, -1)
            self._last_xy = (ix, iy)
        else:
            x0, y0 = self._last_xy
            if self._erase:
                cv2.line(masks[self._edit_idx], (x0, y0), (ix, iy), 0, thickness=2 * r)
            else:
                for j in range(N):
                    if j == self._edit_idx: continue
                    cv2.line(masks[j], (x0, y0), (ix, iy), 0, thickness=2 * r)
                cv2.line(masks[self._edit_idx], (x0, y0), (ix, iy), 255, thickness=2 * r)
            self._last_xy = (ix, iy)

        # # 快速预览
        # self._set("masks_disp", masks.copy())
        self._set("masks_disp", masks)
        # 不要立刻渲染
        self._paint_dirty = True


    def _post_brush_refresh(self):
        masks = self._masks()
        if masks is None:
            return

        # 1) 删空实例
        keep = [i for i in range(masks.shape[0]) if (masks[i] > 0).any()]
        if len(keep) != masks.shape[0]:
            masks = masks[keep]
            self._set("masks", masks)
            for k in ("scores", "classes"):
                arr = self._get(k)
                if arr is not None:
                    self._set(k, np.asarray(arr)[keep])
            if self._edit_idx is not None:
                try:
                    self._edit_idx = keep.index(self._edit_idx)
                except ValueError:
                    self._edit_idx = None

        # 2) 填洞：只填当前实例 / 或全部实例
        if masks.shape[0] > 0:
            if self._edit_idx is not None and 0 <= self._edit_idx < masks.shape[0]:
                masks[self._edit_idx] = self._fill_holes_uint8(masks[self._edit_idx])
            else:
                # 如果你想每次都全局填洞，改用这段：
                # for i in range(masks.shape[0]):
                #     masks[i] = self._fill_holes_uint8(masks[i])
                pass

        # 3) boxes（你不需要就清空，需要就重算）
        self._set("boxes", np.zeros((masks.shape[0], 4), dtype=np.float32))

        # 4) 同步显示用 masks
        self._set("masks", masks)
        # self._set("masks_disp", masks.copy())
        self._set("masks_disp", masks)

        # 5) 重画
        self._render_overlay_fast()
        n_now = masks.shape[0]
        self.main.set_process_info(f"Process Info (Seg): {n_now} objects")

    # ====== 按钮槽函数 ======
    @QtCore.pyqtSlot()
    def on_commit_clicked(self):
        """把当前编辑过的 mask 保存到 last_pred，并刷新 overlay。"""
        masks = self._masks()
        if masks is None or masks.size == 0:
            return

        # 写回最终 masks（内部已经是 fullsize (N,H,W) )
        self.main.last_pred["masks"] = masks.copy()
        self.main.last_pred["masks_disp"] = masks.copy()

        # 不再使用 box（你整个 pipeline 已经 mask-only）
        N = masks.shape[0]
        self.main.last_pred["boxes"] = np.zeros((N, 4), dtype=np.float32)

        # 更新 overlay
        self.main._refresh_overlay_from_masks()

        # 退出当前实例编辑
        self._edit_idx = None
        self.main.append_log("[Brush] Commit accepted.")

    def on_clear_clicked(self):
        """只清当前实例（你特别要求）"""
        if self._edit_idx is None:
            return

        masks = self._masks()
        if masks is None or masks.size == 0:
            return

        # 清当前
        masks[self._edit_idx] = 0

        # 同步写回
        self.main.last_pred["masks"] = masks.copy()
        self.main.last_pred["masks_disp"] = masks.copy()
        self.main.last_pred["boxes"] = np.zeros((masks.shape[0], 4), dtype=np.float32)

        # 刷新 overlay
        self.main._refresh_overlay_from_masks()

        self.main.append_log(f"[Brush] Instance {self._edit_idx} cleared.")
        self._edit_idx = None


    # ====== 内部：绘制 & 工具 ======
    def _render_overlay_fast(self):
        alp = self._alpha()
        thr = self._score_thr()
        overlay = self.draw_overlay(
            self.main.last_img,
            self._get("frame", None),
            None, self._scores(), self._classes(), self._masks(),
            score_thr=thr, alpha=alp, draw_box=False, coords=self.coords_mode
        )
        self.main.show_image(overlay, reset_zoom=False)

    def _ensure_all_masks_fullsize(self):
        """
        把 masks 统一成 (N,H,W) 整图：
        - 对 ROI 小 mask：用 boxes 贴回整图
        - 避免 (a > 0) 这种过度二值化导致“整个 bbox 变成实心块”
        """
        H, W = self.main.last_img.shape[:2]
        masks = self._masks()
        boxes = self._boxes()

        if masks is None:
            self._set("masks", np.zeros((0, H, W), dtype=np.uint8))
            self._set("masks_disp", np.zeros((0, H, W), dtype=np.uint8))
            self._set("disp_size", (H, W))
            return

        # 统一成 list
        if isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                masks = masks[None, ...]
            lst = [np.squeeze(masks[i, ...]) for i in range(masks.shape[0])]
        else:
            lst = [np.squeeze(np.asarray(m)) for m in masks]

        full_list = []
        for i, a in enumerate(lst):
            a = np.asarray(a, dtype=np.float32)  # 保留为 float，用 soft mask

            # 情况 0：已经是整图大小，直接用 soft mask
            if a.shape[:2] == (H, W):
                full = a

            # 情况 1：ROI mask + box，贴回整图
            elif boxes is not None and i < len(boxes):
                full = np.zeros((H, W), dtype=np.float32)
                x1, y1, x2, y2 = [int(round(v)) for v in boxes[i]]
                x1 = max(0, min(W - 1, x1));
                x2 = max(0, min(W, x2))
                y1 = max(0, min(H - 1, y1));
                y2 = max(0, min(H, y2))
                bw, bh = max(0, x2 - x1), max(0, y2 - y1)
                if bw > 0 and bh > 0:
                    # 这里只是把 28x28 映射到 bbox 大小，不做二值化
                    a_roi = cv2.resize(a, (bw, bh), interpolation=cv2.INTER_NEAREST)
                    full[y1:y2, x1:x2] = a_roi
                else:
                    # bbox 异常就丢弃
                    full = np.zeros((H, W), dtype=np.float32)

            # 情况 2：既不是整图又没 box，只能丢弃或居中贴
            else:
                full = np.zeros((H, W), dtype=np.float32)

            full_list.append(full)

        # 先堆成 float 的 (N,H,W) soft mask
        full_float = np.stack(full_list, axis=0) if full_list else np.zeros((0, H, W), dtype=np.float32)

        # 再统一做一次合理的阈值，比如 0.5
        thr = 0.5
        masks_full = (full_float >= thr).astype(np.uint8) * 255

        self._set("masks", masks_full)
        self._set("masks_disp", masks_full.copy())
        self._set("disp_size", (H, W))
        # # 根据二值整图重算 bbox（给后面用）
        # self._set("boxes", self._recompute_boxes_from_full_masks(masks_full))


    def _recompute_boxes_from_full_masks(self, masks_full: np.ndarray):
        # 优先使用 main 的实现
        if hasattr(self.main, "_recompute_boxes_from_full_masks"):
            try:
                return self.main._recompute_boxes_from_full_masks(masks_full)
            except Exception:
                pass
        # 回退：自己算
        if masks_full is None or masks_full.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        H, W = masks_full.shape[1:]
        out = []
        for i in range(masks_full.shape[0]):
            ys, xs = np.where(masks_full[i] > 0)
            if ys.size == 0:
                out.append([0, 0, 0, 0]); continue
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
            x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W, x2));     y2 = max(0, min(H, y2))
            out.append([x1, y1, x2, y2])
        return np.asarray(out, dtype=np.float32)

    # ====== 小工具 ======
    def _is_seg_ready(self) -> bool:
        return (self.main.last_mode == "seg" and self.main.last_pred is not None and self.main.last_img is not None)

    def _w2i(self, posF) -> Optional[Tuple[int, int]]:
        pt = self.main.image_label._map_widget_to_image(posF)
        if not pt: return None
        ix, iy = int(round(pt.x())), int(round(pt.y()))
        H, W = self.main.last_img.shape[:2]
        if 0 <= ix < W and 0 <= iy < H:
            return (ix, iy)
        return None

    def _alpha(self) -> float:
        try:
            return float(getattr(self.main, "slider_alpha").value()) / 100.0
        except Exception:
            return 0.5

    def _score_thr(self) -> float:
        try:
            return float(getattr(self.main, "seg_score_thr").value())
        except Exception:
            return 0.0

    def _log(self, s: str):
        try:
            self.main.append_log(s)
        except Exception:
            print(s)

    def _set_cursor(self, on: bool):
        try:
            self.main.image_label.setCursor(QtCore.Qt.CursorShape.CrossCursor if on
                                            else QtCore.Qt.CursorShape.ArrowCursor)
        except Exception:
            pass

    # last_pred 便捷访问
    def _get(self, key, default=None):
        lp = self.main.last_pred or {}
        return lp.get(key, default)

    def _set(self, key, value):
        if self.main.last_pred is None:
            self.main.last_pred = {}
        self.main.last_pred[key] = value

    def _masks(self) -> Optional[np.ndarray]:
        m = self._get("masks", None)
        if m is None:
            return None
        if isinstance(m, list):
            try:
                m = np.stack([np.asarray(x) for x in m], axis=0)
            except Exception:
                return None
        if isinstance(m, np.ndarray):
            if m.ndim == 2:
                m = m[None, ...]
            return m  # 不再二值化/不再拷贝
        return None

    def _boxes(self) -> Optional[np.ndarray]:
        b = self._get("boxes", None)
        if b is None: return None
        return np.asarray(b, dtype=np.float32)

    def _scores(self) -> Optional[np.ndarray]:
        s = self._get("scores", None)
        return None if s is None else np.asarray(s, dtype=np.float32)

    def _classes(self) -> Optional[np.ndarray]:
        c = self._get("classes", None)
        return None if c is None else np.asarray(c, dtype=np.int32)

    def _fill_holes_uint8(self, m: np.ndarray) -> np.ndarray:
        """
        填充单个实例 mask 内部孔洞。
        输入 m: (H,W) uint8, 0/255
        输出: (H,W) uint8, 0/255
        """
        if m is None or m.size == 0:
            return m
        # 确保是 0/255
        m = (m > 0).astype(np.uint8) * 255

        h, w = m.shape[:2]
        # floodFill 需要 1 像素边界，避免外部背景被误填
        pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        # 从外部背景 flood fill
        mask_ff = np.zeros((h + 2 + 2, w + 2 + 2), np.uint8)  # 注意 floodFill 的 mask 要比图大2
        cv2.floodFill(pad, mask_ff, (0, 0), 255)

        # pad 里被填成 255 的是“外部背景”
        outside = pad[1:-1, 1:-1]
        holes = cv2.bitwise_not(outside)  # 反过来得到孔洞
        filled = cv2.bitwise_or(m, holes)  # 原 mask + 孔洞

        return filled


