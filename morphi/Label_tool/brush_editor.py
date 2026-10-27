from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2
from PyQt6 import QtCore, QtGui
import copy


from Segmentation.seg_infer import draw_overlay

class InstanceBrushEditor(QtCore.QObject):
    """
    A brush editor that makes all instances "paint/erase" :
        - enable ()/disable() switch
        - Left-click smudge (adds pixels to the hit instance and removes the same area from other instances, ensuring mutual exclusion)
        - Right click erase (remove pixels from hit instance)
        - When an instance is missed, left click and start to automatically create an instance
        - '['/']' adjusts the radius; E Switch paint/wipe
    """

    def __init__(self, main, draw_overlay_func, *, coords_mode: str = "orig"):
        """
        :param main:
            - last_img: np.ndarray (H,W,3) BGR
            -last_pred: dict or None. Contains "masks"(N,H,W or ROI), "boxes"(N,4), "scores", "classes"
            -last_mode: str, must be "seg" to be editable
            -image_label: Controls that have the _map_widget_to_image(posF) method
            - append_log(str), set_process_info(str), show_image(img, reset_zoom=False)
            - _recompute_boxes_from_full_masks(masks) method
            - slider_alpha(0.. 100) and seg_score_thr(threshold)
        :param draw_overlay_func
        :param coords_mode: coords parameter to pass to draw_overlay, usually "orig"
        """
        super().__init__(main)
        self.main = main
        self.draw_overlay = draw_overlay_func
        self.coords_mode = coords_mode

        # state
        self.enabled = False
        self._brush_radius = 8
        self._erase = False
        self._is_down = False
        self._last_xy: Optional[Tuple[int, int]] = None
        self._edit_idx: Optional[int] = None

        # last_pred snapshot before entering the brush, used for Clear undo
        self._orig_pred = None

        # Install event filtering
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

    # ====== External interface ======
    def enable(self):
        if self.enabled:
            return
        if not self._is_seg_ready():
            self._log("[Brush] Please complete a Seg inference before starting the brush.")
            return

        lp = self.main.last_pred or {}
        m = lp.get("masks", None)
        self._orig_pred = {**lp, "masks": m.copy() if m is not None else None}


        self._ensure_all_masks_fullsize()  # 统一成 (N,H,W) 整图
        self.enabled = True
        self._set_cursor(True)
        self._log("[Brush] Brush mode ON: Left button = smear, right button = erase; [/] tuning radius; E Switch paint/wipe.")

    def disable(self):
        if not self.enabled:
            return
        self.enabled = False
        self._is_down = False
        self._last_xy = None
        self._edit_idx = None
        self._set_cursor(False)
        self._log("[Brush] Brush mode OFF")

    def set_radius(self, r: int):
        self._brush_radius = max(1, int(r))
        self._log(f"[Brush] radius={self._brush_radius}")

    def toggle_erase(self):
        self._erase = not self._erase
        self._log("[Brush] Pattern=" + ("erasure" if self._erase else "rub"))

    # ====== Event handling ======
    def eventFilter(self, obj, ev):
        if not self.enabled or obj is not self.main.image_label:
            return super().eventFilter(obj, ev)

        t = ev.type()
        if t == QtCore.QEvent.Type.MouseButtonPress:
            if ev.button() in (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton):
                if not self._is_seg_ready():  # Anti shaking
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

    # ====== Internal: Hit/edit object selection ======
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
            # Create an empty instance and "paint" from the blank
            self._create_empty_instance()
            self._edit_idx = masks.shape[0]  # The new instance is appended to the end
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

    # ====== Inside: Brush application ======
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

        # Quick Preview
        self._set("masks_disp", masks)
        # Don't render right away
        self._paint_dirty = True


    def _post_brush_refresh(self):
        masks = self._masks()
        if masks is None:
            return

        # Delete the empty instance
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

        # Fill holes: Fill only the current instance/or all instances
        if masks.shape[0] > 0:
            if self._edit_idx is not None and 0 <= self._edit_idx < masks.shape[0]:
                masks[self._edit_idx] = self._fill_holes_uint8(masks[self._edit_idx])
            else:
                pass

        # boxes
        self._set("boxes", np.zeros((masks.shape[0], 4), dtype=np.float32))

        # masks
        self._set("masks", masks)
        # self._set("masks_disp", masks.copy())
        self._set("masks_disp", masks)

        # redraw
        self._render_overlay_fast()
        n_now = masks.shape[0]
        self.main.set_process_info(f"Process Info (Seg): {n_now} objects")

    # ====== Button slot function ======
    @QtCore.pyqtSlot()
    def on_commit_clicked(self):
        """Save the currently edited mask to last_pred and refresh the overlay."""
        masks = self._masks()
        if masks is None or masks.size == 0:
            return

        # Write back to the final masks
        self.main.last_pred["masks"] = masks.copy()
        self.main.last_pred["masks_disp"] = masks.copy()

        # No more box
        N = masks.shape[0]
        self.main.last_pred["boxes"] = np.zeros((N, 4), dtype=np.float32)

        # update overlay
        self.main._refresh_overlay_from_masks()

        # Quit editing the current instance
        self._edit_idx = None
        self.main.append_log("[Brush] Commit accepted.")

    def on_clear_clicked(self):
        """Clears only the current instance"""
        if self._edit_idx is None:
            return

        masks = self._masks()
        if masks is None or masks.size == 0:
            return

        # Clear current
        masks[self._edit_idx] = 0

        # Synchronous write back
        self.main.last_pred["masks"] = masks.copy()
        self.main.last_pred["masks_disp"] = masks.copy()
        self.main.last_pred["boxes"] = np.zeros((masks.shape[0], 4), dtype=np.float32)

        # update overlay
        self.main._refresh_overlay_from_masks()

        self.main.append_log(f"[Brush] Instance {self._edit_idx} cleared.")
        self._edit_idx = None


    # ====== Inside: Draws & Tools ======
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
        Unify the masks into an (N,H,W) graph:
        - Small mask on ROI: use boxes to paste back into the whole image
        - Avoid (a > 0) such overbinarization that "the whole bbox becomes a solid block"
        """
        H, W = self.main.last_img.shape[:2]
        masks = self._masks()
        boxes = self._boxes()

        if masks is None:
            self._set("masks", np.zeros((0, H, W), dtype=np.uint8))
            self._set("masks_disp", np.zeros((0, H, W), dtype=np.uint8))
            self._set("disp_size", (H, W))
            return

        # Unified as a list
        if isinstance(masks, np.ndarray):
            if masks.ndim == 2:
                masks = masks[None, ...]
            lst = [np.squeeze(masks[i, ...]) for i in range(masks.shape[0])]
        else:
            lst = [np.squeeze(np.asarray(m)) for m in masks]

        full_list = []
        for i, a in enumerate(lst):
            a = np.asarray(a, dtype=np.float32)  # Let's keep it as float and use soft mask

            # It's the full size of the image, just use soft mask
            if a.shape[:2] == (H, W):
                full = a

            # ROI mask + box, paste back to the whole image
            elif boxes is not None and i < len(boxes):
                full = np.zeros((H, W), dtype=np.float32)
                x1, y1, x2, y2 = [int(round(v)) for v in boxes[i]]
                x1 = max(0, min(W - 1, x1));
                x2 = max(0, min(W, x2))
                y1 = max(0, min(H - 1, y1));
                y2 = max(0, min(H, y2))
                bw, bh = max(0, x2 - x1), max(0, y2 - y1)
                if bw > 0 and bh > 0:
                    # Just map 28x28 to the bbox size, without binarization
                    a_roi = cv2.resize(a, (bw, bh), interpolation=cv2.INTER_NEAREST)
                    full[y1:y2, x1:x2] = a_roi
                else:
                    # bbox exceptions are discarded
                    full = np.zeros((H, W), dtype=np.float32)

            # Neither the whole picture nor the box, can only be discarded or centered
            else:
                full = np.zeros((H, W), dtype=np.float32)

            full_list.append(full)

        # Start with a float (N,H,W) soft mask
        full_float = np.stack(full_list, axis=0) if full_list else np.zeros((0, H, W), dtype=np.float32)

        # Again, set a reasonable threshold, such as 0.5
        thr = 0.5
        masks_full = (full_float >= thr).astype(np.uint8) * 255

        self._set("masks", masks_full)
        self._set("masks_disp", masks_full.copy())
        self._set("disp_size", (H, W))



    def _recompute_boxes_from_full_masks(self, masks_full: np.ndarray):
        # Use main's implementation first
        if hasattr(self.main, "_recompute_boxes_from_full_masks"):
            try:
                return self.main._recompute_boxes_from_full_masks(masks_full)
            except Exception:
                pass
        # Backoff
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

    # ====== tools ======
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

    # last_pred for easy access
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
            return m  # No more binarization/no more copying
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
        Fill holes inside a single instance mask.
        Enter m: (H,W) uint8, 0/255
        Output: (H,W) uint8, 0/255
        """
        if m is None or m.size == 0:
            return m
        # make sure 0/255
        m = (m > 0).astype(np.uint8) * 255

        h, w = m.shape[:2]
        # floodFill requires a 1-pixel boundary to avoid the outside background being filled in incorrectly
        pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        # flood fill from the external background
        mask_ff = np.zeros((h + 2 + 2, w + 2 + 2), np.uint8)
        cv2.floodFill(pad, mask_ff, (0, 0), 255)

        # The pad is filled with 255
        outside = pad[1:-1, 1:-1]
        holes = cv2.bitwise_not(outside)
        filled = cv2.bitwise_or(m, holes)  # Original mask + holes

        return filled


