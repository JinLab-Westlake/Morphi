# app.py
from typing import Optional
import math
from PyQt6 import QtWidgets, QtGui, QtCore
import os, json, cv2
from Segmentation.seg_infer import Segmentor, draw_overlay, build_merged_mask
from Segmentation.seg_sur import SurvivalSegmentor
from Tool.det_tensors_norm import _normalize_det_tensors
import numpy as np
from Tool.NMS import _nms_keep_indices, _filter_by_min
from UI.Layout.control import _build_left_panel
from UI.Layout.display import _build_center_panel,install_autoresize
from Label_tool.label_editor import LabelEditorDialog
from UI.Layout.manubar import _build_menubar
from UI.Sup_procedure.edge_cluster import EdgeClusterWorker
from Fluo_proccess.Proj.sobel_proj import focus_stack_sobel
from UI.Information_win.info_win import InfoWindow
import glob, tempfile
from Label_tool.brush_editor import InstanceBrushEditor
from Fluo_proccess.Canny.canny_edge_detec import EdgeClusterParams




class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # ---- running mode ----
        self.seg_seg = None   # Segmentor of instance segmentation model
        self.seg_surv = None  # Segmentor of survival segmentation model
        self.last_img = None
        self.last_pred = None
        self.last_mode = None  # "seg" or "surv"
        self.info_window = None
        self._log_buffer = []

        self.seg_score_thr = 0.8
        self.seg_nms_iou = 0.005
        self.seg_min_area = 0
        self.seg_min_side = 0
        self.seg_top_k = 100
        self.seg_output_dir = None

        # ---- Persistent Settings ----
        self.settings = QtCore.QSettings("CellMonitor", "CellSegUI")
        self.models_seg = self._load_models_dict("models_seg_json")  # {name: path}
        self.models_surv = self._load_models_dict("models_surv_json")


        if not self.models_seg:
            self.models_seg = {
                "Organoids": "./Segmentation/Weight/organoids_seg.onnx",
                "General segmentation": "./Segmentation/Weight/model_segment_general.onnx"
            }
        if not self.models_surv:
            self.models_surv = {
                "Survival analysis": "./Segmentation/Weight/swin_unet.onnx",
            }

        # ============ Three-column layout（QSplitter） ============
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Build the left, middle and right parts respectively and return the QWidget
        left   = _build_left_panel(self)
        center = _build_center_panel(self)
        install_autoresize(self)
        # right  = _build_right_panel(self)

        splitter.addWidget(left)
        splitter.addWidget(center)
        # splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 800, 300])


        self.setCentralWidget(splitter)
        self.setWindowTitle("Morphi")

        # add menubar
        _build_menubar(self)

        # ---- 结果导出 & 标签编辑 菜单 ----
        menu = self.menuBar().addMenu("Annotation")

        # --- 仅导出 masks(PNG) ---
        act_export_masks_png = QtGui.QAction("Export mask (PNG)…", self)
        act_export_masks_png.triggered.connect(self._save_masks_png_only)
        menu.addAction(act_export_masks_png)

        # --- 仅导出 masks(JSON) ---
        act_export_masks_json = QtGui.QAction("Export mask (JSON)…", self)
        act_export_masks_json.triggered.connect(self._save_masks_json_only)
        menu.addAction(act_export_masks_json)

        act_edit = QtGui.QAction("Edit Labels…", self)
        act_edit.triggered.connect(self._open_label_editor)
        menu.addAction(act_edit)

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)


        self.edit_mode = False
        self._deleted_ids = set()
        self._selected_inst = None  # 当前高亮选中的实例索引
        self.chk_edit_mode.toggled.connect(self._on_toggle_edit_mode)

        # click, 只选中并高亮；删除靠键盘
        self.image_label.imageClicked.connect(self._on_image_click_select)

        self.brush_editor = InstanceBrushEditor(self, draw_overlay)

        if hasattr(self, "chk_brush"):
            self.chk_brush.toggled.connect(lambda on: self.brush_editor.enable() if on else self.brush_editor.disable())

        # 这里加：给 Commit / Clear 按钮绑槽函数
        if hasattr(self, "btn_brush_commit"):
            self.btn_brush_commit.clicked.connect(self.brush_editor.on_commit_clicked)

        if hasattr(self, "btn_brush_clear"):
            self.btn_brush_clear.clicked.connect(self.brush_editor.on_clear_clicked)


        # ========= It does not enter the input state initially =========
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        # need to be limited component
        self._click_focus_widgets = [
            getattr(self, "spin_score", None),
            getattr(self, "spin_nms", None),
            getattr(self, "spin_min_area", None),
            getattr(self, "spin_min_side", None),
            getattr(self, "edit_outdir", None),
            getattr(self, "combo_seg", None),
            getattr(self, "combo_surv", None),
        ]
        # filter None
        self._click_focus_widgets = [w for w in self._click_focus_widgets if w is not None]

        # === 调色参数记录字典 ===
        # 用于保存滑块的实时值（只记录，不改变显示）
        self._adj = {
            "bri": 0.0,      # 亮度 [-100,100]
            "con": 1.0,      # 对比度倍率 [0.1,3.0]
            "gamma": 1.0,    # Gamma 校正 [0.1,3.0]
            "r": 1.0,        # R 通道增益 [0.1,3.0]
            "g": 1.0,        # G 通道增益 [0.1,3.0]
            "b": 1.0,        # B 通道增益 [0.1,3.0]
            "sat": 1.0,      # 饱和度 [0.1,3.0]
            "hue": 0.0,      # 色相偏移（度数）[-180,180]
            "lvl_b": 0.0,    # levels 黑场
            "lvl_g": 1.0,    # levels gamma
            "lvl_w": 255.0,  # levels 白场
        }

        # setting only click can be focused
        for w in self._click_focus_widgets:
            w.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        # only click can use button
        for btn in (getattr(self, "btn_run_seg", None),
                    getattr(self, "btn_run_surv", None),
                    getattr(self, "btn_save", None)):
            if btn:
                btn.setAutoDefault(False)
                btn.setDefault(False)
                btn.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self._hist_debounce = QtCore.QTimer(self)
        self._hist_debounce.setSingleShot(True)
        self._hist_debounce.setInterval(33)  # 约 30 FPS
        self._hist_debounce.timeout.connect(self._do_update_levels_histogram)

        # After the window is displayed, pull the focus back to
        # the main window to ensure that no controls are in the input state
        QtCore.QTimer.singleShot(0, lambda: self.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason))

        # The normal focus strategy is restored after the user's first click
        self._focus_restored = False
        self.installEventFilter(self)


        # load model used last time
        last_seg = self.settings.value("current_model_name_seg", "", type=str)
        if last_seg and last_seg in self.models_seg and os.path.exists(self.models_seg[last_seg]):
            self.combo_seg.setCurrentText(last_seg);
            self._set_current_model("seg")
        last_surv = self.settings.value("current_model_name_surv", "", type=str)
        if last_surv and last_surv in self.models_surv and os.path.exists(self.models_surv[last_surv]):
            self.combo_surv.setCurrentText(last_surv);
            self._set_current_model("surv")


        # ======= style setting =======
        with open("./UI/Style/style.qss", "r") as f:
            self.setStyleSheet(f.read())

        self._preview_override = None


    # ----------------- Persistence of the model dictionary -----------------
    def _load_models_dict(self, key):
        s = self.settings.value(key, "{}", type=str)
        try:
            d = json.loads(s)
            return {str(k): str(v) for k, v in d.items()}
        except Exception:
            return {}

    def _save_models_dict(self, key, d):
        self.settings.setValue(key, json.dumps(d))

    # ----------------- Operations on the model dictionary -----------------
    def _on_combo_changed(self, task):
        if task == "seg":
            name = self.combo_seg.currentText()
            path = self.models_seg.get(name, "")
            label = self.lab_model_seg
        else:
            name = self.combo_surv.currentText()
            path = self.models_surv.get(name, "")
            label = self.lab_model_surv
        if path and os.path.exists(path):
            label.setText(path);
            label.setStyleSheet("")
        else:
            label.setText("(missing) " + (path or ""));
            label.setStyleSheet("color: red;")

    def _add_model_from_file(self, task):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose ONNX model", "", "ONNX (*.onnx)")
        if not path: return
        name, ok = QtWidgets.QInputDialog.getText(self, "Model name", "Display name for this model:")
        if not ok or not name.strip(): return
        name = name.strip()
        if task == "seg":
            if name in self.models_seg:
                QtWidgets.QMessageBox.warning(self, "Name exists", "This name already exists.");
                return
            self.models_seg[name] = path
            self._save_models_dict("models_seg_json", self.models_seg)
            self.combo_seg.addItem(name);
            self.combo_seg.setCurrentText(name)
            self.append_log(f"[Seg] Added: {name} -> {path}")
        else:
            if name in self.models_surv:
                QtWidgets.QMessageBox.warning(self, "Name exists", "This name already exists.");
                return
            self.models_surv[name] = path
            self._save_models_dict("models_surv_json", self.models_surv)
            self.combo_surv.addItem(name);
            self.combo_surv.setCurrentText(name)
            self.append_log(f"[Surv] Added: {name} -> {path}")

    def _remove_selected_model(self, task):
        if task == "seg":
            name = self.combo_seg.currentText()
            if not name: return
            self.models_seg.pop(name, None)
            self._save_models_dict("models_seg_json", self.models_seg)
            self.combo_seg.removeItem(self.combo_seg.currentIndex())
            self.lab_model_seg.setText("No segmentation model loaded")
            self.lab_model_seg.setStyleSheet("color: grey;")
            self.settings.setValue("current_model_name_seg", "")
            self.seg_seg = None;
            self.btn_run_seg.setEnabled(False)
            self.append_log(f"[Seg] Removed: {name}")
        else:
            name = self.combo_surv.currentText()
            if not name: return
            self.models_surv.pop(name, None)
            self._save_models_dict("models_surv_json", self.models_surv)
            self.combo_surv.removeItem(self.combo_surv.currentIndex())
            self.lab_model_surv.setText("No survival model loaded")
            self.lab_model_surv.setStyleSheet("color: grey;")
            self.settings.setValue("current_model_name_surv", "")
            self.seg_surv = None;
            self.btn_run_surv.setEnabled(False)
            self.append_log(f"[Surv] Removed: {name}")

    def _set_current_model(self, task):
        if task == "surv":
            name = self.combo_surv.currentText()
            if not name: return
            path = self.models_surv.get(name, "")
            if not path or not os.path.exists(path):
                QtWidgets.QMessageBox.warning(self, "Missing file", f"[Surv] Not found:\n{path}")
                return
            try:
                self.seg_surv = SurvivalSegmentor(path)  # Survival class
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load failed", f"[Surv] {e}")
                return
            self.lab_model_surv.setText(path);
            self.lab_model_surv.setStyleSheet("")
            self.settings.setValue("current_model_name_surv", name)
            self.append_log(f"[Surv] Loaded: {name} -> {path}")
            self._update_run_buttons()
            return

        if task == "seg":
            name = self.combo_seg.currentText()
            if not name: return
            path = self.models_seg.get(name, "")
            if not path or not os.path.exists(path):
                QtWidgets.QMessageBox.warning(self, "Missing file", f"[Seg] Not found:\n{path}");
                return
            try:
                self.seg_seg = Segmentor(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load failed", f"[Seg] {e}");
                return
            self.lab_model_seg.setText(path);
            self.lab_model_seg.setStyleSheet("")
            self.settings.setValue("current_model_name_seg", name)
            self.append_log(f"[Seg] Loaded: {name} -> {path}")
            self.btn_run_seg.setEnabled(self.last_img is not None)
        else:
            name = self.combo_surv.currentText()
            if not name: return
            path = self.models_surv.get(name, "")
            if not path or not os.path.exists(path):
                QtWidgets.QMessageBox.warning(self, "Missing file", f"[Surv] Not found:\n{path}");
                return
            try:
                self.seg_surv = Segmentor(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load failed", f"[Surv] {e}");
                return
            self.lab_model_surv.setText(path);
            self.lab_model_surv.setStyleSheet("")
            self.settings.setValue("current_model_name_surv", name)
            self.append_log(f"[Surv] Loaded: {name} -> {path}")
            self.btn_run_surv.setEnabled(self.last_img is not None)



    #  ----------------- Choose images -----------------

    def choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if not path:
            return
        self._load_and_preview(path)

    def _on_files_dropped(self, paths: list[str]):
        """接收从 display 拖入的路径（文件或文件夹），将图片加入列表，并预览一张。"""
        self._ensure_image_paths_list()

        # 展开：若是目录，抓取其中所有图片；若是图片则保留
        def _collect_images(path: str) -> list[str]:
            if os.path.isdir(path):
                imgs = []
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
                    imgs.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
                return sorted(set(imgs))
            # 文件：判断是否图片
            ext = os.path.splitext(path)[1].lower()
            if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} and os.path.isfile(path):
                return [path]
            return []

        all_new = []
        for p in paths:
            all_new.extend(_collect_images(p))
        all_new = [p for p in all_new if p]  # 清理空项
        if not all_new:
            return

        # 去重加入 self.image_paths / self.list_images
        added = 0
        for p in all_new:
            if p not in self.image_paths:
                self.image_paths.append(p)
                self.list_images.addItem(p)
                added += 1

        if added:
            self.append_log(f"[Images] Added {added} file(s) from drag-and-drop.")

            # 记住最近目录
            if hasattr(self, "settings"):
                base_dir = os.path.dirname(all_new[0])
                self.settings.setValue("recent/dir_images", base_dir)

            # 预览第一张
            first = all_new[0]
            img = cv2.imread(first)
            if img is not None:
                self.last_img = img
                self.last_img_path = first
                if hasattr(self, "path_label"):
                    self.path_label.setText(first)
                self.show_image(img)
                self._update_run_buttons()
        else:
            self.append_log("[Images] Dragged files already in list, nothing added.")

    def _load_and_preview(self, path: str):
        img = cv2.imread(path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Read failed", "The image file cannot be read.")
            return
        self.path_label.setText(path)
        self.append_log(f"Loaded image: {path}")
        # self.lab_count.setText("Process Info: -")
        self.show_image(img)
        self.last_img = img
        self.last_img_path = path
        self.last_pred = None
        self.last_mode = None
        self.btn_save.setEnabled(False)
        # only have model loaded can do Run
        self.btn_run_seg.setEnabled(self.seg_seg is not None)
        self.btn_run_surv.setEnabled(self.seg_surv is not None)

        self._update_hist_from_image()

    # ---------------  segmentation inference -----------------

    def run_current(self):
        if self.seg is None:
            QtWidgets.QMessageBox.information(self, "No model",
                                              "Please set the current model first.")
            return
        if self.last_img is None:
            QtWidgets.QMessageBox.information(self, "No image",
                                              "Please select an image first.")
            return

        # inference
        pred = self.seg.predict(self.last_img)

        # Statistics
        n = 0
        if isinstance(pred, dict):
            boxes = pred.get("boxes", None)
            if boxes is not None and hasattr(boxes, "shape") and boxes.ndim >= 2:
                n = boxes.shape[0]
        self.lab_count.setText(f"Process Info: {n} objects")
        self.append_log(f"Detected {n} objects")

        # Rendering overlay mask
        thr = self.seg_score_thr.value() / 100.0
        alp = self.slider_alpha.value() / 100.0
        overlay = draw_overlay(self.last_img, pred, score_thr=thr, alpha=alp, render_on="orig")
        self.show_image(overlay)

        # Cache & Button status
        self.last_pred = pred
        self.btn_save.setEnabled(True)


    # ----------------- Dynamic parameter variation -----------------
    def _on_params_changed(self, *_):
        thr = self.seg_score_thr.value() / 100.0
        alp = self.slider_alpha.value() / 100.0
        self.lab_thr.setText(f"{thr:.2f}")
        # self.lab_alpha.setText(f"{alp:.2f}")
        if self.last_img is None or self.last_pred is None:
            return
        overlay = draw_overlay(self.last_img, self.last_pred,
                               score_thr=thr, alpha=alp, render_on="orig")

        self.show_image(overlay)
        if getattr(self, "_preview_override", None) is not None:
            return

    def _save_overlay(self):
        """
        保存当前显示图像。优先使用内存中的 _display_img（纯图像，无背景）。
        若没有 _display_img，则抓取 image_label，再按居中等比适配规则裁掉留白区域。
        """
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save current display", "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)"
        )
        if not path:
            return

        # 自动补后缀
        base, ext = os.path.splitext(path)
        if not ext:
            if "JPEG" in (selected_filter or "").upper():
                path = base + ".jpg"
            elif "TIFF" in (selected_filter or "").upper():
                path = base + ".tif"
            else:
                path = base + ".png"

        # —— 方案A：优先用 _display_img（纯图像、无背景）——
        try:
            if getattr(self, "_display_img", None) is not None:
                import cv2
                ok = cv2.imwrite(path, self._display_img)
                if not ok:
                    raise RuntimeError("cv2.imwrite returned False")
                self.append_log(f"[Save] Saved current display image: {path}")
                return
        except Exception as e:
            print("[DBG] Save via _display_img failed:", e)

        # —— 方案B：抓取 image_label 并裁剪出真实图像区域（等比填充产生的留白将被去掉）——
        try:
            target = getattr(self, "image_label", self)
            pm = target.grab()  # QPixmap（包含背景）
            if pm.isNull():
                raise RuntimeError("grab() returned null pixmap")
            qimg = pm.toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)

            # 估算图像在 label 内的摆放矩形（常见的等比居中自适应：AspectFit）
            # 需要 pm 的原始像素大小（抓屏像素），以及 label 的逻辑尺寸
            label_w = target.width()
            label_h = target.height()

            # 当前展示的原始图像尺寸（优先从 _pixmap_ref 拿）
            if getattr(self, "_pixmap_ref", None) is not None:
                img_w = self._pixmap_ref.width()
                img_h = self._pixmap_ref.height()
            else:
                # 兜底：用 label 尺寸近似
                img_w, img_h = label_w, label_h

            # 等比缩放以适配 label
            scale = min(label_w / max(1, img_w), label_h / max(1, img_h))
            draw_w = int(round(img_w * scale))
            draw_h = int(round(img_h * scale))
            # 居中留白
            off_x = (label_w - draw_w) // 2
            off_y = (label_h - draw_h) // 2

            # 设备像素比（HiDPI）
            dpr = pm.devicePixelRatio() if hasattr(pm, "devicePixelRatio") else 1.0
            rx = int(round(off_x * dpr))
            ry = int(round(off_y * dpr))
            rw = int(round(draw_w * dpr))
            rh = int(round(draw_h * dpr))

            # 防越界
            rw = max(1, min(rw, qimg.width() - rx))
            rh = max(1, min(rh, qimg.height() - ry))

            cropped = qimg.copy(rx, ry, rw, rh)

            # 写盘（QImageWriter）
            fmt = os.path.splitext(path)[1].lower().lstrip(".") or "png"
            if fmt in ("jpg", "jpeg"): fmt = "jpeg"
            if fmt in ("tif", "tiff"): fmt = "tiff"
            writer = QtGui.QImageWriter(path, fmt.encode("ascii"))
            writer.setQuality(95)
            if writer.write(cropped):
                self.append_log(f"[Save] Saved current display image: {path}")
                return

            # QImageWriter 失败则降级为 OpenCV
            import numpy as np, cv2
            width, height = cropped.width(), cropped.height()
            ptr = cropped.bits()
            ptr.setsize(cropped.sizeInBytes())
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            ok = cv2.imwrite(path, bgr)
            if not ok:
                raise RuntimeError("cv2.imwrite returned False")

            self.append_log(f"[Save] Saved current display image: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"Failed to save image:\n{path}\n\nDetail: {e}")


    # ----------------- showing image -----------------

    def _update_save_button(self):
        has_pix = (self.image_label.pixmap() is not None)
        self.btn_save.setEnabled(has_pix)

    def show_image(self, img_bgr: np.ndarray, reset_zoom=True):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QtGui.QImage(img_rgb.data, w, h, img_rgb.strides[0],
                            QtGui.QImage.Format.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg)

        # 交给 zoom label；reset_zoom=True 表示每次展示新图回到 1:1
        self.image_label.set_zoom_pixmap(pm, reset=reset_zoom)

        # enable save
        if hasattr(self, "btn_save"):
            self.btn_save.setEnabled(True)

        # 保留一份当前显示像素，供“保存显示内容”使用
        self._pixmap_ref = pm
        self._update_levels_histogram()



    def _run_seg(self):
        self._clear_preview_override()

        if self.seg_seg is None:
            QtWidgets.QMessageBox.information(self, "No model", "Please set a segmentation model.")
            return
        if self.last_img is None:
            QtWidgets.QMessageBox.information(self, "No image", "Please select an image first.")
            return


        # add top-k value before pred
        max_dets_runtime = int(getattr(self, "max_dets", 0)) or None  # 0 mean do not cover
        iou_thr_runtime = None  # or float(self.seg_nms_iou)
        score_thr_graph = None  # or float(self.seg_score_thr)
        if hasattr(self.seg_seg, "set_runtime_params"):
            self.seg_seg.set_runtime_params(
                max_dets=max_dets_runtime,
                iou_thr=iou_thr_runtime,
                score_thr=score_thr_graph,
                topk_k=max_dets_runtime,  # if image have TopK_k
            )

        working_img = self._apply_adjust_for_processing(self.last_img)

        pred = self.seg_seg.predict(working_img)

        frame = pred.get("frame", self.last_img)
        boxes = pred.get("boxes", None)
        scores = pred.get("scores", None)
        classes = pred.get("classes", None)
        masks = pred.get("masks", None)

        # reshape to (N,), (N,), (N,)
        boxes, scores, classes, masks = _normalize_det_tensors(boxes, scores, classes, masks)

        # ------- reshape box 回原图大小 -------
        # 原图尺寸（显示用）
        orig_H, orig_W = self.last_img.shape[:2]

        # 模型 frame 尺寸（推理用）
        frm_H, frm_W = frame.shape[:2] if frame is not None else (orig_H, orig_W)

        # 如果 frame 尺寸与原图不同，把 boxes 缩放回原图坐标
        if (frm_H, frm_W) != (orig_H, orig_W) and boxes is not None:
            boxes = self._rescale_boxes(boxes, (frm_H, frm_W), (orig_H, orig_W))

        # statistical
        n = int(boxes.shape[0]) if (boxes is not None and hasattr(boxes, "shape") and boxes.ndim >= 2) else 0

        # === 1. score threshold ===
        score_thr = float(self.seg_score_thr)
        if scores is not None and len(scores) > 0:
            keep = np.where(scores >= score_thr)[0]
            boxes = boxes[keep] if boxes is not None else None
            scores = scores[keep] if scores is not None else None
            classes = classes[keep] if classes is not None else None
            masks = masks[keep] if masks is not None else None

        # === 2. NMS ===
        # classwise (False=NMS；True=按类别NMS）
        if boxes is not None and scores is not None and len(boxes) > 0 and self.seg_nms_iou > 0:
            keep = _nms_keep_indices(boxes, scores, iou_thr=float(self.seg_nms_iou),
                                     classes=classes, classwise=False)
            boxes = boxes[keep] if boxes is not None else None
            scores = scores[keep] if scores is not None else None
            classes = classes[keep] if classes is not None else None
            masks = masks[keep] if masks is not None else None

        # === 3. minimize size filter ===
        if boxes is not None and len(boxes) > 0 and (self.seg_min_area > 0 ):
            m = _filter_by_min(boxes, min_area=float(self.seg_min_area), min_side=float(self.seg_min_side))
            boxes = boxes[m]
            scores = scores[m] if scores is not None else None
            classes = classes[m] if classes is not None else None
            masks = masks[m] if masks is not None else None

        # === 4. keep top-K instances ===
        if scores is not None and self.seg_top_k > 0:
            # 1) 按分数排序（从高到低）
            idx = scores.argsort()[::-1][:self.seg_top_k]    # 降序 → 取前 K


            # 2) 应用索引
            boxes = boxes[idx]
            scores = scores[idx]
            classes = classes[idx] if classes is not None else None
            masks = masks[idx] if masks is not None else None

        # === statistic ===
        n = int(boxes.shape[0]) if (boxes is not None and hasattr(boxes, "shape") and boxes.ndim >= 2) else 0
        self.set_process_info(f"Process Info (Seg): {n} objects")
        self.append_log(
            f"[Seg] Detected {n} objects | thr={score_thr:.2f}, nms={self.seg_nms_iou}, min_area={self.seg_min_area}, min_side={self.seg_min_side}")


        self.set_process_info(f"Process Info (Seg): {n} objects")
        self.append_log(f"[Seg] Detected {n} objects")


        # === Build full-size binary masks once (for overlay + brush + export) ===
        disp_H, disp_W = self.last_img.shape[:2]


        mask_thr = 0.7

        masks_full = None
        if masks is not None:
            canvas_list = []
            for i, m in enumerate(masks):
                if m is None:
                    continue
                if m.ndim == 3 and m.shape[2] == 1:
                    m = m[..., 0]

                full = np.zeros((disp_H, disp_W), dtype=np.float32)

                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = [int(round(v)) for v in boxes[i]]
                    x1 = max(0, min(disp_W - 1, x1))
                    x2 = max(0, min(disp_W, x2))
                    y1 = max(0, min(disp_H - 1, y1))
                    y2 = max(0, min(disp_H, y2))
                    bw, bh = max(0, x2 - x1), max(0, y2 - y1)

                    if bw > 0 and bh > 0:
                        # ROI 小 mask → bbox 尺寸（只做几何缩放，不二值化）
                        if m.ndim == 2 and (m.shape[0] != disp_H or m.shape[1] != disp_W):
                            m_roi = cv2.resize(m.astype(np.float32), (bw, bh),
                                               interpolation=cv2.INTER_NEAREST)
                            full[y1:y2, x1:x2] = m_roi
                        else:
                            # 已经是整图 soft mask
                            full = m.astype(np.float32)
                    else:
                        # bbox 异常：退化为整图
                        full = m.astype(np.float32)
                else:
                    # 无 bbox：若已整图就用；否则整体拉到整图
                    if m.ndim == 2 and m.shape[:2] == (disp_H, disp_W):
                        full = m.astype(np.float32)
                    else:
                        full = cv2.resize(m.astype(np.float32), (disp_W, disp_H),
                                          interpolation=cv2.INTER_NEAREST)

                # === 统一阈值二值化（只在这里做一次）===
                full_bin = (full > mask_thr).astype(np.uint8) * 255
                canvas_list.append(full_bin)

            if canvas_list:
                masks_full = np.stack(canvas_list, axis=0)  # (N,H,W)
            else:
                masks_full = np.zeros((0, disp_H, disp_W), dtype=np.uint8)

        self.last_pred = {
            "frame": frame,
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
            "masks": masks_full,  # 整图mask（编辑/展示/导出复用）
            "masks_disp": masks_full,  # 保留键名以兼容Brush，不再生成独立版本
            "disp_size": (disp_H, disp_W),
        }

        # draw
        alp = self.slider_alpha.value() / 100.0
        overlay = draw_overlay(working_img, None, boxes, scores, classes, masks_full,
                               score_thr=score_thr, alpha=alp, draw_box=False, coords="orig")

        self.show_image(overlay)

        self.last_mode = "seg"
        self.btn_save.setEnabled(True)

        # save the merged mask at the same time
        merged_mask_frame = build_merged_mask(frame, boxes, masks_full)
        if merged_mask_frame.shape[:2] != self.last_img.shape[:2]:
            merged_mask = cv2.resize(merged_mask_frame, (self.last_img.shape[1], self.last_img.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            merged_mask = merged_mask_frame
        self.last_mode = "seg"
        self.btn_save.setEnabled(True)



    def _run_surv(self):
        self._clear_preview_override()

        if self.seg_surv is None:
            QtWidgets.QMessageBox.information(self, "No model", "Please set a survival model.")
            return
        if self.last_img is None:
            QtWidgets.QMessageBox.information(self, "No image", "Please select an image first.")
            return

        working_img = self._apply_adjust_for_processing(self.last_img)

        pred = self.seg_surv.predict(working_img)  # SurvivalSegmentor size fit

        alp = self.slider_alpha.value() / 100.0
        overlay = self.last_img.copy()

        # 1. death_prob heatmap
        death = pred.get("death_prob", None)
        if death is not None:
            heat8 = (np.clip(death, 0, 1) * 255).astype(np.uint8)
            cm = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)  # BGR color
            overlay = cv2.addWeighted(overlay, 1 - alp, cm, alp, 0)

        # 2. stack the colored areas and strokes according to the three types of masks
        mask = pred.get("mask", None)
        n_alive = n_dead = 0
        if mask is not None:
            # Make sure it is a two-dimensional integer
            m = np.asarray(mask).astype(np.int32)
            alive = (m == 1)
            dead = (m == 2)

            def _blend_class(ov, cls_mask, color_bgr, alpha_fill=0.35, edge_thick=2):
                if not np.any(cls_mask):
                    return 0
                # Semi-transparent filling (only fusion in mask=ture)
                idx = cls_mask
                ov[idx] = (
                        ov[idx].astype(np.float32) * (1 - alpha_fill)
                        + np.array(color_bgr, dtype=np.float32) * alpha_fill
                ).astype(np.uint8)

                # draw contours
                cnts, _ = cv2.findContours(cls_mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cv2.drawContours(ov, cnts, -1, color_bgr, edge_thick)

                # calculate number of connective area
                num = max(cv2.connectedComponents(cls_mask.astype(np.uint8))[0] - 1, 0)
                return num

            # color(BGR): alive(green); died(red)
            n_alive = _blend_class(overlay, alive, (0, 220, 0), alpha_fill=0.30, edge_thick=2)
            n_dead = _blend_class(overlay, dead, (0, 0, 255), alpha_fill=0.30, edge_thick=2)

        self.show_image(overlay)
        if mask is not None:
            self.set_process_info(f"Process Info (Surv): alive={n_alive}, dead={n_dead}")
            self.append_log(f"[Surv] Alive={n_alive}, Dead={n_dead}")
        else:
            self.set_process_info("Process Info (Surv): heatmap only")
            self.append_log("[Surv] Rendered survival heatmap")

        self.last_pred = pred
        self.last_mode = "surv"
        self.btn_save.setEnabled(True)

    def _on_params_changed(self, *_):
        # transparency setting
        alp = self.slider_alpha.value() / 100.0
        # self.lab_alpha.setText(f"{alp:.2f}")

        if self.last_img is None or self.last_pred is None:
            return

        if self.last_mode == "seg":
            frame = self.last_pred.get("frame", self.last_img)
            boxes = self.last_pred.get("boxes", None)
            scores = self.last_pred.get("scores", None)
            classes = self.last_pred.get("classes", None)
            masks = self.last_pred.get("masks", None)

            # Based on the current operating state seg_score_thr, perform another "display use" threshold
            # (without repeating NMS)
            score_thr = float(self.seg_score_thr)
            keep = (np.where(scores >= score_thr)[0]
                    if (scores is not None and len(scores) > 0) else slice(None))
            boxes_v = boxes[keep] if boxes is not None else None
            scores_v = scores[keep] if scores is not None else None
            classes_v = classes[keep] if classes is not None else None
            masks_v = masks[keep] if masks is not None else None

            overlay = draw_overlay(self.last_img, frame, boxes_v, scores_v, classes_v, masks_v,
                                   score_thr=score_thr, alpha=alp, draw_box=False)
        else:
            overlay = self.last_img.copy()
            death = self.last_pred.get("death_prob", None)
            if death is not None:
                heat = (np.clip(death, 0, 1) * 255).astype(np.uint8)
                cm = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(overlay, 1 - alp, cm, alp, 0)
            mask = self.last_pred.get("mask", None)
            if mask is not None:
                cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 1)

        self.show_image(overlay)

    def _update_run_buttons(self):
        """two running buttons are only can be use when image and weight ready"""
        seg_ready  = (self.seg_seg  is not None) and (self.last_img is not None)
        surv_ready = (self.seg_surv is not None) and (self.last_img is not None)
        self.btn_run_seg.setEnabled(seg_ready)
        self.btn_run_surv.setEnabled(surv_ready)


    # --------------- json annotation -------------
    def _serialize_pred_to_json(self) -> dict:
        """
        将 self.last_pred 标准化为一个可 JSON 序列化的 dict。
        要求：在 _run_seg() 之后，last_pred = {"frame","boxes","scores","classes","masks"}.
        """
        import numpy as np
        import cv2

        if self.last_pred is None or self.last_mode != "seg":
            raise RuntimeError("当前没有可导出的分割/检测结果（请先运行 Seg）。")

        H, W = (self.last_img.shape[0], self.last_img.shape[1]) if self.last_img is not None else (None, None)
        boxes = self.last_pred.get("boxes", None)
        scores = self.last_pred.get("scores", None)
        classes = self.last_pred.get("classes", None)
        masks = self.last_pred.get("masks", None)

        # 类别名（可选）
        id2name = None
        if hasattr(self, "seg_seg") and hasattr(self.seg_seg, "class_names"):
            # 期望：list[str] 或 dict[int,str]
            cn = self.seg_seg.class_names
            if isinstance(cn, dict):
                id2name = {int(k): str(v) for k, v in cn.items()}
            elif isinstance(cn, (list, tuple)):
                id2name = {i: str(n) for i, n in enumerate(cn)}

        n = 0 if boxes is None else int(boxes.shape[0])
        objs = []
        for i in range(n):
            bb = boxes[i].tolist() if boxes is not None else None
            sc = float(scores[i]) if scores is not None else None
            lb = int(classes[i]) if classes is not None else None
            item = {
                "id": i,
                "bbox_xyxy": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])] if bb is not None else None,
                "score": sc,
                "label_id": lb,
            }
            if id2name is not None and lb is not None and lb in id2name:
                item["label_name"] = id2name[lb]

            # ---- 可选：从实例 mask 提取 polygon（单外轮廓）----
            if masks is not None and len(masks) > i and masks[i] is not None:
                m = (masks[i] > 0).astype(np.uint8)
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 合并所有外轮廓为一个多边形数组（COCO 通常允许多个），这里存第一条也可
                polys = []
                for c in cnts:
                    pts = c.reshape(-1, 2)
                    polys.append([[int(x), int(y)] for (x, y) in pts])
                if polys:
                    item["mask"] = {"type": "polygon", "data": polys}

            objs.append(item)

        jd = {
            "image_path": getattr(self, "last_img_path", None),
            "image_size": [int(H), int(W)] if H is not None else None,
            "objects": objs,
        }
        return jd

    def _save_masks_png_only(self):
        """
        只导出当前 Seg 结果的 mask PNG：
        - 每个实例一张二值 PNG（整图大小，0/255），命名：mask_{idx:03d}_{cls或name}.png
        - 另外导出 merged_mask.png（所有实例并集）
        """
        if self.last_pred is None or self.last_mode != "seg":
            QtWidgets.QMessageBox.information(self, "Export Mask (PNG)", "请先运行 Seg。")
            return

        masks = self.last_pred.get("masks", None)
        boxes = self.last_pred.get("boxes", None)
        classes = self.last_pred.get("classes", None)
        if masks is None:
            QtWidgets.QMessageBox.information(self, "Export Mask (PNG)", "当前结果没有 mask。")
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if not out_dir:
            return

        H, W = self.last_img.shape[:2]

        # 类别名（可选，仅用于文件名）
        id2name = None
        if hasattr(self, "seg_seg") and hasattr(self.seg_seg, "class_names"):
            cn = self.seg_seg.class_names
            if isinstance(cn, dict):
                id2name = {int(k): str(v) for k, v in cn.items()}
            elif isinstance(cn, (list, tuple)):
                id2name = {i: str(n) for i, n in enumerate(cn)}

        def _full_binary(m, box):
            m = np.asarray(m)
            #如果已经是整图大小 -> 直接用
            if m.ndim >= 2 and m.shape[-2:] == (H, W):
                full = m
            else:
                # 仍然是 ROI mask（比如 28x28）时，才用 box 贴回整图
                full = self._mask_to_fullsize(m, box, (H, W))
            return (full > 0).astype(np.uint8) * 255

        N = len(masks) if isinstance(masks, (list, tuple)) else (masks.shape[0] if hasattr(masks, "shape") else 0)
        merged = np.zeros((H, W), dtype=np.uint8)

        for i in range(N):
            m = masks[i] if isinstance(masks, (list, tuple)) else masks[i, ...]
            b = boxes[i] if boxes is not None and i < len(boxes) else None
            full = _full_binary(m, b)
            merged |= (full > 0).astype(np.uint8) * 255

            cls_id = int(classes[i]) if classes is not None and i < len(classes) and classes[i] is not None else -1
            cls_tag = f"class{cls_id}"
            if id2name and cls_id in id2name:
                cls_tag = str(id2name[cls_id]).replace("/", "_").replace("\\", "_").replace(" ", "_")

            save_path = os.path.join(out_dir, f"mask_{i:03d}_{cls_tag}.png")
            cv2.imwrite(save_path, full)

        cv2.imwrite(os.path.join(out_dir, "merged_mask.png"), merged)
        self.append_log(f"[Export] {N} mask PNG saved to: {out_dir}")
        QtWidgets.QMessageBox.information(self, "Export Mask (PNG)", f"已导出 {N} 个实例 PNG 到：\n{out_dir}")

    def _save_masks_json_only(self):
        """
        只导出当前 Seg 结果的 mask JSON（不写 PNG 文件）：
        - masks_meta.json: { image_path, image_size, objects:[{id,label_id,label_name?,bbox_xyxy,mask:polygon}] }
        - mask 以外轮廓 polygon 表达（可能多条）
        """
        import os, json, cv2, numpy as np

        if self.last_pred is None or self.last_mode != "seg":
            QtWidgets.QMessageBox.information(self, "Export Mask (JSON)", "请先运行 Seg。")
            return

        masks = self.last_pred.get("masks", None)
        boxes = self.last_pred.get("boxes", None)
        classes = self.last_pred.get("classes", None)
        if masks is None:
            QtWidgets.QMessageBox.information(self, "Export Mask (JSON)", "当前结果没有 mask。")
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output directory")
        if not out_dir:
            return

        H, W = self.last_img.shape[:2]

        # 类别名（可选）
        id2name = None
        if hasattr(self, "seg_seg") and hasattr(self.seg_seg, "class_names"):
            cn = self.seg_seg.class_names
            if isinstance(cn, dict):
                id2name = {int(k): str(v) for k, v in cn.items()}
            elif isinstance(cn, (list, tuple)):
                id2name = {i: str(n) for i, n in enumerate(cn)}

        def _full_binary(m, box):
            m = np.asarray(m)
            #如果已经是整图大小 -> 直接用
            if m.ndim >= 2 and m.shape[-2:] == (H, W):
                full = m
            else:
                # 仍然是 ROI mask（比如 28x28）时，才用 box 贴回整图
                full = self._mask_to_fullsize(m, box, (H, W))
            return (full > 0).astype(np.uint8) * 255

        N = len(masks) if isinstance(masks, (list, tuple)) else (masks.shape[0] if hasattr(masks, "shape") else 0)
        objects = []

        for i in range(N):
            m = masks[i] if isinstance(masks, (list, tuple)) else masks[i, ...]
            b = boxes[i] if boxes is not None and i < len(boxes) else None
            full = _full_binary(m, b)

            # 多边形（外轮廓）
            cnts, _ = cv2.findContours((full > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = []
            for c in cnts:
                pts = c.reshape(-1, 2)
                if len(pts) >= 3:
                    polys.append([[int(x), int(y)] for (x, y) in pts])

            # bbox
            if b is None:
                ys, xs = np.where(full > 0)
                if xs.size == 0 or ys.size == 0:
                    bbox_xyxy = [0.0, 0.0, 0.0, 0.0]
                else:
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
                    bbox_xyxy = [float(x1), float(y1), float(x2), float(y2)]
            else:
                bbox_xyxy = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

            cls_id = int(classes[i]) if classes is not None and i < len(classes) and classes[i] is not None else -1
            obj = {
                "id": i,
                "label_id": cls_id,
                "bbox_xyxy": bbox_xyxy,
            }
            if id2name and cls_id in id2name:
                obj["label_name"] = id2name[cls_id]
            if polys:
                obj["mask"] = {"type": "polygon", "data": polys}

            objects.append(obj)

        jd = {
            "image_path": getattr(self, "last_img_path", None),
            "image_size": [int(H), int(W)],
            "objects": objects
        }

        save_path = os.path.join(out_dir, "masks_meta.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(jd, f, ensure_ascii=False, indent=2)

        self.append_log(f"[Export] mask JSON saved: {save_path}")
        QtWidgets.QMessageBox.information(self, "Export Mask (JSON)", f"已导出：\n{save_path}")

    def _on_ec_finished_show_only(self, result=None):
        """EdgeCluster 完成：仅预览，并把保存动作交给现有的“保存按钮”."""
        import os, cv2, glob, shutil
        try:
            if hasattr(self, "_ec_progress") and self._ec_progress:
                self._ec_progress.close()

            ctx = getattr(self, "_ec_ctx", {}) or {}
            temp_out, temp_lab = ctx.get("temp_out"), ctx.get("temp_lab")

            # 1) 选一张可预览的结果图
            preview_path = None
            if temp_out and os.path.isdir(temp_out):
                cand = []
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
                    cand.extend(glob.glob(os.path.join(temp_out, ext)))
                if cand:
                    preview_path = sorted(cand)[0]

            # 2) 预览展示
            if preview_path and os.path.isfile(preview_path):
                img = cv2.imread(preview_path)
                if img is not None:
                    self.show_image(img)
                    self.append_log(f"[EdgeCluster] Preview: {preview_path}")
                else:
                    self.append_log("[EdgeCluster] Preview read failed.")
            else:
                self.append_log("[EdgeCluster] No preview file found.")

            # 3) 与 seg/surv 同步：更新状态并启用保存按钮
            self.last_pred = {"edge_preview": preview_path}
            self.last_mode = "edge"
            self.btn_save.setEnabled(True)  # 现在用户点击“保存”就会走 _save_overlay


        finally:
            # 4) 清理临时资源（不落地结果目录）
            ctx = getattr(self, "_ec_ctx", {}) or {}
            try:
                ti = ctx.get("temp_input")
                if ti and os.path.isfile(ti):
                    os.remove(ti)
            except Exception:
                pass
            for k in ("temp_out", "temp_lab"):
                try:
                    td = ctx.get(k)
                    if td and os.path.isdir(td):
                        shutil.rmtree(td, ignore_errors=True)
                except Exception:
                    pass
            self._ec_ctx = {}

    def _on_ec_failed_cleanup(self, err_msg: str):
        """EdgeCluster 失败时的清理，与 seg/surv 一致的 UX：只提示，不弹保存。"""
        if hasattr(self, "_ec_progress") and self._ec_progress:
            self._ec_progress.close()
        QtWidgets.QMessageBox.critical(self, "Edge clustering failed", str(err_msg))

        # 清理临时
        ctx = getattr(self, "_ec_ctx", {}) or {}
        import os, shutil
        try:
            ti = ctx.get("temp_input")
            if ti and os.path.isfile(ti):
                os.remove(ti)
        except Exception:
            pass
        for k in ("temp_out", "temp_lab"):
            try:
                td = ctx.get(k)
                if td and os.path.isdir(td):
                    shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass
        self._ec_ctx = {}


    def _run_edge_cluster(self):
        self._clear_preview_override()
        if self.last_img is None:
            QtWidgets.QMessageBox.information(self, "No image", "Please select an image first.")
            return

        # 读取参数
        params = EdgeClusterParams(
            canny_low=int(self.spin_ec_canny_low.value()),
            canny_high=int(self.spin_ec_canny_high.value()),
            db_eps=float(self.spin_ec_eps.value()),
            db_min_samples=int(self.spin_ec_min_samples.value()),
            db_min_cluster_size=int(self.spin_ec_min_cluster.value()),
            morph_kernel=int(self.spin_ec_kernel.value()),
        )

        # ★ 强制以处理图为输入写临时文件
        img_proc = self._apply_adjust_for_processing(self.last_img)

        self._ec_ctx = {"temp_input": None, "temp_out": None, "temp_lab": None}
        fd, temp_in = tempfile.mkstemp(suffix=".png", prefix="ec_in_")
        os.close(fd)
        cv2.imwrite(temp_in, img_proc)
        input_path = temp_in
        self._ec_ctx["temp_input"] = temp_in


        # params = EdgeClusterParams()

        # 临时目录（仅用于预览，预览后清理）
        import tempfile as _tf
        temp_out = _tf.mkdtemp(prefix="ec_out_")
        temp_lab = _tf.mkdtemp(prefix="ec_lab_")
        self._ec_ctx["temp_out"] = temp_out
        self._ec_ctx["temp_lab"] = temp_lab

        # 3) 进度框
        self._ec_progress = QtWidgets.QProgressDialog("Processing…", "Cancel", 0, 0, self)
        self._ec_progress.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self._ec_progress.setMinimumDuration(0)
        self._ec_progress.show()

        # 4) 线程与 worker（只处理这一张）
        self._ec_thread = QtCore.QThread(self)
        self._ec_worker = EdgeClusterWorker(input_path, temp_out, temp_lab, params)
        self._ec_worker.moveToThread(self._ec_thread)

        self._ec_thread.started.connect(self._ec_worker.run)
        self._ec_worker.progressed.connect(self._on_ec_progress)
        self._ec_worker.finished.connect(self._on_ec_finished_show_only)  # 只展示，不询问保存
        self._ec_worker.failed.connect(self._on_ec_failed_cleanup)
        self._ec_progress.canceled.connect(self._ec_worker.cancel)

        self._ec_worker.finished.connect(self._ec_thread.quit)
        self._ec_worker.finished.connect(self._ec_worker.deleteLater)
        self._ec_thread.finished.connect(self._ec_thread.deleteLater)

        self._ec_thread.start()


    def _on_ec_failed_cleanup(self, err_msg: str):
        if hasattr(self, "_ec_progress") and self._ec_progress:
            self._ec_progress.close()
        QtWidgets.QMessageBox.critical(self, "Edge clustering failed", str(err_msg))
        # 清理
        ctx = getattr(self, "_ec_ctx", {}) or {}
        import os, shutil
        try:
            ti = ctx.get("temp_input")
            if ti and os.path.isfile(ti): os.remove(ti)
        except Exception:
            pass
        for k in ("temp_out", "temp_lab"):
            try:
                td = ctx.get(k)
                if td and os.path.isdir(td): shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass
        self._ec_ctx = {}


    def _on_ec_progress(self, i, n, path, stats):
        if n > 0:
            self._ec_progress.setMaximum(n)
            self._ec_progress.setValue(i)
            self._ec_progress.setLabelText(
                f"[{i}/{n}] {os.path.basename(path)} | clusters={stats.get('clusters', '-')}")
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().showMessage(self._ec_progress.labelText())


    def _ensure_image_paths_list(self):
        if not hasattr(self, "image_paths"):
            self.image_paths = []

    def _add_images_to_list(self):
        self._ensure_image_paths_list()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add images",
            self.settings.value("recent/dir_images", "", type=str) if hasattr(self, "settings") else "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)"
        )
        if not files:
            return
        if hasattr(self, "settings"):
            self.settings.setValue("recent/dir_images", os.path.dirname(files[0]))
        for f in files:
            if f not in self.image_paths:
                self.image_paths.append(f)
                self.list_images.addItem(f)
        self.append_log(f"[Images] Added {len(files)} file(s).")


    def _remove_selected_images(self):
        self._ensure_image_paths_list()
        items = self.list_images.selectedItems()
        if not items:
            return

        removed_paths = []
        for it in items:
            path = it.text()
            if path in self.image_paths:
                self.image_paths.remove(path)
                removed_paths.append(path)
            self.list_images.takeItem(self.list_images.row(it))

        self.append_log(f"[Images] Removed {len(items)} item(s).")

        # === 如果当前显示的图像路径已被删除，则清空显示 ===
        if getattr(self, "last_img_path", None) in removed_paths:
            self._reset_current_image_state()


    def _clear_images_list(self):
        self._ensure_image_paths_list()
        self.image_paths.clear()
        self.list_images.clear()
        self.append_log("[Images] Cleared list.")
        self._reset_current_image_state()


    def _reset_current_image_state(self):
        # 清空显示
        try:
            if hasattr(self, "image_label") and self.image_label is not None:
                # 如果 ZoomableLabel 有自定义清除方法，优先用；否则用 clear()
                if hasattr(self.image_label, "set_zoom_pixmap"):
                    self.image_label.set_zoom_pixmap(QtGui.QPixmap(), reset=True)
                else:
                    self.image_label.clear()
        except Exception:
            pass

        # 清空缓存&状态
        self._pixmap_ref = None
        self._preview_override = None

        self.last_img = None
        self.last_img_path = None
        self.last_pred = None
        self.last_mode = None

        # 禁用相关按钮&信息
        if hasattr(self, "btn_save"): self.btn_save.setEnabled(False)
        self._update_run_buttons()
        if hasattr(self, "hist_widget"): self.hist_widget.set_hist(None)
        self.set_process_info("Process Info: -")

    def _preview_selected_image(self):
        items = self.list_images.selectedItems()
        if not items:
            return
        path = items[0].text()
        img = cv2.imread(path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Read failed", f"Cannot read:\n{path}")
            return
        self.last_img = img
        self.last_img_path = path
        self.last_pred = None
        self.last_mode = None
        if hasattr(self, "path_label"):
            self.path_label.setText(path)
        self.show_image(img)
        self._update_run_buttons()
        self._update_levels_histogram()

    def _focus_stack_selected(self, method: str = "sobel"):
        """
        与 seg/surv 保持一致的交互：计算 -> 预览 -> 启用“保存”按钮（不在此处弹保存对话框）
        method: "sobel" | "max" | "mean"
        """
        items = self.list_images.selectedItems() if hasattr(self, "list_images") else []
        paths = [it.text() for it in items]
        if len(paths) < 2:
            QtWidgets.QMessageBox.information(self, "Projection", "Please select at least two images.")
            return

        try:
            extra_info = {}
            method_l = method.lower()

            if method_l == "sobel":
                composite_full, idx_map, sharp_stack = focus_stack_sobel(paths)
                extra_info = {"stack_idx": idx_map, "sharp_stack_shape": getattr(sharp_stack, "shape", None)}
                tag = "Sobel"
                preview8 = composite_full  # sobel 本身是拷贝原像素，通常已是 uint8/可直接预览

            else:
                # stack: (N,H,W) or (N,H,W,3), dtype 可能是 uint8/uint16
                stack = self._load_image_stack_same_size(paths)
                out_dtype = stack.dtype
                out_max = np.iinfo(out_dtype).max if np.issubdtype(out_dtype, np.integer) else None

                if method_l == "max":
                    tag = "Max"
                    composite_f = np.max(stack.astype(np.float32), axis=0)
                elif method_l == "mean":
                    tag = "Mean"
                    composite_f = np.mean(stack.astype(np.float32), axis=0)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # === full 输出：保持原位深（用于保存）===
                if out_max is not None:
                    composite_f = np.clip(composite_f, 0, out_max)
                    composite_full = np.rint(composite_f).astype(out_dtype)
                    # === 8bit 预览：正确缩放 ===
                    preview8 = (composite_f / out_max * 255.0).astype(np.uint8)
                else:
                    composite_full = composite_f.astype(out_dtype)
                    mn, mx = composite_f.min(), composite_f.max()
                    preview8 = ((composite_f - mn) / (mx - mn + 1e-6) * 255.0).astype(np.uint8)

            # 3) UI 预览永远走 uint8
            if preview8.ndim == 2:
                comp_bgr_preview = cv2.cvtColor(preview8, cv2.COLOR_GRAY2BGR)
            else:
                comp_bgr_preview = preview8  # 已是 uint8

            self.show_image(comp_bgr_preview)

            # 4) 缓存：full 用于 Save，preview 用于 UI
            self.last_pred = {
                "projection_preview": comp_bgr_preview,  # uint8 预览
                "projection_full": composite_full,  # 原位深结果
                "method": method,
                **extra_info
            }
            self.last_mode = "projection"
            self.btn_save.setEnabled(True)

            self.append_log(f"[Projection] {tag} from {len(paths)} images. Preview shown. Use 'Save' to export.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Projection", f"Failed:\n{e}")
            return

    # ======== 辅助函数：读入并校验尺寸一致 ========
    def _load_image_stack_same_size(self, paths):
        imgs = []
        H = W = None
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"Cannot read: {p}")
            h, w = img.shape[:2]
            if H is None:
                H, W = h, w
            elif (H, W) != (h, w):
                raise ValueError(f"Different sizes: {os.path.basename(p)} is {h}x{w}, expected {H}x{W}")
            imgs.append(img)
        # shape: (N,H,W) or (N,H,W,3)
        return np.stack(imgs, axis=0)

    def _mean_to_dtype(self, stack: np.ndarray) -> np.ndarray:
        """
        对 (N,H,W[,C]) 求均值，并安全地转换回原始 dtype（uint8/uint16/float）。
        """
        orig_dtype = stack.dtype
        mean_f = stack.astype(np.float32).mean(axis=0)  # (H,W[,C])
        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            mean_f = np.clip(np.rint(mean_f), info.min, info.max)
            return mean_f.astype(orig_dtype)
        else:
            return mean_f.astype(orig_dtype)

    def _on_image_adjust(self, *_):
        if self.last_img is None:
            return
        # 从 slider 取参数
        brightness = self.slider_brightness.value()  # -100 ~ 100
        contrast = self.slider_contrast.value() / 100.0  # 0.1 ~ 3.0

        # 调整图像
        img = self.last_img.astype(np.float32)
        img = img * contrast + brightness
        img = np.clip(img, 0, 255).astype(np.uint8)

        # 预览更新
        self.show_image(img)
        # self.text_log.append(f"[Adjust] brightness={brightness}, contrast={contrast:.2f}")


    # —— 统一追加日志 ——
    def append_log(self, text: str):
        # 可用在任何地方：self.append_log("xxx")
        if self.info_window:
            self.info_window.append_log(text)
        else:
            # 窗口未开时先缓存在内存（可选）
            self._log_buffer.append(text)

    def set_process_info(self, text: str):
        if hasattr(self, "lab_count") and self.lab_count is not None:
            self.lab_count.setText(text)
        if self.info_window:
            self.info_window.set_info(text)

    # —— 打开/关闭独立信息窗 ——
    def _toggle_info_window(self, checked: bool, force_show: bool=False):
        if checked or force_show:
            if self.info_window is None:
                self.info_window = InfoWindow(parent=self)
                # 把缓存日志刷进去
                if self._log_buffer:
                    self.info_window.append_log("\n".join(self._log_buffer))
                    self._log_buffer.clear()
                # 当前 Process Info 文本也刷一次
                if hasattr(self, "lab_count") and self.lab_count is not None:
                    self.info_window.set_info(self.lab_count.text())
                # 关闭时同步取消勾选
                self.info_window.destroyed.connect(self._on_info_window_closed)
            self.info_window.show()
            self.info_window.raise_()
            if hasattr(self, "chk_info_window") and not self.chk_info_window.isChecked():
                self.chk_info_window.setChecked(True)
        else:
            if self.info_window:
                self.info_window.close()
                self.info_window = None
            if hasattr(self, "chk_info_window") and self.chk_info_window.isChecked():
                self.chk_info_window.setChecked(False)

    def _on_info_window_closed(self, *args):
        # 窗口被用户手动关闭时，同步取消勾选
        self.info_window = None
        if hasattr(self, "chk_info_window"):
            self.chk_info_window.setChecked(False)

    def _update_hist_from_image(self):
        if self.last_img is None:
            self.hist_widget.set_hist(None)
            return
        img = self.last_img
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        self.hist_widget.set_hist(hist)

    def _apply_levels_to_preview(self, b, gamma, w):
        if self.last_img is None:
            return
        import cv2, numpy as np
        img = self.last_img

        # 工作在 float32，按通道处理（BGR）
        x = img.astype(np.float32)
        # 归一化到 0..1 的线性拉伸（先按阈截断）
        x = np.clip((x - b) / max(1.0, (w - b)), 0.0, 1.0)
        # γ 校正
        x = np.power(x, 1.0 / max(0.05, float(gamma)))

        out = (x * 255.0).clip(0,255).astype(np.uint8)
        # 只用于展示（不覆盖 last_img 原图）
        self.show_image(out)



    # ==== 预览覆盖：正在调色时锁住显示 ====
    def _set_preview_override(self, img_bgr: np.ndarray):
        """进入预览覆盖态：Display 只显示这个，不允许其它回调改画面。"""
        self._preview_override = img_bgr  # 只用于显示，不改 last_img
        self.show_image(img_bgr, reset_zoom=False)
        self._update_levels_histogram()

    def _clear_preview_override(self):
        """退出预览覆盖态：恢复正常绘制（seg/surv 等可以刷新画面）"""
        self._preview_override = None
        self._update_levels_histogram()

    def _apply_adjust_for_processing(self, img_bgr):
        """
        把 self._adj 中记录的调色参数一次性作用到原图，返回给模型/算法使用的图像。
        - 输入:  img_bgr (np.uint8), BGR 或灰度
        - 输出:  new_img (np.uint8), 同通道数
        - 顺序:  Levels -> RGB增益 -> 亮度/对比度 -> Gamma -> 饱和度/色相
        """
        import numpy as np, cv2

        if img_bgr is None:
            return None
        x = img_bgr

        # ---- 转 float32 便于计算 ----
        x = x.astype(np.float32)

        # 0) 读参数（容错：无 _adj 时给默认值）
        adj = getattr(self, "_adj", None) or {
            "bri": 0.0, "con": 1.0, "gamma": 1.0, "r": 1.0, "g": 1.0, "b": 1.0,
            "sat": 1.0, "hue": 0.0, "lvl_b": 0.0, "lvl_g": 1.0, "lvl_w": 255.0
        }
        bri = float(adj.get("bri", 0.0))
        con = float(adj.get("con", 1.0))
        gma = float(adj.get("gamma", 1.0))
        r_g = float(adj.get("r", 1.0))
        g_g = float(adj.get("g", 1.0))
        b_g = float(adj.get("b", 1.0))
        sat = float(adj.get("sat", 1.0))
        hue = float(adj.get("hue", 0.0))  # 单位：度
        lvl_b = float(adj.get("lvl_b", 0.0))
        lvl_g = float(adj.get("lvl_g", 1.0))
        lvl_w = float(adj.get("lvl_w", 255.0))

        # 1) Levels（先做线性拉伸，再做 gamma）
        #   x' = ((x - lvl_b) / (lvl_w - lvl_b))^ (1/lvl_g) * 255
        den = max(1.0, (lvl_w - lvl_b))  # 防零
        x = np.clip((x - lvl_b) / den, 0.0, 1.0)
        x = np.power(x, 1.0 / max(0.05, lvl_g)) * 255.0  # 防极小

        # 2) RGB 增益（仅在三通道图生效，灰度图跳过）
        if x.ndim == 3 and x.shape[2] == 3:
            # BGR 顺序
            x[..., 0] *= b_g
            x[..., 1] *= g_g
            x[..., 2] *= r_g

        # 3) 亮度/对比度： x = x * con + bri
        x = x * con + bri

        # 4) 全局 Gamma（二次 gamma，可选）
        x = np.clip(x, 0, 255) / 255.0
        x = np.power(x, 1.0 / max(0.05, gma)) * 255.0

        # 先裁剪回 0~255 再转回 uint8
        x = np.clip(x, 0, 255).astype(np.uint8)

        # 5) 饱和度/色相（仅对三通道 BGR）
        if x.ndim == 3 and x.shape[2] == 3:
            hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
            # OpenCV: H ∈ [0,179], S ∈ [0,255], V ∈ [0,255]
            hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)  # S
            hsv[..., 0] = (hsv[..., 0] + hue / 2.0) % 180.0  # H(度转OpenCV单位)
            x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return x

    # 色阶
    def _on_levels_changed_from_widget(self, b: int, g: float, w: int):
        # 记录
        self._adj["lvl_b"] = int(b)
        self._adj["lvl_g"] = float(g)
        self._adj["lvl_w"] = int(w)

        # 预览（仅覆盖显示，不改 last_img）
        if getattr(self, "last_img", None) is not None:
            prev = self._apply_adjust_for_processing(self.last_img)
            self._set_preview_override(prev)  # 你已有的方法：只换显示层
            self._update_levels_histogram(from_preview=True)  # 直方图也跟着预览图更新




    # # 色阶预览
    # def _apply_levels_to_preview(self, b, gamma, w):
    #     if self.last_img is None:
    #         return
    #     img = self.last_img.astype(np.float32)
    #     x = np.clip((img - b) / max(1.0, (w - b)), 0.0, 1.0)
    #     x = np.power(x, 1.0 / max(0.05, float(gamma)))
    #     out = (x * 255.0).clip(0, 255).astype(np.uint8)
    #     self._set_preview_override(out)
    #
    # # 亮度/对比度等预览
    # def _on_image_adjust(self, *_):
    #     if self.last_img is None:
    #         return
    #     adj = self._build_adjusted(self.last_img)
    #     self._set_preview_override(adj)

    # --- 显示display图的色阶bar图
    # Main 类中添加：
    def _get_current_display_image(self):
        """
        返回当前 display 上看的那张图：
        - 如果有覆盖预览(self._preview_override)，优先用它（即调色后的图）
        - 否则退回原图 self.last_img
        """
        if getattr(self, "_preview_override", None) is not None:
            return self._preview_override
        return getattr(self, "last_img", None)

    def _update_levels_histogram(self, from_preview: bool = True):
        """
        用“当前显示图像”更新色阶控件的直方图。
        """
        if not hasattr(self, "levels_widget"):
            return

        img = self._get_current_display_image()
        if img is None:
            self.levels_widget.clear_histogram()
            return

        import cv2, numpy as np
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 计算原始直方图（counts），喂给控件
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
        self.levels_widget.set_histogram(hist)

        # # 可选：1% / 99% 建议挡位
        # cdf = hist.cumsum()
        # if cdf[-1] > 0:
        #     cdf = cdf / float(cdf[-1])
        #     b_suggest = int(np.searchsorted(cdf, 0.01))
        #     w_suggest = int(np.searchsorted(cdf, 0.99))
        #     g_suggest = 1.0
        #     # 只更新 UI，不触发回调（避免死循环）
        #     self.levels_widget.set_levels(b_suggest, g_suggest, w_suggest, emit_signal=False)

    def _do_update_levels_histogram(self):
        if not hasattr(self, "levels_widget"):
            return
        img = self._get_current_display_image()
        if img is None:
            self.levels_widget.clear_histogram()
            return

        # 计算灰度直方图（尽量避免重复拷贝）
        import cv2, numpy as np
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)

        # 静默喂给控件（控件内部也有 16ms debounce）
        self.levels_widget.set_histogram(hist)

    def _on_toggle_edit_mode(self, checked: bool):
        self.edit_mode = bool(checked)
        if self.edit_mode:
            # 提示：编辑只对 seg 结果有效
            if self.last_mode != "seg" or self.last_pred is None:
                self.append_log("[Edit] 请先运行分割（Seg），再开启编辑模式。")
            # 改鼠标形状
            self.image_label.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.image_label.setCursor(QtCore.Qt.CursorShape.ArrowCursor)


    #规范化函数
    def _ensure_masks_NHW(self, masks_any, disp_size=None):
        """
        把各种形状的掩码统一成 (N, H, W)，并确保 dtype=uint8 (0/255)。
        支持：list/ndarray，单mask/多mask，(H,W)、(1,H,W)、(H,W,1)、(C,H,W)、(H,W,C) 等。
        如果提供 disp_size=(H,W)，会在需要时用最近邻 resize。
        """
        import numpy as np, cv2

        if masks_any is None:
            return None

        # 把输入变成 list
        if isinstance(masks_any, np.ndarray):
            # 可能是 (N, H, W) / (N,1,H,W) / (H,W) / (1,H,W)
            if masks_any.ndim == 2:
                masks_list = [masks_any]
            elif masks_any.ndim == 3:
                # (N,H,W) 或 (H,W,C)
                if masks_any.shape[0] <= 4 and masks_any.shape[-1] not in (1, 3):  # 粗略判断是不是 (C,H,W)
                    # 形如 (C,H,W) -> 拆成 C 个
                    masks_list = [masks_any[c, ...] for c in range(masks_any.shape[0])]
                else:
                    # (N,H,W) 直接拆
                    masks_list = [masks_any[i, ...] for i in range(masks_any.shape[0])]
            elif masks_any.ndim == 4:
                # (N,1,H,W) / (N,H,W,1) / (N,C,H,W) / (N,H,W,C)
                N = masks_any.shape[0]
                masks_list = []
                for i in range(N):
                    m = masks_any[i]
                    # 去掉所有 size=1 的通道
                    m = np.squeeze(m)
                    masks_list.append(m)
            else:
                # 其它情况直接返回
                return masks_any
        else:
            # list：逐个 squeeze
            masks_list = []
            for m in masks_any:
                a = np.asarray(m)
                a = np.squeeze(a)
                masks_list.append(a)

        # 统一到 (H,W) + uint8 0/255
        norm_list = []
        for m in masks_list:
            a = np.asarray(m)
            if a.ndim == 3:
                # 若还有多通道，取任意非零（你也可以改成 a.max(axis=2)）
                a = a.max(axis=-1)
            if a.dtype != np.uint8:
                if np.issubdtype(a.dtype, np.floating):
                    a = (a > 0.5).astype(np.uint8) * 255
                else:
                    a = (a != 0).astype(np.uint8) * 255
            else:
                a = (a > 0).astype(np.uint8) * 255

            # 需要的话 resize 到 disp_size
            if disp_size is not None and a.shape[:2] != tuple(disp_size):
                H, W = disp_size
                a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)

            norm_list.append(a)

        # stack -> (N,H,W)
        try:
            return np.stack(norm_list, axis=0)
        except Exception:
            return norm_list


    # 点击选中
    def _on_image_click_select(self, ix: int, iy: int):
        """点击显示区：按“到 bbox 的最近距离”选择实例；选中后仅高亮（不删除）。"""
        if not self.edit_mode or self.last_mode != "seg" or self.last_pred is None:
            return

        # 取显示尺寸，准备 masks_disp（命中调试备用）
        disp_size = self.last_pred.get("disp_size", self.last_img.shape[:2])
        masks_disp = self._ensure_masks_NHW(self.last_pred.get("masks_disp"), disp_size=disp_size)
        H, W = disp_size
        if not (0 <= ix < W and 0 <= iy < H):
            return

        boxes = self.last_pred.get("boxes", None)
        scores = self.last_pred.get("scores", None)

        sel = None

        # ---- 基于 bbox 的“最近距离”选择 ----
        if boxes is not None and len(boxes) > 0:
            # 计算每个 bbox 到点击点的距离（矩形外欧氏距离；在框内则为 0）
            best_d = float("inf");
            best_i = None

            def _dist_point_to_rect(x, y, x1, y1, x2, y2):
                # 若点在矩形内，dx=dy=0；否则到边的曼哈顿投影后算欧氏距离
                dx = 0.0 if x1 <= x <= x2 else (x1 - x if x < x1 else x - x2)
                dy = 0.0 if y1 <= y <= y2 else (y1 - y if y < y1 else y - y2)
                return math.hypot(dx, dy)

            n = boxes.shape[0] if hasattr(boxes, "shape") else len(boxes)
            for i in range(n):
                x1, y1, x2, y2 = map(float, boxes[i])
                # 裁边以免负值干扰
                x1 = max(0.0, min(float(W - 1), x1));
                x2 = max(0.0, min(float(W - 1), x2))
                y1 = max(0.0, min(float(H - 1), y1));
                y2 = max(0.0, min(float(H - 1), y2))
                if x2 < x1 or y2 < y1:
                    continue
                d = _dist_point_to_rect(ix, iy, x1, y1, x2, y2)
                # 如需“分数优先”可在距离相同情况下用分数破平
                if d < best_d - 1e-6 or (abs(d - best_d) <= 1e-6 and scores is not None and i < len(
                        scores) and best_i is not None and i != best_i and float(scores[i]) > float(scores[best_i])):
                    best_d, best_i = d, i

            sel = best_i

        # ---- 兜底：若没有 boxes（极少数情况），再尝试 mask 命中 ----
        if sel is None and masks_disp is not None:
            if isinstance(masks_disp, np.ndarray):
                for i in range(masks_disp.shape[0]):
                    if masks_disp[i, iy, ix] > 0:
                        sel = i;
                        break
            else:
                for i, m in enumerate(masks_disp):
                    if m[iy, ix] > 0:
                        sel = i;
                        break

        # 结果处理
        if sel is None:
            self._selected_inst = None
            self.set_process_info("Process Info (Seg): 未命中任何实例")
            self._render_overlay_with_selection(None)
            return

        self._selected_inst = int(sel)
        self.set_process_info(f"Process Info (Seg): 选中实例 #{sel}")
        self._render_overlay_with_selection(sel)

        # 抢回主窗焦点，确保 Delete 被 keyPressEvent 接收
        self.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)

    def _render_overlay_with_selection(self, sel_idx: Optional[int]):
        frame = self.last_pred.get("frame", self.last_img)
        boxes = self.last_pred.get("boxes", None)
        scores = self.last_pred.get("scores", None)
        classes = self.last_pred.get("classes", None)
        masks = self.last_pred.get("masks", None)

        alp = self.slider_alpha.value() / 100.0 if hasattr(self, "slider_alpha") else 0.5
        score_thr = float(self.seg_score_thr) if hasattr(self, "seg_score_thr") else 0.0

        # 先按原逻辑渲染（可能自带绿色框）
        overlay = draw_overlay(self.last_img, frame, boxes, scores, classes, masks,
                               score_thr=score_thr, alpha=alp, draw_box=False)

        if sel_idx is not None:
            H, W = self.last_img.shape[:2]

            # ① 若有 mask，叠加黄色半透明以示高亮
            masks_disp = self._ensure_masks_NHW(self.last_pred.get("masks_disp"),
                                                disp_size=self.last_pred.get("disp_size", (H, W)))
            if isinstance(masks_disp, np.ndarray) and 0 <= sel_idx < masks_disp.shape[0]:
                m = (masks_disp[sel_idx] > 0).astype(np.uint8)
                if np.any(m):
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 3)  # 黄框
                    # 半透明黄填充，强化区别
                    overlay[m.astype(bool)] = (
                            overlay[m.astype(bool)].astype(np.float32) * 0.70
                            + np.array([0, 255, 255], dtype=np.float32) * 0.30
                    ).astype(np.uint8)

            # 不管有没有 mask，都在 bbox 上再画一次“黄色矩形”覆盖原先的绿框
            if boxes is not None and 0 <= sel_idx < (boxes.shape[0] if hasattr(boxes, "shape") else len(boxes)):
                x1, y1, x2, y2 = [int(round(v)) for v in boxes[sel_idx]]
                x1 = max(0, min(W - 1, x1));
                x2 = max(0, min(W - 1, x2))
                y1 = max(0, min(H - 1, y1));
                y2 = max(0, min(H - 1, y2))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 黄框覆盖

        self.show_image(overlay, reset_zoom=False)


    # 删除选中实例
    def keyPressEvent(self, e: QtGui.QKeyEvent):
        """Delete/Backspace 删除当前选中实例；Esc 取消选中。"""
        if e.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            self._delete_selected_instance()
            return
        if e.key() == QtCore.Qt.Key.Key_Escape:
            self._selected_inst = None
            if self.last_mode == "seg" and self.last_pred is not None:
                self._render_overlay_with_selection(None)
            self.set_process_info("Process Info (Seg): cancel selection")
            return
        super().keyPressEvent(e)

    # 从后端删除annotation记录
    def _delete_selected_instance(self):
        """真正执行删除，并重绘。"""
        if self._selected_inst is None or self.last_mode != "seg" or self.last_pred is None:
            return

        k = int(self._selected_inst)

        def _safe_del(arr):
            if arr is None: return None
            try:
                return np.delete(arr, k, axis=0)
            except Exception:
                if isinstance(arr, list): return arr[:k] + arr[k + 1:]
                return arr

        self.last_pred["masks"] = _safe_del(self.last_pred.get("masks"))
        self.last_pred["masks_disp"] = _safe_del(self.last_pred.get("masks_disp"))
        self.last_pred["boxes"] = _safe_del(self.last_pred.get("boxes"))
        self.last_pred["scores"] = _safe_del(self.last_pred.get("scores"))
        self.last_pred["classes"] = _safe_del(self.last_pred.get("classes"))

        self._selected_inst = None
        self._render_overlay_with_selection(None)

        bx = self.last_pred.get("boxes")
        n_now = (bx.shape[0] if (bx is not None and hasattr(bx, "shape")) else (len(bx) if isinstance(bx, list) else 0))
        self.set_process_info(f"Process Info (Seg): {n_now} objects")
        self.append_log(f"[Edit] already deleted instance, remain {n_now}")


    # box 统一放大回原图坐标
    def _rescale_boxes(self, boxes, from_hw, to_hw):
        """boxes: (N,4) [x1,y1,x2,y2]，把从 from_hw(H,W) 坐标系缩放到 to_hw(H,W)。"""
        if boxes is None:
            return None
        if hasattr(boxes, "shape"):
            b = boxes.astype(np.float32).copy()
        else:
            b = np.asarray(boxes, dtype=np.float32).copy()

        H0, W0 = from_hw
        H1, W1 = to_hw
        if H0 == 0 or W0 == 0:  # 保险
            return b

        sx = float(W1) / float(W0)
        sy = float(H1) / float(H0)
        # [x1, y1, x2, y2]
        b[:, [0, 2]] *= sx
        b[:, [1, 3]] *= sy
        return b

    def _open_label_editor(self):
        if self.last_pred is None or self.last_mode != "seg":
            QtWidgets.QMessageBox.information(self, "Edit Labels", "Please do segmentation first.")
            return

        boxes  = self.last_pred.get("boxes", None)
        scores = self.last_pred.get("scores", None)
        classes= self.last_pred.get("classes", None)

        # 类别名（可选）
        id2name = None
        if hasattr(self, "seg_seg") and hasattr(self.seg_seg, "class_names"):
            cn = self.seg_seg.class_names
            if isinstance(cn, dict):
                id2name = {int(k): str(v) for k, v in cn.items()}
            elif isinstance(cn, (list, tuple)):
                id2name = {i: str(n) for i, n in enumerate(cn)}

        dlg = LabelEditorDialog(self, boxes, scores, classes, id2name=id2name)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_ids, _ = dlg.get_labels()
            # 回写到 last_pred
            if classes is not None and len(new_ids) == len(classes):
                self.last_pred["classes"] = np.asarray(new_ids, dtype=classes.dtype if hasattr(classes, "dtype") else np.int32)

                # 刷新显示（根据新 label 重绘一次）
                alp = self.slider_alpha.value() / 100.0 if hasattr(self, "slider_alpha") else 0.5
                score_thr = float(self.seg_score_thr) if hasattr(self, "seg_score_thr") else 0.0
                overlay = draw_overlay(
                    self.last_img,
                    self.last_pred.get("frame", self.last_img),
                    self.last_pred.get("boxes", None),
                    self.last_pred.get("scores", None),
                    self.last_pred.get("classes", None),
                    self.last_pred.get("masks", None),
                    score_thr=score_thr, alpha=alp, draw_box=False
                )
                self.show_image(overlay)
                self.append_log("[Labels] already renew")

    def _ensure_brush_buffer(self):
        if self.last_img is None:
            return False
        H, W = self.last_img.shape[:2]
        if self._brush_mask is None or self._brush_mask.shape[:2] != (H, W):
            import numpy as np
            self._brush_mask = np.zeros((H, W), dtype=np.uint8)
        return True

    def _on_toggle_brush(self, checked: bool):
        # 切换 ZoomableLabel 的模式；编辑 & 画笔可同时存在，但交互优先画笔
        self.image_label.brush_mode = bool(checked)
        if checked:
            if not self._ensure_brush_buffer():
                self.chk_brush.setChecked(False)
                return
            self.append_log("[Brush] 画笔已开启：左键绘制，Alt/右键擦除；中键可平移。")
        else:
            self.append_log("[Brush] 画笔已关闭。")
        self._on_brush_preview_refresh()

    # def _on_brush_point(self, ix: int, iy: int, is_erasing: bool):
    #     """收到一枚图像坐标点，按画笔半径在 _brush_mask 上画/擦。"""
    #     if not self._ensure_brush_buffer():
    #         return
    #     r = max(1, int(self.slider_brush.value()) if hasattr(self, "slider_brush") else 12)
    #     if is_erasing:
    #         # 擦除就画黑色实心圆
    #         cv2.circle(self._brush_mask, (ix, iy), r, 0, thickness=-1, lineType=cv2.LINE_8)
    #     else:
    #         # 绘制就画白色实心圆
    #         cv2.circle(self._brush_mask, (ix, iy), r, 255, thickness=-1, lineType=cv2.LINE_8)
    #
    #     self._on_brush_preview_refresh()

    def _on_brush_preview_refresh(self, *_):
        """把画笔层以半透明叠加到当前显示（仅预览，不改 last_img/last_pred）。"""
        if self.last_img is None:
            return
        # 基底：当前模式下的 overlay（如果有seg/surv结果按你现有逻辑重绘一次）
        base = self.last_img.copy()
        if self.last_pred is not None:
            # 复用你已有的“基于 last_pred 重绘”逻辑（这里给出最简单的：seg 时重画一次）
            if self.last_mode == "seg":
                alp = self.slider_alpha.value()/100.0 if hasattr(self, "slider_alpha") else 0.5
                score_thr = float(self.seg_score_thr) if hasattr(self, "seg_score_thr") else 0.0
                base = draw_overlay(self.last_img,
                                    self.last_pred.get("frame", self.last_img),
                                    self.last_pred.get("boxes", None),
                                    self.last_pred.get("scores", None),
                                    self.last_pred.get("classes", None),
                                    self.last_pred.get("masks", None),
                                    score_thr=score_thr, alpha=alp, draw_box=False)

            elif self.last_mode == "surv":
                # 简化：直接使用 _on_params_changed 的那一套重画也可以
                pass

        if self._brush_mask is not None:
            # 半透明青色叠加画笔层
            alpha = float(self._brush_preview_alpha)
            idx = self._brush_mask > 0
            if np.any(idx):
                base[idx] = (base[idx].astype(np.float32) * (1 - alpha) +
                             np.array([255, 200, 0], dtype=np.float32) * alpha).astype(np.uint8)
                # 可选：描边
                cnts, _ = cv2.findContours((self._brush_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    cv2.drawContours(base, [c], -1, (255, 200, 0), 2)

        self.show_image(base, reset_zoom=False)

    def _on_brush_clear(self):
        if self._brush_mask is not None:
            self._brush_mask.fill(0)
            self._on_brush_preview_refresh()
            self.append_log("[Brush] 画笔层已清空。")


    # 提交绘制的实例
    def _on_brush_commit(self):
        if self.last_mode != "seg":
            self.append_log("[Brush] 仅在 Seg 模式下可提交。")
            return
        if getattr(self, "_brush_mask_full", None) is None:
            self.append_log("[Brush] 当前没有可提交的画笔掩码。")
            return

        H, W = self.last_img.shape[:2]
        target_hw = (H, W)

        # 取旧字段
        old_masks = self.last_pred.get("masks", None)
        old_boxes = self.last_pred.get("boxes", None)
        old_scores = self.last_pred.get("scores", None)
        old_cls = self.last_pred.get("classes", None)

        # 先把旧的所有 mask 统一成整图 (H,W)
        full_list = []
        if old_masks is not None:
            if isinstance(old_masks, np.ndarray) and old_masks.ndim == 3 and old_masks.shape[1:] == (H, W):
                # 已经是 (N,H,W) 的整图
                for i in range(old_masks.shape[0]):
                    full_list.append((old_masks[i] > 0).astype(np.uint8) * 255)
            else:
                # 可能是 list 或 ROI 大小
                N = len(old_masks) if isinstance(old_masks, (list, tuple)) else (
                    old_masks.shape[0] if hasattr(old_masks, "shape") else 0)
                for i in range(N):
                    m = old_masks[i] if isinstance(old_masks, (list, tuple)) else old_masks[i, ...]
                    b = None
                    if old_boxes is not None and i < len(old_boxes):
                        b = old_boxes[i]
                    full_list.append(self._mask_to_fullsize(m, b, target_hw))

        # 追加新画笔实例（确保也是整图大小）
        brush_full = (self._brush_mask_full > 0).astype(np.uint8) * 255
        full_list.append(brush_full)

        # 尝试 stack 成 (N,H,W)
        masks_full = np.stack(full_list, axis=0) if full_list else None

        # 统一后重算 boxes
        boxes_new = self._recompute_boxes_from_full_masks(masks_full) if masks_full is not None else None

        # 补齐 scores / classes
        N_old = 0 if old_boxes is None else (old_boxes.shape[0] if hasattr(old_boxes, "shape") else len(old_boxes))
        score_brush = 1.0
        cls_brush = int(getattr(self, "_brush_class_id", 0))

        if old_scores is None:
            scores_new = np.array([1.0] * N_old + [score_brush], dtype=np.float32)
        else:
            scores_new = np.concatenate([np.asarray(old_scores).astype(np.float32),
                                         np.array([score_brush], dtype=np.float32)], axis=0)

        if old_cls is None:
            classes_new = np.array([0] * N_old + [cls_brush], dtype=np.int32)
        else:
            classes_new = np.concatenate([np.asarray(old_cls).astype(np.int32),
                                          np.array([cls_brush], dtype=np.int32)], axis=0)

        # 更新 last_pred
        self.last_pred["masks"] = masks_full  # (N,H,W) uint8
        self.last_pred["boxes"] = boxes_new  # (N,4) float32
        self.last_pred["scores"] = scores_new
        self.last_pred["classes"] = classes_new

        # 同步命中检测用的 masks_disp（直接用整图即可）
        self.last_pred["masks_disp"] = masks_full.copy() if masks_full is not None else None
        self.last_pred["disp_size"] = (H, W)

        # 重绘
        alp = self.slider_alpha.value() / 100.0 if hasattr(self, "slider_alpha") else 0.5
        score_thr = float(self.seg_score_thr) if hasattr(self, "seg_score_thr") else 0.0
        frame_ref = self.last_pred.get("frame", None)  # 允许 None，draw_overlay 内已支持
        overlay = draw_overlay(self.last_img, frame_ref,
                               self.last_pred["boxes"],
                               self.last_pred["scores"],
                               self.last_pred["classes"],
                               self.last_pred["masks"],
                               score_thr=score_thr, alpha=alp, draw_box=False, coords="orig")
        self.show_image(overlay)

        n_now = self.last_pred["boxes"].shape[0] if self.last_pred["boxes"] is not None else 0
        self.set_process_info(f"Process Info (Seg): {n_now} objects")
        self.append_log("[Brush] 已提交画笔实例并重绘。")


    def _mask_to_fullsize(self, m, box, target_hw):
        """
        把任意形状/通道的 mask 变成整图 (H,W) 的 uint8 二值(0/255)。
        - m: np.ndarray, 可能是 (H,W), (H,W,1), ROI 尺寸, 或者 float/bool 类型
        - box: [x1,y1,x2,y2] 或 None
        - target_hw: (H,W) 目标整图尺寸（通常是 self.last_img.shape[:2]）
        """
        H, W = target_hw
        a = np.asarray(m)
        a = np.squeeze(a)

        # -> 二值 uint8
        if a.dtype != np.uint8:
            if np.issubdtype(a.dtype, np.floating):
                a = (a > 0.5).astype(np.uint8) * 255
            else:
                a = (a != 0).astype(np.uint8) * 255
        else:
            a = (a > 0).astype(np.uint8) * 255

        full = np.zeros((H, W), dtype=np.uint8)

        if box is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(W-1, x1)); x2 = max(0, min(W, x2))
            y1 = max(0, min(H-1, y1)); y2 = max(0, min(H, y2))
            bw, bh = max(0, x2-x1), max(0, y2-y1)
            if bw > 0 and bh > 0:
                # 把 a 调整到 bbox 尺寸再贴回
                if a.shape[:2] != (bh, bw):
                    a_roi = cv2.resize(a, (bw, bh), interpolation=cv2.INTER_NEAREST)
                else:
                    a_roi = a
                full[y1:y2, x1:x2] = (a_roi > 0).astype(np.uint8) * 255
                return full

        # 没有 box 或者 box 无效：若 a 与整图同尺寸，直接用，否则整体 resize 到整图
        if a.shape[:2] == (H, W):
            return a
        else:
            return cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)


    def _recompute_boxes_from_full_masks(self, masks_full):
        """
        从整图 (N,H,W) 二值 mask 重新计算 boxes（外接矩形）。
        返回 float32 的 (N,4) xyxy；若某个实例为空，则给零框。
        """
        if isinstance(masks_full, list):
            arr = np.stack(masks_full, axis=0) if masks_full else np.zeros((0, 0, 0), np.uint8)
        else:
            arr = masks_full

        N = 0 if arr.ndim != 3 else arr.shape[0]
        boxes = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            m = (arr[i] > 0).astype(np.uint8)
            if m.any():
                ys, xs = np.where(m > 0)
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                x1, x2 = int(xs.min()), int(xs.max()) + 1
                boxes[i] = [x1, y1, x2, y2]
            else:
                boxes[i] = [0, 0, 0, 0]
        return boxes

    def _refresh_overlay_from_masks(self):
        """用当前 full masks 重画 overlay（无 box）"""
        pred = self.last_pred
        if pred is None:
            return

        masks = pred.get("masks", None)
        boxes = pred.get("boxes", None)
        classes = pred.get("classes", None)
        scores = pred.get("scores", None)

        alp = self.slider_alpha.value() / 100.0
        overlay = draw_overlay(
            self.last_img, None, boxes, scores, classes, masks,
            score_thr=0.0, alpha=alp, draw_box=False, coords="orig"
        )
        self.show_image(overlay)
