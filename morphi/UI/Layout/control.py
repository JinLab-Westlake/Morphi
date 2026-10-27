# UI/Layout/control.py
from PyQt6 import QtWidgets, QtCore
from UI.Component.ui_widgets import HistogramLevelsWidget
from UI.Helpers.toggle_exclusive import TwoCheckExclusive


# 导航枚举
PAGE_HOME, PAGE_SEG, PAGE_SURV, PAGE_EDGE, PAGE_PROJ = range(5)


from PyQt6 import QtWidgets, QtCore

def _build_left_panel(main) -> QtWidgets.QWidget:
    # === 用滚动区承载左侧内容 ===
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)           # 根据可视区域自动收缩/扩展
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    # scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)  # 需要无边框可开

    # 真正的左侧容器（放原来的控件）
    left = QtWidgets.QWidget()
    l_lay = QtWidgets.QVBoxLayout(left)
    l_lay.setContentsMargins(6, 6, 6, 6)
    l_lay.setSpacing(8)

    # --- color 黑白平衡 ---
    box_levels = QtWidgets.QGroupBox("Color Levels")
    vlv = QtWidgets.QVBoxLayout(box_levels)
    vlv.setContentsMargins(10, 6, 8, 6)
    vlv.setSpacing(10)

    main.levels_widget = HistogramLevelsWidget()
    # main.hist_widget = main.levels_widget  # 兼容老名字
    vlv.addWidget(main.levels_widget)
    # main.levels_widget.set_levels(0, 1.0, 255, emit_signal=False)



    # def _on_levels_changed(b, g, w):
    #     main._apply_levels_to_preview(b, g, w)
    # UI/Layout/control.py 里的 _build_left_panel(main) 内
    def _on_levels_changed(b, g, w):
        # 1) 记录
        if not hasattr(main, "_adj"):
            main._adj = {"bri": 0.0, "con": 1.0, "gamma": 1.0, "r": 1.0, "g": 1.0, "b": 1.0,
                         "sat": 1.0, "hue": 0.0, "lvl_b": 0.0, "lvl_g": 1.0, "lvl_w": 255.0}
        main._adj["lvl_b"] = float(b)
        main._adj["lvl_g"] = float(g)
        main._adj["lvl_w"] = float(w)

        # 2) 预览（不改 last_img，只覆盖显示层）
        if getattr(main, "last_img", None) is not None:
            prev = main._apply_adjust_for_processing(main.last_img)
            main._set_preview_override(prev)  # 只改变显示，不持久化

    main.levels_widget.levelsChanged.connect(_on_levels_changed)

    l_lay.addWidget(box_levels)

    # ===== Mask & Brush Tools 综合面板 =====
    box_tool = QtWidgets.QGroupBox("Mask editor")
    vtool = QtWidgets.QVBoxLayout(box_tool)
    vtool.setContentsMargins(10, 6, 8, 6)
    vtool.setSpacing(6)

    # --- Edit mode ---
    main.chk_edit_mode = QtWidgets.QCheckBox("Delete mask")
    main.chk_edit_mode.setToolTip("After opening it, click on a mask on the right image to delete the instance.")
    vtool.addWidget(main.chk_edit_mode)

    # --- Mask transparency ---
    row_alpha = QtWidgets.QHBoxLayout()
    row_alpha.addWidget(QtWidgets.QLabel("Mask Transparency:"))
    main.slider_alpha = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    main.slider_alpha.setRange(0, 100)
    main.slider_alpha.setValue(35)
    main.slider_alpha.valueChanged.connect(main._on_params_changed)
    row_alpha.addWidget(main.slider_alpha)
    vtool.addLayout(row_alpha)

    # --- Enable Brush ---
    main.chk_brush = QtWidgets.QCheckBox("Draw mask")
    vtool.addWidget(main.chk_brush)


    # --- 互斥锁 ---
    def _edit_logic(on: bool):
        if hasattr(main, "_on_toggle_edit_mode"):
            main._on_toggle_edit_mode(on)

    def _brush_logic(on: bool):
        if hasattr(main, "brush_editor") and main.brush_editor:
            (main.brush_editor.enable() if on else main.brush_editor.disable())

    main._edit_brush_ex = TwoCheckExclusive(
        main.chk_edit_mode,
        main.chk_brush,
        on_edit_toggled=_edit_logic,
        on_brush_toggled=_brush_logic,
    )
    main._edit_brush_ex.init_state()



    # --- Brush Size ---
    lab_br = QtWidgets.QLabel("Brush Size:")
    main.slider_brush = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    main.slider_brush.setRange(1, 50)
    main.slider_brush.setValue(12)
    vtool.addWidget(lab_br)
    vtool.addWidget(main.slider_brush)

    # --- Commit & Clear ---
    hbtn = QtWidgets.QHBoxLayout()
    main.btn_brush_commit = QtWidgets.QPushButton("Commit")
    main.btn_brush_clear = QtWidgets.QPushButton("Clear")
    hbtn.addWidget(main.btn_brush_commit)
    hbtn.addWidget(main.btn_brush_clear)
    vtool.addLayout(hbtn)

    # 添加到左栏
    l_lay.addWidget(box_tool)


    # 居中堆叠容器
    main.stack_left = AutoSizeStack()
    v_center = QtWidgets.QVBoxLayout()
    v_center.addStretch(1)
    v_center.addWidget(main.stack_left)
    v_center.addStretch(1)
    l_lay.addLayout(v_center, 1)

    # 5个页面
    page_home = _build_page_home(main)
    page_seg  = _build_page_seg(main)
    page_surv = _build_page_surv(main)
    page_edge = _build_page_edge(main)
    page_proj = _build_page_proj(main)
    main.stack_left.addWidget(page_home)
    main.stack_left.addWidget(page_seg)
    main.stack_left.addWidget(page_surv)
    main.stack_left.addWidget(page_edge)
    main.stack_left.addWidget(page_proj)
    main.stack_left.setCurrentIndex(PAGE_HOME)

    # ===== Image List 面板 =====
    box_list = QtWidgets.QGroupBox("Image List")
    list_lay = QtWidgets.QVBoxLayout(box_list)

    main.list_images = QtWidgets.QListWidget()
    main.list_images.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    main.list_images.itemSelectionChanged.connect(main._preview_selected_image)
    list_lay.addWidget(main.list_images, 1)

    row_btns = QtWidgets.QHBoxLayout()
    btn_add = QtWidgets.QPushButton("Select")
    btn_add.clicked.connect(main._add_images_to_list)
    btn_remove = QtWidgets.QPushButton("Remove")
    btn_remove.clicked.connect(main._remove_selected_images)
    btn_clear = QtWidgets.QPushButton("Clear")
    btn_clear.clicked.connect(main._clear_images_list)
    row_btns.addWidget(btn_add)
    row_btns.addWidget(btn_remove)
    row_btns.addWidget(btn_clear)
    row_btns.addStretch(1)
    list_lay.addLayout(row_btns)

    # 让列表区域在垂直方向尽量扩展
    box_list.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Expanding)
    l_lay.addWidget(box_list, 1)

    main.list_images.setMinimumHeight(200)

    # 塞个弹簧，顶部内容更紧凑，滚动更自然
    l_lay.addStretch(1)

    # 将 left 放进滚动区
    scroll.setWidget(left)

    # 保持整体宽度体验
    left.setMinimumWidth(180)
    scroll.setMinimumWidth(200)  # 视需要微调

    return scroll





class AutoSizeStack(QtWidgets.QStackedWidget):
    def sizeHint(self) -> QtCore.QSize:
        w = self.currentWidget()
        if w is None:
            return super().sizeHint()
        # 用当前页的 sizeHint
        sh = w.sizeHint()
        # 加上自身内容边距
        return sh

    def minimumSizeHint(self) -> QtCore.QSize:
        w = self.currentWidget()
        if w is None:
            return super().minimumSizeHint()
        return w.minimumSizeHint()




def _build_page_home(main) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    # lay.setSpacing(5)

    title = QtWidgets.QLabel("Choose a task")
    title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    title.setStyleSheet("font-size:16pt; font-weight:600; margin:6px 0;")
    lay.addWidget(title)

    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(10)

    def _big_button(text, page_idx):
        btn = QtWidgets.QPushButton(text)
        btn.setMinimumHeight(48)
        btn.setStyleSheet("font-size:12pt;")
        btn.clicked.connect(lambda: _goto(main, page_idx))
        return btn

    grid.addWidget(_big_button("Segmentation", PAGE_SEG),  0, 0)
    grid.addWidget(_big_button("Survival analysis", PAGE_SURV), 0, 1)
    grid.addWidget(_big_button("Edge clustering", PAGE_EDGE),   1, 0)
    grid.addWidget(_big_button("Projection", PAGE_PROJ),       1, 1)

    lay.addLayout(grid)
    lay.addStretch(1)
    return w

def _goto(main, page_idx: int):
    main.stack_left.setCurrentIndex(page_idx)
    # 关键：让上层布局重新计算
    main.stack_left.updateGeometry()



def _toolbar_with_back(main, title_text: str, back_to=PAGE_HOME):
    bar = QtWidgets.QHBoxLayout()
    btn_back = QtWidgets.QToolButton()
    btn_back.setText("← Back")
    btn_back.clicked.connect(lambda: _goto(main, back_to))
    lab = QtWidgets.QLabel(title_text)
    lab.setStyleSheet("font-weight:600;")
    bar.addWidget(btn_back)
    bar.addSpacing(6)
    bar.addWidget(lab)
    bar.addStretch(1)
    return bar


def _build_page_seg(main) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.addLayout(_toolbar_with_back(main, "Segmentation"))

    # ======  Segmentation UI ======
    box_seg = QtWidgets.QGroupBox("Segmentation")
    seg_lay = QtWidgets.QVBoxLayout(box_seg)

    # 选择模型
    row_seg_top = QtWidgets.QHBoxLayout()
    row_seg_top.addWidget(QtWidgets.QLabel("Model:"))
    main.combo_seg = QtWidgets.QComboBox()
    main.combo_seg.addItems(list(main.models_seg.keys()))
    main.combo_seg.currentTextChanged.connect(lambda _: main._on_combo_changed("seg"))
    row_seg_top.addWidget(main.combo_seg, 1)

    # 管理模型按钮
    row_seg_btns = QtWidgets.QHBoxLayout()
    btn_seg_add = QtWidgets.QPushButton("Add…");     btn_seg_add.clicked.connect(lambda: main._add_model_from_file("seg"))
    btn_seg_del = QtWidgets.QPushButton("Remove");   btn_seg_del.clicked.connect(lambda: main._remove_selected_model("seg"))
    btn_seg_set = QtWidgets.QPushButton("Set current"); btn_seg_set.clicked.connect(lambda: main._set_current_model("seg"))
    row_seg_btns.addWidget(btn_seg_add); row_seg_btns.addWidget(btn_seg_del); row_seg_btns.addWidget(btn_seg_set)

    main.lab_model_seg = QtWidgets.QLabel("No segmentation model loaded")
    main.lab_model_seg.setWordWrap(True)
    main.lab_model_seg.setStyleSheet("color: grey;")

    main.btn_run_seg = QtWidgets.QPushButton("Run Segmentation")
    main.btn_run_seg.setEnabled(False)
    main.btn_run_seg.clicked.connect(main._run_seg)

    # 参数表格
    params_form = QtWidgets.QGridLayout()
    params_form.setHorizontalSpacing(10); params_form.setVerticalSpacing(8)

    main.spin_score = QtWidgets.QDoubleSpinBox(); main.spin_score.setRange(0.0,1.0); main.spin_score.setDecimals(3)
    main.spin_score.setSingleStep(0.05); main.spin_score.setValue(main.seg_score_thr)
    main.spin_score.valueChanged.connect(lambda v: _on_param_changed(main, "seg_score_thr", float(v)))
    params_form.addWidget(QtWidgets.QLabel("Score thr"), 0, 0); params_form.addWidget(main.spin_score, 0, 1)

    main.spin_nms = QtWidgets.QDoubleSpinBox(); main.spin_nms.setRange(0.0,1.0); main.spin_nms.setDecimals(3)
    main.spin_nms.setSingleStep(0.05); main.spin_nms.setValue(main.seg_nms_iou)
    main.spin_nms.valueChanged.connect(lambda v: _on_param_changed(main, "seg_nms_iou", float(v)))
    params_form.addWidget(QtWidgets.QLabel("NMS IoU"), 0, 2); params_form.addWidget(main.spin_nms, 0, 3)

    main.spin_min_area = QtWidgets.QSpinBox(); main.spin_min_area.setRange(0, 1_000_000_000)
    main.spin_min_area.setSingleStep(10); main.spin_min_area.setValue(main.seg_min_area)
    main.spin_min_area.valueChanged.connect(lambda v: _on_param_changed(main, "seg_min_area", int(v)))
    params_form.addWidget(QtWidgets.QLabel("Min area"), 1, 0); params_form.addWidget(main.spin_min_area, 1, 1)

    main.spin_top_k = QtWidgets.QSpinBox(); main.spin_top_k.setRange(0, 100000)
    main.spin_top_k.setSingleStep(1); main.spin_top_k.setValue(main.seg_top_k)
    main.spin_top_k.valueChanged.connect(lambda v: _on_param_changed(main, "seg_top_k", int(v)))
    params_form.addWidget(QtWidgets.QLabel("Top K"), 1, 2); params_form.addWidget(main.spin_top_k, 1, 3)

    btn_reset = QtWidgets.QPushButton("Reset to defaults")
    btn_reset.clicked.connect(lambda checked=False, m=main: _reset_defaults(m, checked))
    params_form.addWidget(btn_reset, 2, 0, 1, 4, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

    seg_lay.addLayout(row_seg_top)
    seg_lay.addLayout(row_seg_btns)
    seg_lay.addWidget(main.lab_model_seg)
    seg_lay.addLayout(params_form)
    seg_lay.addWidget(main.btn_run_seg)

    lay.addWidget(box_seg)
    lay.addStretch(1)
    return w

def _build_page_surv(main) -> QtWidgets.QWidget:
    v = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(v)
    lay.addLayout(_toolbar_with_back(main, "Survival analysis"))

    # ========== Survival analysis ==========
    box_surv = QtWidgets.QGroupBox("Survival analysis")
    surv_lay = QtWidgets.QVBoxLayout(box_surv)

    row_surv_top = QtWidgets.QHBoxLayout()
    row_surv_top.addWidget(QtWidgets.QLabel("Model:"))
    main.combo_surv = QtWidgets.QComboBox()
    main.combo_surv.addItems(list(main.models_surv.keys()))
    main.combo_surv.currentTextChanged.connect(lambda _: main._on_combo_changed("surv"))
    row_surv_top.addWidget(main.combo_surv, 1)

    row_surv_btns = QtWidgets.QHBoxLayout()
    btn_surv_add = QtWidgets.QPushButton("Add...")
    btn_surv_add.clicked.connect(lambda: main._add_model_from_file("surv"))
    btn_surv_del = QtWidgets.QPushButton("Remove")
    btn_surv_del.clicked.connect(lambda: main._remove_selected_model("surv"))
    btn_surv_set = QtWidgets.QPushButton("Set current")
    btn_surv_set.clicked.connect(lambda: main._set_current_model("surv"))
    row_surv_btns.addWidget(btn_surv_add)
    row_surv_btns.addWidget(btn_surv_del)
    row_surv_btns.addWidget(btn_surv_set)

    main.lab_model_surv = QtWidgets.QLabel("No survival model loaded")
    main.lab_model_surv.setWordWrap(True)
    main.lab_model_surv.setStyleSheet("color: grey;")

    main.btn_run_surv = QtWidgets.QPushButton("Run Survival")
    main.btn_run_surv.setEnabled(False)
    main.btn_run_surv.clicked.connect(main._run_surv)

    surv_lay.addLayout(row_surv_top)
    surv_lay.addLayout(row_surv_btns)
    surv_lay.addWidget(main.lab_model_surv)
    surv_lay.addWidget(main.btn_run_surv)

    lay.addWidget(box_surv)
    lay.addStretch(1)

    return v


def _build_page_edge(main) -> QtWidgets.QWidget:
    e = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(e)
    lay.addLayout(_toolbar_with_back(main, "Edge cluster"))

    # ========== Edge Clustering ==========
    box_edge = QtWidgets.QGroupBox("Edge Clustering")
    edge_lay = QtWidgets.QVBoxLayout(box_edge)

    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(8)

    # Canny low / high
    main.spin_ec_canny_low = QtWidgets.QSpinBox()
    main.spin_ec_canny_low.setRange(0, 255)
    main.spin_ec_canny_low.setValue(70)
    grid.addWidget(QtWidgets.QLabel("Canny low"), 1, 0)
    grid.addWidget(main.spin_ec_canny_low, 1, 1)

    main.spin_ec_canny_high = QtWidgets.QSpinBox()
    main.spin_ec_canny_high.setRange(0, 255)
    main.spin_ec_canny_high.setValue(110)
    grid.addWidget(QtWidgets.QLabel("Canny high"), 1, 2)
    grid.addWidget(main.spin_ec_canny_high, 1, 3)

    # DBSCAN eps / min_samples
    main.spin_ec_eps = QtWidgets.QDoubleSpinBox()
    main.spin_ec_eps.setRange(0.1, 50.0)
    main.spin_ec_eps.setSingleStep(0.1)
    main.spin_ec_eps.setDecimals(2)
    main.spin_ec_eps.setValue(3.0)
    grid.addWidget(QtWidgets.QLabel("DBSCAN eps"), 2, 0)
    grid.addWidget(main.spin_ec_eps, 2, 1)

    main.spin_ec_min_samples = QtWidgets.QSpinBox()
    main.spin_ec_min_samples.setRange(1, 1000)
    main.spin_ec_min_samples.setValue(5)
    grid.addWidget(QtWidgets.QLabel("min_samples"), 2, 2)
    grid.addWidget(main.spin_ec_min_samples, 2, 3)

    # min_cluster_size / morph_kernel
    main.spin_ec_min_cluster = QtWidgets.QSpinBox()
    main.spin_ec_min_cluster.setRange(1, 100000)
    main.spin_ec_min_cluster.setValue(10)
    grid.addWidget(QtWidgets.QLabel("min_cluster_size"), 3, 0)
    grid.addWidget(main.spin_ec_min_cluster, 3, 1)

    main.spin_ec_kernel = QtWidgets.QSpinBox()
    main.spin_ec_kernel.setRange(1, 25)
    main.spin_ec_kernel.setSingleStep(2)  # 奇数核更常用
    main.spin_ec_kernel.setValue(3)
    grid.addWidget(QtWidgets.QLabel("morph kernel"), 3, 2)
    grid.addWidget(main.spin_ec_kernel, 3, 3)

    edge_lay.addLayout(grid)

    # 行尾按钮：Run / Reset
    row_btns = QtWidgets.QHBoxLayout()
    row_btns.addStretch(1)
    btn_edge_reset = QtWidgets.QPushButton("Reset")
    btn_edge_reset.clicked.connect(lambda: _reset_edge_params(main))
    btn_edge_run = QtWidgets.QPushButton("Run Edge Clustering")
    btn_edge_run.clicked.connect(main._run_edge_cluster)
    row_btns.addWidget(btn_edge_reset)
    row_btns.addWidget(btn_edge_run)
    edge_lay.addLayout(row_btns)

    lay.addWidget(box_edge)

    lay.addStretch(1)

    return e




def _build_page_proj(main) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.addLayout(_toolbar_with_back(main, "Projection"))

    box = QtWidgets.QGroupBox("Projection")
    g = QtWidgets.QGridLayout(box)
    g.setHorizontalSpacing(10); g.setVerticalSpacing(8)

    btn_sobel = QtWidgets.QPushButton("Sobel projection")
    btn_sobel.clicked.connect(lambda: main._focus_stack_selected("sobel"))
    g.addWidget(btn_sobel, 0, 0)

    btn_max = QtWidgets.QPushButton("Max projection")
    btn_max.clicked.connect(lambda: main._focus_stack_selected("max"))
    g.addWidget(btn_max, 0, 1)

    btn_mean = QtWidgets.QPushButton("Mean projection")
    btn_mean.clicked.connect(lambda: main._focus_stack_selected("mean"))
    g.addWidget(btn_mean, 1, 0)

    lay.addWidget(box)
    lay.addStretch(1)
    return w


def _reset_defaults(main, checked=False):
    """恢复默认参数。"""

    main.spin_score.setValue(0.8)
    main.spin_nms.setValue(0.005)
    main.spin_min_area.setValue(0)
    main.spin_top_k.setValue(100)

    main.seg_min_side = 0

    if hasattr(main, "edit_outdir"):
        main.edit_outdir.clear()
        _on_param_changed(main, "seg_output_dir", None)

def _on_param_changed(main, key, value):
    # 回写到 Main 的运行态
    if   key == "seg_score_thr": main.seg_score_thr = value
    elif key == "seg_nms_iou":   main.seg_nms_iou   = value
    elif key == "seg_min_area":  main.seg_min_area  = value
    elif key == "seg_top_k":  main.seg_top_k  = value
    elif key == "seg_output_dir": main.seg_output_dir = value

    # 可选：立即刷新叠图
    if getattr(main, "last_img", None) is not None and getattr(main, "last_pred", None) is not None:
        # 只让透明度/阈值变化时重绘即可；这里简单调用你已有方法
        if hasattr(main, "_on_params_changed"):
            main._on_params_changed()

    # 持久化
    if hasattr(main, "settings"):
        main.settings.setValue(f"param/{key}", "" if value is None else value)


def _pick_output_dir(main):
    path = QtWidgets.QFileDialog.getExistingDirectory(main, "Select output directory", main.seg_output_dir or "")
    if path:
        if hasattr(main, "edit_outdir"):
            main.edit_outdir.setText(path)
        _on_param_changed(main, "seg_output_dir", path)


def _on_max_dets_changed(main,v):
    main.max_dets = int(v)
    if hasattr(main, "settings"):
        main.settings.setValue("param/max_dets", main.max_dets)
    # 已有图像+预测时，按新阈值/上限即时重绘
    if getattr(main, "last_img", None) is not None and getattr(main, "last_pred", None) is not None:
        if hasattr(main, "_on_params_changed"):
            main._on_params_changed()


def _reset_edge_params(main):
    # 还原到默认值
    main.spin_ec_canny_low.setValue(70)
    main.spin_ec_canny_high.setValue(110)
    main.spin_ec_eps.setValue(3.0)
    main.spin_ec_min_samples.setValue(5)
    main.spin_ec_min_cluster.setValue(10)
    main.spin_ec_kernel.setValue(3)



def _toolbar_with_back(main, title_text: str, back_to=PAGE_HOME):
    bar = QtWidgets.QHBoxLayout()

    btn_back = QtWidgets.QToolButton()
    btn_back.setText("← Back")
    btn_back.clicked.connect(lambda: _goto(main, back_to))

    lab = QtWidgets.QLabel(title_text)
    lab.setStyleSheet("font-weight:600;")

    # === 新增：Info 区域 + 独立窗口开关 ===
    # info（放在工具栏右侧）
    info_row = QtWidgets.QHBoxLayout()
    # 小的查看日志按钮
    btn_show_log = QtWidgets.QToolButton()
    btn_show_log.setText("View Log")
    btn_show_log.clicked.connect(lambda: main._toggle_info_window(True, force_show=True))

    info_row.addSpacing(8)
    info_row.addWidget(btn_show_log)

    # 组装
    bar.addWidget(btn_back)
    bar.addSpacing(6)
    bar.addWidget(lab)
    bar.addStretch(1)
    bar.addLayout(info_row)

    return bar


