from PyQt6 import QtWidgets,QtGui
import os


def _build_menubar(self):
    menubar = self.menuBar()
    # Use the native menu bar on macOS
    try:
        menubar.setNativeMenuBar(True)
    except Exception:
        pass

    m_file = menubar.addMenu("&File")

    # --- Open image ---
    act_open_img = QtGui.QAction("Open Image…", self)
    act_open_img.setShortcut(QtGui.QKeySequence.StandardKey.Open)
    act_open_img.triggered.connect(self.choose_image)
    m_file.addAction(act_open_img)

    # --- Open model (Submenu: Seg / Surv) ---
    m_open_model = m_file.addMenu("Open Model")
    act_open_seg = QtGui.QAction("Segmentation Model…", self)
    act_open_seg.triggered.connect(lambda: self._add_model_from_file("seg"))
    m_open_model.addAction(act_open_seg)

    act_open_surv = QtGui.QAction("Survival Model…", self)
    act_open_surv.triggered.connect(lambda: self._add_model_from_file("surv"))
    m_open_model.addAction(act_open_surv)

    # --- Set output directory ---
    act_set_outdir = QtGui.QAction("Set Output Directory…", self)
    act_set_outdir.triggered.connect(lambda: _pick_output_dir(self))
    m_file.addAction(act_set_outdir)

    m_file.addSeparator()

    # --- Save current result ---
    act_save = QtGui.QAction("Save Current Result", self)
    act_save.setShortcut(QtGui.QKeySequence.StandardKey.Save)
    act_save.triggered.connect(self._save_overlay)
    m_file.addAction(act_save)

    m_file.addSeparator()

    # --- Recent files---
    self.m_recent = m_file.addMenu("Open Recent")
    _rebuild_recent_menu(self)

    m_file.addSeparator()

    # --- Preferences ---
    act_prefs = QtGui.QAction("Preferences…", self)
    # act_prefs.setShortcut(QtGui.QKeySequence("Ctrl+,"))
    act_prefs.triggered.connect(self._open_preferences_dialog if hasattr(self, "_open_preferences_dialog") else lambda: None)
    m_file.addAction(act_prefs)

    m_file.addSeparator()

    # --- Exit / Quit ---
    act_quit = QtGui.QAction("Exit", self)
    act_quit.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
    act_quit.triggered.connect(self.close)
    m_file.addAction(act_quit)


def _pick_output_dir(main):
    d = QtWidgets.QFileDialog.getExistingDirectory(main, "Select output directory", main.seg_output_dir or "")
    if d:
        main.seg_output_dir = d
        if hasattr(main, "settings"):
            main.settings.setValue("param/output_dir", d)
        # If there is an edit box on the left: Display synchronously
        if hasattr(main, "edit_outdir") and main.edit_outdir is not None:
            main.edit_outdir.setText(d)



def _add_recent_image(self, path: str, key="recent/images", max_items=10):
    if not hasattr(self, "settings"):
        return
    items = self.settings.value(key, [], type=list) or []
    # Remove duplicates and put the latest ones first
    items = [p for p in items if p != path]
    items.insert(0, path)
    items = items[:max_items]
    self.settings.setValue(key, items)
    self._rebuild_recent_menu()

def _rebuild_recent_menu(self, key="recent/images"):
    if not hasattr(self, "settings") or not hasattr(self, "m_recent"):
        return
    self.m_recent.clear()
    items = self.settings.value(key, [], type=list) or []
    if not items:
        a = self.m_recent.addAction("(No recent files)")
        a.setEnabled(False)
        return

    for p in items:
        act = self.m_recent.addAction(p)
        act.triggered.connect(lambda _, path=p: self._open_image_from_path(path))

    self.m_recent.addSeparator()
    act_clear = self.m_recent.addAction("Clear Menu")
    act_clear.triggered.connect(lambda: (self.settings.remove(key), self._rebuild_recent_menu()))

def _open_image_from_path(self, path: str):
    if not os.path.exists(path):
        QtWidgets.QMessageBox.warning(self, "Open Recent", f"File not found:\n{path}")
        return
    self._load_image(path) if hasattr(self, "_load_image") else self._load_image_fallback(path)

def _load_image_fallback(self, path: str):

    self.path_label.setText(path)








