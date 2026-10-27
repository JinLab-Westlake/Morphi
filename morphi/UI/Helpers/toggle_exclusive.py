# UI/Helpers/toggle_exclusive.py
from PyQt6 import QtWidgets

class TwoCheckExclusive:
    """
    Two QCheckboxes are mutually exclusive: checking one automatically cancels the other. Do not disable/put ash.
    The optional callback on_edit_toggled/on_brush_toggled handles the business.
    """
    def __init__(self,
                 edit_cb: QtWidgets.QCheckBox,
                 brush_cb: QtWidgets.QCheckBox,
                 *,
                 on_edit_toggled=None,
                 on_brush_toggled=None):
        self.edit_cb = edit_cb
        self.brush_cb = brush_cb
        self._on_edit  = on_edit_toggled
        self._on_brush = on_brush_toggled
        self._in_change = False

        self.edit_cb.toggled.connect(self._edit_changed)
        self.brush_cb.toggled.connect(self._brush_changed)

    def _edit_changed(self, on: bool):
        if self._in_change:
            return
        self._in_change = True
        try:
            if on:
                self._set_checked(self.brush_cb, False)
            if callable(self._on_edit):
                self._on_edit(bool(on))
        finally:
            self._in_change = False

    def _brush_changed(self, on: bool):
        if self._in_change:
            return
        self._in_change = True
        try:
            if on:
                self._set_checked(self.edit_cb, False)
            if callable(self._on_brush):
                self._on_brush(bool(on))
        finally:
            self._in_change = False

    @staticmethod
    def _set_checked(cb: QtWidgets.QCheckBox, checked: bool):
        cb.blockSignals(True)
        try:
            cb.setChecked(checked)
        finally:
            cb.blockSignals(False)

    def init_state(self):
        """
        Initialize once: If both are checked, leave edit, unbrush;
        If not, do not do processing.
        """
        if self.edit_cb.isChecked() and self.brush_cb.isChecked():
            self._set_checked(self.brush_cb, False)
