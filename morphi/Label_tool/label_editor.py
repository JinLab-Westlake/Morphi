from PyQt6 import QtWidgets, QtGui, QtCore

class LabelEditorDialog(QtWidgets.QDialog):
    """
    Simple table: id/score/label_id(/name)
    - label column editable (QLineEdit); If you have a fixed list of class names, you can also change it to QComboBox.
    """
    def __init__(self, parent, boxes, scores, classes, id2name=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Labels")
        self.resize(520, 420)
        self._id2name = id2name

        table = QtWidgets.QTableWidget(self)
        table.setColumnCount(4 if id2name else 3)
        headers = ["id", "score", "label_id"]
        if id2name:
            headers.append("label_name")
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked |
                              QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked)

        n = 0 if boxes is None else int(boxes.shape[0])
        table.setRowCount(n)
        for i in range(n):
            it_id = QtWidgets.QTableWidgetItem(str(i))
            it_id.setFlags(it_id.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(i, 0, it_id)

            sc = float(scores[i]) if scores is not None else None
            it_sc = QtWidgets.QTableWidgetItem(f"{sc:.4f}" if sc is not None else "-")
            it_sc.setFlags(it_sc.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(i, 1, it_sc)

            lb = int(classes[i]) if classes is not None else -1
            it_lb = QtWidgets.QTableWidgetItem(str(lb))
            table.setItem(i, 2, it_lb)

            if id2name:
                nm = id2name.get(lb, "")
                it_nm = QtWidgets.QTableWidgetItem(str(nm))
                # The name column is also editable
                table.setItem(i, 3, it_nm)

        btn_ok = QtWidgets.QPushButton("OK")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        lay_btn = QtWidgets.QHBoxLayout()
        lay_btn.addStretch(1)
        lay_btn.addWidget(btn_ok)
        lay_btn.addWidget(btn_cancel)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(table)
        lay.addLayout(lay_btn)

        self._table = table

    def get_labels(self):
        """
        Return: new classes column, and optional name mapping
        """
        n = self._table.rowCount()
        new_ids = []
        new_names = {}
        has_name_col = (self._table.columnCount() >= 4)
        for i in range(n):
            # label_id
            it_lb = self._table.item(i, 2)
            try:
                v = int(it_lb.text())
            except Exception:
                v = -1
            new_ids.append(v)
            if has_name_col:
                it_nm = self._table.item(i, 3)
                new_names[i] = it_nm.text().strip() if it_nm else ""
        return new_ids, new_names
