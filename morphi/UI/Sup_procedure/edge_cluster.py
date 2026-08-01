# Fluo_proccess/Canny/edge_cluster_worker.py
from PyQt6 import QtCore
from Fluo_proccess.Canny.canny_edge_detec import EdgeClusterParams, process_folder, process_one_image
import os

class EdgeClusterWorker(QtCore.QObject):
    progressed = QtCore.pyqtSignal(int, int, str, dict)  # i, n, path, stats
    finished   = QtCore.pyqtSignal()
    failed     = QtCore.pyqtSignal(str)

    def __init__(self, folder: str, out_dir: str, lab_dir: str, params: EdgeClusterParams):
        super().__init__()
        self.folder = folder
        self.out_dir = out_dir
        self.lab_dir = lab_dir
        self.params = params
        self._cancel = False

    @QtCore.pyqtSlot()
    def run(self):
        try:
            def cb(i, n, p, s): self.progressed.emit(i, n, p, s)
            def should_cancel(): return self._cancel

            # 如果传入的是“文件路径”，则只处理这一张
            if os.path.isfile(self.folder):
                process_one_image(self.folder, self.out_dir, self.lab_dir,
                                  self.params, progress_cb=cb)
            else:
                # 否则按文件夹批处理
                process_folder(self.folder, self.out_dir, self.lab_dir,
                               self.params, progress_cb=cb, should_cancel=should_cancel)

            self.finished.emit()
        except Exception as e:
            self.failed.emit(str(e))

    @QtCore.pyqtSlot()
    def cancel(self):
        self._cancel = True
