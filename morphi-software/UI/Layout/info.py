from PyQt6 import QtCore, QtGui, QtWidgets


# def _build_right_panel(main) -> QtWidgets.QWidget:
#     right = QtWidgets.QWidget()
#     r_lay = QtWidgets.QVBoxLayout(right)
#
#
#
#     return right








# def _add_images_to_list(self):
#     self._ensure_image_paths_list()
#     files, _ = QtWidgets.QFileDialog.getOpenFileNames(
#         self, "Add images", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
#     )
#     for f in files:
#         if f not in self.image_paths:
#             self.image_paths.append(f)
#             self.list_images.addItem(f)
#
# def _remove_selected_images(self):
#     self._ensure_image_paths_list()
#     for it in self.list_images.selectedItems():
#         path = it.text()
#         if path in self.image_paths:
#             self.image_paths.remove(path)
#         self.list_images.takeItem(self.list_images.row(it))
#
# def _clear_images_list(self):
#     self._ensure_image_paths_list()
#     self.image_paths.clear()
#     self.list_images.clear()
#
# def _preview_selected_image(self):
#     items = self.list_images.selectedItems()
#     if not items:
#         return
#     path = items[0].text()
#     import cv2
#     img = cv2.imread(path)
#     if img is None:
#         return
#     self.last_img = img
#     self.show_image(img)  # 这里用你已有的 show_image
#
#
#
#
#
