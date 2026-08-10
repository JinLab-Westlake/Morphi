# Processing/edge_cluster.py
from dataclasses import dataclass
import os, glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from skimage import measure
from typing import Callable, Optional

@dataclass
class EdgeClusterParams:
    # alpha: float = 2.0          # 对比度
    # beta: float = 1.0           # 亮度
    canny_low: int = 70
    canny_high: int = 110
    db_eps: float = 3.0
    db_min_samples: int = 5
    db_min_cluster_size: int = 10
    morph_kernel: int = 3       # 膨胀核大小
    image_glob: str = None  # 允许改成 *.tif

    def __post_init__(self):
        if self.image_glob is None:
            self.image_globs = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp"]

def _cluster_edges_with_dbscan(edges: np.ndarray,
                               eps=3.0,
                               min_samples=5,
                               min_cluster_size=10):
    y_coords, x_coords = np.where(edges > 0)
    edge_points = np.column_stack((x_coords, y_coords))
    if edge_points.shape[0] == 0:
        return None, 0

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(edge_points)

    clustered_img = np.zeros((*edges.shape, 3), dtype=np.uint8)
    unique_labels = set(labels)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    count = 0
    for label in unique_labels:
        if label == -1:
            continue
        points_in_cluster = np.sum(labels == label)
        if points_in_cluster < min_cluster_size:
            continue
        color = (np.array(colors(label)[:3]) * 255).astype(np.uint8)
        pts = edge_points[labels == label]
        clustered_img[pts[:, 1], pts[:, 0]] = color
        count += 1
    return clustered_img, count


def process_one_image(image_path: str,
                      output_dir: str,
                      label_dir: str,
                      params: EdgeClusterParams,
                      progress_cb=None):
    """
    只处理单张图片，保存可视化结果和标签图。
    - progress_cb(1, 1, path, stats)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    img = cv2.imread(image_path)
    vis, lab, stats = process_image(img, params)

    # 可能无有效边缘
    if lab is not None:
        cv2.imwrite(os.path.join(output_dir, f"clustered_{os.path.basename(image_path)}"), vis)
        cv2.imwrite(os.path.join(label_dir,   os.path.basename(image_path)), lab * 255)

    if progress_cb:
        progress_cb(1, 1, image_path, stats or {"clusters": 0})

    return 1


def process_image(image_bgr: np.ndarray, params: EdgeClusterParams):
    """对单张图像执行：增强→Canny→DBSCAN→膨胀→连通域→轮廓 & 标签图"""
    if image_bgr is None or image_bgr.size == 0:
        return None, None, {"clusters": 0}

    # 边缘
    edges = cv2.Canny(image_bgr, params.canny_low, params.canny_high)

    # DBSCAN 聚类
    clustered_edges, n_clusters = _cluster_edges_with_dbscan(
        edges,
        eps=params.db_eps,
        min_samples=params.db_min_samples,
        min_cluster_size=params.db_min_cluster_size
    )
    if clustered_edges is None:
        return image_bgr, np.zeros(image_bgr.shape[:2], np.uint8), {"clusters": 0}

    # 形态学（膨胀）
    kernel = np.ones((params.morph_kernel, params.morph_kernel), np.uint8)
    clustered_edges = cv2.dilate(clustered_edges, kernel, iterations=1)

    # 连通域（用 R 通道就够）
    labels = measure.label(clustered_edges[..., 0], connectivity=2)
    regions = measure.regionprops(labels, intensity_image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))

    h, w = image_bgr.shape[:2]
    label_image = np.zeros((h, w), dtype=np.uint8)
    out_vis = image_bgr.copy()

    for region in regions:
        if len(region.bbox) != 4:
            continue
        minr, minc, maxr, maxc = region.bbox
        region_mask = np.zeros((h, w), dtype=np.uint8)
        region_mask[minr:maxr, minc:maxc] = region.filled_image.astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_vis, contours, -1, (0, 0, 255), 2)
        for contour in contours:
            cv2.drawContours(label_image, [contour], -1, 2, thickness=cv2.FILLED)

    stats = {"clusters": n_clusters}
    return out_vis, label_image, stats

def process_folder(image_folder: str,
                   output_dir: str,
                   label_dir: str,
                   params: EdgeClusterParams,
                   progress_cb=None,
                   should_cancel: Optional[Callable[[], bool]] = None):
    """
    批处理整个文件夹。
    - progress_cb(i, n, path, stats) 用于 UI 更新进度
    - should_cancel() 返回 True 时提前终止
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(image_folder, params.image_glob)))
    n = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        if should_cancel and should_cancel():
            break
        img = cv2.imread(path)
        vis, lab, stats = process_image(img, params)
        if lab is None:  # 无有效边缘
            if progress_cb: progress_cb(i, n, path, stats)
            continue

        # 保存
        cv2.imwrite(os.path.join(output_dir, f"clustered_{os.path.basename(path)}"), vis)
        cv2.imwrite(os.path.join(label_dir,   os.path.basename(path)), lab * 255)

        if progress_cb:
            progress_cb(i, n, path, stats)

    return n


