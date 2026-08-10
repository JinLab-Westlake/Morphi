# FocusStack/sobel_stack.py
import cv2
import numpy as np
from typing import List, Tuple

def _sobel_energy(gray: np.ndarray) -> np.ndarray:
    """Sobel 清晰度能量：Gx^2 + Gy^2（float32）"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy

def focus_stack_sobel(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    基于 Sobel 清晰度做 focus stacking。
    Args:
        image_paths: 多张同视野不同焦面的图片路径（尺寸必须一致）
    Returns:
        composite:  合成图 (H,W) 或 (H,W,3)
        idx:        每个像素取自哪一层的索引 (H,W)
        sharp_stack:清晰度堆栈 (N,H,W) 方便调试/可视化
    """
    if len(image_paths) < 2:
        raise ValueError("至少选择两张图片做融合")

    imgs = []
    sharps = []
    H = W = None

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"读取失败: {p}")

        # 记录尺寸并校验一致
        h, w = img.shape[:2]
        if H is None:
            H, W = h, w
        elif (H, W) != (h, w):
            raise ValueError(f"尺寸不一致：{p} 是 {h}x{w}，而前一张是 {H}x{W}")

        # 灰度 & float32
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        imgs.append(img)
        sharps.append(_sobel_energy(gray))

    stack_img = np.stack(imgs, axis=0)        # (N,H,W) 或 (N,H,W,3)
    sharp_stack = np.stack(sharps, axis=0)    # (N,H,W)

    # 每个像素上，选清晰度最大的那一层
    idx = np.argmax(sharp_stack, axis=0)      # (H,W)

    # 利用高级索引一次性取像素（彩色/灰度都支持）
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    composite = stack_img[idx, rows, cols]    # (H,W) 或 (H,W,3)

    return composite, idx, sharp_stack
