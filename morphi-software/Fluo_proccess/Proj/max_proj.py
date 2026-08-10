# FocusStack/sobel_stack.py
import cv2
import numpy as np
from typing import List, Tuple

def focus_stack_sobel(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    改为：基于最大强度投影（Max Intensity Projection）做 focus stacking。
    Args:
        image_paths: 多张同视野不同焦面的图片路径（尺寸必须一致）
    Returns:
        composite:  合成图 (H,W) 或 (H,W,3)
        idx:        每个像素取自哪一层的索引 (H,W)
        sharp_stack:强度堆栈 (N,H,W) —— 此处为灰度强度，便于调试/可视化
    """
    if len(image_paths) < 2:
        raise ValueError("至少选择两张图片做融合")

    imgs = []
    intensities = []
    H = W = None

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"读取失败: {p}")

        # 校验尺寸一致
        h, w = img.shape[:2]
        if H is None:
            H, W = h, w
        elif (H, W) != (h, w):
            raise ValueError(f"尺寸不一致：{p} 是 {h}x{w}，而前一张是 {H}x{W}")

        # 灰度（用于投影度量）；保持原图 dtype 供最终拣选像素
        if img.ndim == 2:
            gray = img
        else:
            # cv2.cvtColor 支持 8/16 位；统一转 float32 便于后续运算
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imgs.append(img)
        intensities.append(gray.astype(np.float32))

    stack_img = np.stack(imgs, axis=0)           # (N,H,W) 或 (N,H,W,3)
    sharp_stack = np.stack(intensities, axis=0)  # (N,H,W) —— 此处为“强度堆栈”

    # 每个像素选灰度最大的那一层
    idx = np.argmax(sharp_stack, axis=0)         # (H,W)

    # 高级索引拣选像素（彩色/灰度均可）
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    composite = stack_img[idx, rows, cols]       # (H,W) 或 (H,W,3)

    return composite, idx, sharp_stack
