# FocusStack/sobel_stack.py
import cv2
import numpy as np
from typing import List, Tuple

def focus_stack_sobel(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    改为：基于平均投影（Mean Projection）做 focus stacking。
    Args:
        image_paths: 多张同视野不同焦面的图片路径（尺寸必须一致）
    Returns:
        composite:  合成图 (H,W) 或 (H,W,3)，为逐像素平均结果
        idx:        每个像素取自哪一层的索引 (H,W)，此处恒为 0（无索引意义，仅兼容旧接口）
        sharp_stack:灰度强度堆栈 (N,H,W)，用于调试或可视化
    """
    if len(image_paths) < 2:
        raise ValueError("至少选择两张图片做融合")

    imgs = []
    grays = []
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

        # 灰度转换用于记录强度堆栈
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imgs.append(img.astype(np.float32))
        grays.append(gray.astype(np.float32))

    stack_img = np.stack(imgs, axis=0)    # (N,H,W) 或 (N,H,W,3)
    sharp_stack = np.stack(grays, axis=0) # (N,H,W)

    # === Mean Projection ===
    composite = np.mean(stack_img, axis=0)  # 平均融合
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    # 平均投影没有“选层”概念，idx 填 0 占位即可
    idx = np.zeros((H, W), dtype=np.int32)

    return composite, idx, sharp_stack
