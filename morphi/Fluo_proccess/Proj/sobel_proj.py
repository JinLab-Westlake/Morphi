# FocusStack/sobel_stack.py
import cv2
import numpy as np
from typing import List, Tuple

def _sobel_energy(gray: np.ndarray) -> np.ndarray:
    """Sobel clarity energy：Gx^2 + Gy^2（float32）"""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy

def focus_stack_sobel(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Focus stacking based on Sobel sharpness.
    Args:
        image_paths: Paths for multiple images with different focal planes in the same field of view (must be of the same size)
    Returns:
        composite: Composite graph (H,W) or (H,W,3)
        idx: the index of the layer from which each pixel is taken (H,W)
        sharp_stack: Sharpness stack (N,H,W) for debugging/visualization
    """
    if len(image_paths) < 2:
        raise ValueError("Select at least two images to fuse")

    imgs = []
    sharps = []
    H = W = None

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Fail to load: {p}")

        # Record dimensions and check for consistency
        h, w = img.shape[:2]
        if H is None:
            H, W = h, w
        elif (H, W) != (h, w):
            raise ValueError(f"Inconsistent size：{p} is {h}x{w}, and the previous one is {H}x{W}")

        # gray & float32
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)

        imgs.append(img)
        sharps.append(_sobel_energy(gray))

    stack_img = np.stack(imgs, axis=0)        # (N,H,W) or (N,H,W,3)
    sharp_stack = np.stack(sharps, axis=0)    # (N,H,W)

    # For each pixel, choose the layer with the highest definition
    idx = np.argmax(sharp_stack, axis=0)      # (H,W)

    # Fetch pixels at once using advanced indexing (rgb/gray)
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    composite = stack_img[idx, rows, cols]    # (H,W) or (H,W,3)

    return composite, idx, sharp_stack
