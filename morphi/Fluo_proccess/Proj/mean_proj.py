# FocusStack/sobel_stack.py
import cv2
import numpy as np
from typing import List, Tuple

def focus_stack_sobel(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    focus stacking is based on the Mean Projection.
    Args:
        image_paths: Paths for multiple images with different focal planes in the same field of view (must be of the same size)
    Returns:
        composite: Composite plot (H,W) or (H,W,3), which is the per-pixel average result
        idx: the index (H,W) of the layer from which each pixel is taken, which is always 0 (no index meaning, only compatible with the old interface)
        sharp_stack: Grayscale intensity stack (N,H,W) for debugging or visualization
    """
    if len(image_paths) < 2:
        raise ValueError("Select at least two images to fuse")

    imgs = []
    grays = []
    H = W = None

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Fail to load: {p}")

        # Check the size is consistent
        h, w = img.shape[:2]
        if H is None:
            H, W = h, w
        elif (H, W) != (h, w):
            raise ValueError(f"Inconsistent size：{p} is {h}x{w}, and the previous one is {H}x{W}")

        # Grayscale (for projection metric)
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imgs.append(img.astype(np.float32))
        grays.append(gray.astype(np.float32))

    stack_img = np.stack(imgs, axis=0)    # (N,H,W) or (N,H,W,3)
    sharp_stack = np.stack(grays, axis=0) # (N,H,W)

    # === Mean Projection ===
    composite = np.mean(stack_img, axis=0)  # Average fusion
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    # There is no "layer selection" concept for average projection, idx fills in 0 placeholders
    idx = np.zeros((H, W), dtype=np.int32)

    return composite, idx, sharp_stack
