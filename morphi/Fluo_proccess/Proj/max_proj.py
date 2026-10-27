# FocusStack/sobel_stack.py
import cv2
import numpy as np
from typing import List, Tuple

def focus_stack_sobel(image_paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    focus stacking is performed based on Max Intensity Projection.
    Args:
        image_paths: Paths for multiple images with different focal planes in the same field of view (must be of the same size)
    Returns:
        composite: Composite graph (H,W) or (H,W,3)
        idx: the index of the layer from which each pixel is taken (H,W)
        sharp_stack: Intensity stack (N,H,W) - here is grayscale intensity for easy debugging/visualization
    """
    if len(image_paths) < 2:
        raise ValueError("Select at least two images to fuse")

    imgs = []
    intensities = []
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
            # cv2.cvtColor for 8/16 bit
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imgs.append(img)
        intensities.append(gray.astype(np.float32))

    stack_img = np.stack(imgs, axis=0)           # (N,H,W) or (N,H,W,3)
    sharp_stack = np.stack(intensities, axis=0)

    # Select the layer with the largest gray level for each pixel
    idx = np.argmax(sharp_stack, axis=0)         # (H,W)

    # Advanced index picks pixels (rgb/gray)
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    composite = stack_img[idx, rows, cols]       # (H,W) or (H,W,3)

    return composite, idx, sharp_stack
