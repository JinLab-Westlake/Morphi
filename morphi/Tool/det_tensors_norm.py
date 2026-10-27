import numpy as np


# to numpy
def _to_np(a):
    if a is None:
        return None
    if isinstance(a, (list, tuple)):
        try:
            a = np.array(a)
        except Exception:
            a = np.asarray(a, dtype=object)
    else:
        a = np.asarray(a)
    # If it's an object, stack as much as possible
    if a.dtype == object:
        try:
            a = np.stack(list(a))
        except Exception:
            # Try flipping float lists
            try:
                a = np.array([float(x) for x in a], dtype=np.float32)
            except Exception:
                pass
    return a



# squeeze when the last dimension is 1 (for scores/classes only)
def _squeeze_last1(a):
    if a is None: return None
    while a.ndim >= 1 and a.shape[-1] == 1:
        a = np.squeeze(a, axis=-1)
    return a

# Align N
def _len(a):
    return a.shape[0] if (a is not None and hasattr(a, "shape") and a.ndim >= 1) else None




def _normalize_det_tensors(boxes, scores, classes, masks):
    """
    The shape of the unified boxes/scores/classes/masks:
        - scores result in (N,).
        - If scores are (N,C), use classes to take the corresponding column, otherwise take argmax, and synchronize classes
        - squeeze out the size=1 dimension at the end
        - classes -> int64, shape (N,)
    Returns: boxes, scores, classes, masks
    """

    boxes  = _to_np(boxes)
    scores = _to_np(scores)
    classes= _to_np(classes)
    masks  = _to_np(masks)

    scores  = _squeeze_last1(scores)
    classes = _squeeze_last1(classes)

    # Process the scores of (N,C)
    if scores is not None and scores.ndim == 2:
        N, C = scores.shape
        if classes is not None and classes.ndim == 1 and classes.shape[0] == N:
            idx = np.arange(N)
            scores = scores[idx, classes.astype(int)]
        else:
            classes = np.argmax(scores, axis=1).astype(np.int64)
            scores  = np.max(scores, axis=1)

    # If scores still have multiple dimensions, they degrade to the maximum value per instance
    if scores is not None and scores.ndim > 1:
        N = scores.shape[0]
        scores = scores.reshape(N, -1).max(axis=1)

    # classes type/shape
    if classes is not None:
        classes = classes.astype(np.int64).reshape(-1)

    lens = [x for x in (_len(boxes), _len(scores), _len(classes), _len(masks)) if x is not None]
    if lens:
        N = min(lens)
        if boxes  is not None:  boxes  = boxes[:N]
        if scores is not None:  scores = scores[:N]
        if classes is not None: classes= classes[:N]
        if masks  is not None:  masks  = masks[:N]

    return boxes, scores, classes, masks