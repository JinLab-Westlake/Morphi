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
    # 若是 object（罕见情况），尽量 stack
    if a.dtype == object:
        try:
            a = np.stack(list(a))
        except Exception:
            # 尝试转 float 列表
            try:
                a = np.array([float(x) for x in a], dtype=np.float32)
            except Exception:
                pass
    return a



# squeeze 最后一维为1的情况（只对 scores/classes）
def _squeeze_last1(a):
    if a is None: return None
    while a.ndim >= 1 and a.shape[-1] == 1:
        a = np.squeeze(a, axis=-1)
    return a

# 对齐 N
def _len(a):
    return a.shape[0] if (a is not None and hasattr(a, "shape") and a.ndim >= 1) else None




def _normalize_det_tensors(boxes, scores, classes, masks):
    """
    统一 boxes/scores/classes/masks 的形状：
      - scores 最终为 (N,)
      - 若 scores 为 (N,C)，使用 classes 取对应列，否则取 argmax，并同步 classes
      - squeeze 掉末尾的 size=1 维度
      - classes -> int64, shape (N,)
    返回：boxes, scores, classes, masks
    """

    boxes  = _to_np(boxes)
    scores = _to_np(scores)
    classes= _to_np(classes)
    masks  = _to_np(masks)

    scores  = _squeeze_last1(scores)
    classes = _squeeze_last1(classes)

    # 处理 (N,C) 的 scores
    if scores is not None and scores.ndim == 2:
        N, C = scores.shape
        if classes is not None and classes.ndim == 1 and classes.shape[0] == N:
            idx = np.arange(N)
            scores = scores[idx, classes.astype(int)]
        else:
            classes = np.argmax(scores, axis=1).astype(np.int64)
            scores  = np.max(scores, axis=1)

    # 若 scores 依然有多维（例如 (N,H,W) 这类不合理输出），退化为每实例最大值
    if scores is not None and scores.ndim > 1:
        N = scores.shape[0]
        scores = scores.reshape(N, -1).max(axis=1)

    # classes 类型/形状
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