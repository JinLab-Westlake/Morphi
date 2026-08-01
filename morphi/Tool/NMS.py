# iou;nms;minimum thresh

import numpy as np
def _iou_xyxy(a, b):
    x1, y1, x2, y2 = [float(v) for v in a[:4]]
    x3, y3, x4, y4 = [float(v) for v in b[:4]]
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_b = max(0.0, x4 - x3) * max(0.0, y4 - y3)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0

def _nms_keep_indices(boxes, scores, iou_thr=0.5, classes=None, classwise=False):
    """
    返回保留的索引列表。
    - classwise=False：全局 NMS（常见做法）
    - classwise=True ：按类别分组做 NMS
    """
    if boxes is None or scores is None or len(boxes) == 0:
        return np.array([], dtype=np.int64)

    if not classwise or classes is None:
        order = np.argsort(-scores)  # 高分优先
        keep = []
        while len(order) > 0:
            i = int(order[0]); keep.append(i)
            if len(order) == 1: break
            rest = order[1:]
            ious = np.array([_iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
            order = rest[ious <= iou_thr]
        return np.array(keep, dtype=np.int64)

    # 按类别分组
    keep_all = []
    classes = classes.astype(np.int64).reshape(-1)
    for c in np.unique(classes):
        idx_c = np.where(classes == c)[0]
        if len(idx_c) == 0:
            continue
        order = idx_c[np.argsort(-scores[idx_c])]
        keep_c = []
        while len(order) > 0:
            i = int(order[0]); keep_c.append(i)
            if len(order) == 1: break
            rest = order[1:]
            ious = np.array([_iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
            order = rest[ious <= iou_thr]
        keep_all.extend(keep_c)
    return np.array(keep_all, dtype=np.int64)

def _filter_by_min(boxes, min_area=0, min_side=0):
    if boxes is None or len(boxes) == 0:
        return np.array([], dtype=bool)
    boxes = np.asarray(boxes)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    m_area = (w * h) >= float(min_area) if min_area > 0 else np.ones(len(boxes), dtype=bool)
    m_side = ((w >= float(min_side)) & (h >= float(min_side))) if min_side > 0 else np.ones(len(boxes), dtype=bool)
    return m_area & m_side
