# seg_infer.py
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.cm as cm


def resize_shortest_edge(img_bgr, min_size=800, max_size=1333):
    h, w = img_bgr.shape[:2]
    scale = min_size / min(h, w)
    if round(scale * max(h, w)) > max_size:
        scale = max_size / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return img_bgr, 1.0
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale

class Segmentor:
    """
    Supports runtime parameter feeding to adjustable points in the graph:
      - max_output_boxes_per_class (int64 [1])
      - iou_threshold               (float32 [1])
      - score_threshold             (float32 [1])
      - TopK_k                      (int64 [1])
    """
    def __init__(self, onnx_path, use_short_edge_resize=True, min_size=800, max_size=1333):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inputs  = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        self.output_names = [o.name for o in self.outputs]

        # Keep track of all the input names
        self.input_names  = [i.name for i in self.inputs]

        # Select image input name
        self.input_name, self.input_shape, self.input_rank = None, None, None
        for i in self.inputs:
            shp = i.shape
            if isinstance(shp, (list, tuple)) and len(shp) in (3, 4):
                self.input_name  = i.name
                self.input_shape = shp
                self.input_rank  = len(shp)
                break
        if self.input_name is None:
            # The bottom of the pocket uses the first input
            self.input_name  = self.inputs[0].name
            self.input_shape = self.inputs[0].shape
            self.input_rank  = len(self.input_shape)

        # Fixed size and scaling strategy
        if self.input_rank == 3:
            _, H, W = self.input_shape
        elif self.input_rank == 4:
            _, _, H, W = self.input_shape
        else:
            raise RuntimeError(f"Unsupported input rank: {self.input_rank}")
        self.fixed_hw = (int(H), int(W)) if isinstance(H, int) and isinstance(W, int) else None
        self.use_short_edge_resize = use_short_edge_resize
        self.min_size, self.max_size = min_size, max_size

        # —— Find the adjustable point input name ——
        def _find_by_suffix(*suffixes):
            for name in self.input_names:
                for suf in suffixes:
                    if name.endswith(suf) or name == suf:
                        return name
            return None

        self.nms_max_name   = _find_by_suffix("max_output_boxes_per_class", "nms_max_output_boxes_per_class")
        self.nms_iou_name   = _find_by_suffix("iou_threshold", "nms_iou_threshold")
        self.nms_score_name = _find_by_suffix("score_threshold", "nms_score_threshold")
        self.topk_k_name    = _find_by_suffix("TopK_k", "topk_k", "k")

        # Runtime parameters
        self.runtime_params = {
            "max_output_boxes_per_class": None,  # int
            "iou_threshold": None,               # float
            "score_threshold": None,             # float
            "TopK_k": None,                      # int
        }

    # External interface: Set from UI/ business layer
    def set_runtime_params(self, *, max_dets=None, iou_thr=None, score_thr=None, topk_k=None):
        if max_dets is not None:
            self.runtime_params["max_output_boxes_per_class"] = int(max_dets)
        if iou_thr is not None:
            self.runtime_params["iou_threshold"] = float(iou_thr)
        if score_thr is not None:
            self.runtime_params["score_threshold"] = float(score_thr)
        if topk_k is not None:
            self.runtime_params["TopK_k"] = int(topk_k)

    # Debugging: To see what inputs are in the model
    def list_input_names(self):
        return self.input_names

    def _post_name_map(self, tensors):
        outs = dict(zip(self.output_names, tensors))
        boxes = scores = classes = masks = None
        for k, v in outs.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == 'f' and v.ndim >= 2 and v.shape[-1] == 4:
                boxes = v; break
        for k, v in outs.items():
            if not isinstance(v, np.ndarray): continue
            if v.ndim >= 1 and v.shape[-1] != 4:
                if v.dtype.kind == 'f' and scores is None:  scores = v
                if v.dtype.kind in ('i','u') and classes is None: classes = v
        for k, v in outs.items():
            if not isinstance(v, np.ndarray): continue
            if v.ndim >= 3 and v.shape[-1] not in (1,4):
                masks = v; break

        def squeeze_last1(a):
            if a is None: return None
            if a.ndim >= 2 and a.shape[-1] == 1: a = a[..., 0]
            return a

        scores  = squeeze_last1(scores)
        classes = squeeze_last1(classes)
        if scores is not None and scores.ndim == 2:
            N = scores.shape[0]
            if classes is not None and classes.ndim == 1 and classes.shape[0] == N:
                idx = np.arange(N)
                scores = scores[idx, classes.astype(int)]
            else:
                classes = np.argmax(scores, axis=1).astype(np.int64)
                scores  = np.max(scores, axis=1)

        return {"boxes": boxes, "scores": scores, "classes": classes, "masks": masks}

    def predict(self, img_bgr):
        if img_bgr is None:
            raise ValueError("img_bgr is None")
        img_in = img_bgr
        if self.use_short_edge_resize:
            img_in, _ = resize_shortest_edge(img_bgr, self.min_size, self.max_size)
        if self.fixed_hw is not None and img_in.shape[:2] != self.fixed_hw:
            H, W = self.fixed_hw
            img_in = cv2.resize(img_in, (W, H), interpolation=cv2.INTER_LINEAR)

        x = img_in.astype(np.float32)
        x = np.transpose(x, (2, 0, 1))  # HWC->CHW
        x = np.ascontiguousarray(x)
        if self.input_rank == 4:
            x = x[None, ...]

        # —— Constructing feeds ——
        feeds = { self.input_name: x }
        rp = self.runtime_params

        if self.nms_max_name and rp["max_output_boxes_per_class"] is not None:
            feeds[self.nms_max_name] = np.array([rp["max_output_boxes_per_class"]], dtype=np.int64)
        if self.nms_iou_name and rp["iou_threshold"] is not None:
            feeds[self.nms_iou_name] = np.array([rp["iou_threshold"]], dtype=np.float32)
        if self.nms_score_name and rp["score_threshold"] is not None:
            feeds[self.nms_score_name] = np.array([rp["score_threshold"]], dtype=np.float32)
        if self.topk_k_name and rp["TopK_k"] is not None:
            feeds[self.topk_k_name] = np.array([rp["TopK_k"]], dtype=np.int64)

        tensors = self.session.run(self.output_names, feeds)
        out = self._post_name_map(tensors)
        out["frame"] = img_in
        return out

def iou_xyxy(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_b = max(0.0, x4 - x3) * max(0.0, y4 - y3)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0

def nms_keep_indices(boxes, scores, iou_thr=0.5):
    if boxes is None or len(boxes) == 0:
        return []
    order = np.argsort(-scores)  # Higher score preferred
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1: break
        rest = order[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest], dtype=np.float32)
        order = rest[ious <= iou_thr]
    return keep

def filter_by_min(boxes, min_area=0, min_side=0):
    if len(boxes) == 0:
        return np.array([], dtype=bool)
    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]
    area_mask = (w*h) >= float(min_area) if min_area > 0 else np.ones(len(boxes), dtype=bool)
    side_mask = ((w >= float(min_side)) & (h >= float(min_side))) if min_side > 0 else np.ones(len(boxes), dtype=bool)
    return area_mask & side_mask

def _color_bgr(i):
    # Set1 is a high saturation classification color system
    rgb = cm.get_cmap("Set1")(i % 9)[:3]
    return (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))  # transfer to BGR



def draw_overlay(
    img_bgr,                   # original image
    frame_ref,                 # Can be None
    boxes=None,
    scores=None,
    classes=None,
    masks=None,
    score_thr: float = 0.0,
    alpha: float = 0.4,
    draw_box: bool = False,
    coords: str = "orig",      # "orig" indicates that boxes/masks are in the original image coordinates."
):
    """
    Render the overlay on img_bgr:
        - if coords="orig" : boxes/masks are considered to be in the img_bgr coordinate system (recommended)
        - If coords="frame" : It will be scaled to the original image by frame_ref→img_bgr
        - frame_ref may be None; In this case, coords="frame" falls back to the same size as img_bgr
    """
    if img_bgr is None:
        raise ValueError("img_bgr must not be None")
    canvas = img_bgr.copy()
    Hc, Wc = canvas.shape[:2]

    # Size of reference
    if frame_ref is None:
        Hf, Wf = Hc, Wc
    else:
        Hf, Wf = frame_ref.shape[:2]

    # Filtering threshold
    keep_idx = None
    if scores is not None and len(scores) > 0 and score_thr > 0:
        keep_idx = np.where(np.asarray(scores) >= float(score_thr))[0]

        def kpick(x):
            if x is None: return None
            try:
                return x[keep_idx]
            except Exception:
                return [x[i] for i in keep_idx]
        boxes   = kpick(boxes)
        scores  = kpick(scores)
        classes = kpick(classes)
        masks   = kpick(masks)

    # === Unify boxes to original coordinates ===
    if boxes is not None:
        boxes = np.asarray(boxes).astype(np.float32)
        if coords == "frame" and (Hf != Hc or Wf != Wc):
            sx = Wc / float(Wf)
            sy = Hc / float(Hf)
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

    # === Color only on the instance mask ===
    if masks is not None:
        overlay = canvas.copy()
        color_pool = [
            (0, 180, 255), (0, 220, 0), (255, 160, 0),
            (255, 0, 160), (128, 0, 255), (0, 255, 255)
        ]

        N = len(masks) if isinstance(masks, list) else (
            masks.shape[0] if hasattr(masks, "shape") and masks.ndim >= 3 else 1)
        for i in range(N):
            # Extract mask -> squeeze the current instance into 2D
            m = masks[i] if N > 1 else masks
            m = np.asarray(m).squeeze()

            # Binarization
            if m.dtype != np.uint8:
                m = (m > 0.5).astype(np.uint8)
            else:
                m = (m > 0).astype(np.uint8)

            # Goal: Construct a full binary image of the same size as the overlay,
            # with only 1 pixels in the real mask
            full = np.zeros((Hc, Wc), dtype=np.uint8)

            # If there is a bbox and m appears to be a small ROI cropped to the bbox size,
            # paste back to the full image
            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = [int(round(v)) for v in boxes[i]]
                x1 = max(0, min(Wc - 1, x1));
                x2 = max(0, min(Wc, x2))
                y1 = max(0, min(Hc - 1, y1));
                y2 = max(0, min(Hc, y2))
                bw, bh = max(0, x2 - x1), max(0, y2 - y1)

                if bw > 0 and bh > 0:
                    # ask is the whole graph size -> just use the whole graph
                    if m.shape[:2] == (Hc, Wc):
                        full = m.astype(np.uint8)
                    else:
                        # mask is ROI (bh,bw) or other size -> resize to (bw,bh) and paste back to bbox
                        m_roi = m
                        if m_roi.shape[:2] != (bh, bw):
                            m_roi = cv2.resize(m_roi.astype(np.uint8), (bw, bh), interpolation=cv2.INTER_NEAREST)
                        full[y1:y2, x1:x2] = (m_roi > 0).astype(np.uint8)

            else:
                # The case of no bbox
                if m.shape[:2] == (Hc, Wc):
                    full = m.astype(np.uint8)
                else:
                    full = cv2.resize(m.astype(np.uint8), (Wc, Hc), interpolation=cv2.INTER_NEAREST)

            # alpha blending is applied only to pixels with full==1
            if np.any(full):
                color = color_pool[
                    (int(classes[i]) if classes is not None and i < len(classes) else i) % len(color_pool)]
                idx = full.astype(bool)
                overlay[idx] = (
                        overlay[idx].astype(np.float32) * (1 - alpha)
                        + np.array(color, dtype=np.float32) * alpha
                ).astype(np.uint8)

                # Draw mask outline (not bbox)
                cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cv2.drawContours(overlay, cnts, -1, color, 2)

        canvas = overlay

    return canvas


def build_merged_mask(frame_ref, boxes, masks):
    Hf, Wf = frame_ref.shape[:2]
    merged = np.zeros((Hf, Wf), dtype=np.uint8)
    if boxes is None or masks is None or len(boxes) == 0:
        return merged
    N = boxes.shape[0]
    for i in range(N):
        m = masks[i]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        m_f = m.astype(np.float32)
        if m_f.max() > 1.0 or m_f.min() < 0.0:
            m_f = 1.0 / (1.0 + np.exp(-m_f))
        if m_f.ndim == 2 and m_f.shape != (Hf, Wf):
            x1, y1, x2, y2 = [int(round(v)) for v in boxes[i][:4]]
            x1 = max(0, min(Wf-1, x1)); x2 = max(0, min(Wf-1, x2))
            y1 = max(0, min(Hf-1, y1)); y2 = max(0, min(Hf-1, y2))
            if x2 <= x1 or y2 <= y1: continue
            bw, bh = (x2-x1), (y2-y1)
            m_resized = cv2.resize(m_f, (bw, bh), interpolation=cv2.INTER_LINEAR)
            patch = (m_resized >= 0.5).astype(np.uint8)
            merged[y1:y2, x1:x2] = np.maximum(merged[y1:y2, x1:x2], patch)
        else:
            m_map = m_f if m_f.shape == (Hf, Wf) else cv2.resize(m_f, (Wf, Hf), interpolation=cv2.INTER_LINEAR)
            merged = np.maximum(merged, (m_map >= 0.5).astype(np.uint8))
    return merged * 255

