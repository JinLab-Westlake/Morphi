# seg_infer.py
import os, cv2, numpy as np, onnxruntime as ort

class SurvivalSegmentor:
    """
    专门跑 survival onnx 的推理器：
    - 读取 onnx 输入形状，固定把图像 letterbox 到 (H,W)
    - 运行后把输出映射回原图大小
    - 输出字典：{"death_prob": float32[H0,W0], "mask": uint8[H0,W0](可选)}
    """
    def __init__(self, onnx_path, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        ishape = self.session.get_inputs()[0].shape  # [1, C, H, W]
        # 若导出时是动态，兜底成 224
        self.C = int(ishape[1]) if isinstance(ishape[1], int) else 3
        self.H = int(ishape[2]) if isinstance(ishape[2], int) else 224
        self.W = int(ishape[3]) if isinstance(ishape[3], int) else 224

    # --------- 预/后处理 ----------
    def _letterbox(self, img):
        h0, w0 = img.shape[:2]
        scale = min(self.W / w0, self.H / h0)
        nw, nh = int(round(w0*scale)), int(round(h0*scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top = (self.H - nh) // 2
        left = (self.W - nw) // 2
        bottom = self.H - nh - top
        right  = self.W - nw - left
        pad = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=0)
        return pad, (scale, left, top, w0, h0, nw, nh)

    def _unletterbox(self, arr_hw, info, is_prob=True):
        scale, left, top, w0, h0, nw, nh = info
        crop = arr_hw[top:top+nh, left:left+nw]
        interp = cv2.INTER_LINEAR if is_prob else cv2.INTER_NEAREST
        return cv2.resize(crop, (w0, h0), interpolation=interp)

    # --------- 推理 ----------
    def predict(self, img_bgr):
        # 1) 通道处理：按 onnx 输入通道数来
        if self.C == 1:
            img_in = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)[..., None]  # H×W×1
        else:
            img_in = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)              # H×W×3

        # 2) 固定到 (H,W)
        pad, info = self._letterbox(img_in)          # → H×W×C
        x = pad.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]         # 1×C×H×W

        # 3) 推理（兼容不同输出命名/数量）
        outs = self.session.run(self.output_names, {self.input_name: x})

        # 把输出做成 name->ndarray 的字典
        name2out = {n: o for n, o in zip(self.output_names, outs)}
        # 可能的情况：
        #   a) ["logits", "death_prob"]
        #   b) ["death_prob"] only
        #   c) 无名 or 乱名，只能靠形状猜
        death = None
        logits = None
        # 优先按名字
        if "death_prob" in name2out:
            death = name2out["death_prob"]        # 1×1×H×W
        if "logits" in name2out:
            logits = name2out["logits"]           # 1×K×H×W

        # 没有规范名字就按形状猜
        if death is None or (isinstance(death, np.ndarray) and death.ndim != 4):
            for o in outs:
                if isinstance(o, np.ndarray) and o.ndim == 4:
                    # 单通道概率图
                    if o.shape[1] == 1:
                        death = o
                    # 多通道更像 logits
                    elif o.shape[1] >= 2:
                        logits = o

        # 4) 回原图尺寸
        result = {}
        if death is not None:
            dp = death[0, 0]                          # H×W
            result["death_prob"] = self._unletterbox(dp, info, is_prob=True)

        if logits is not None:
            if logits.shape[1] == 1:
                prob = 1/(1+np.exp(-logits[:, 0]))
                mask = (prob[0] > 0.5).astype(np.uint8)
                result["mask"] = self._unletterbox(mask, info, is_prob=False)
            else:
                pred = logits.argmax(1)[0].astype(np.uint8)
                result["mask"] = self._unletterbox(pred, info, is_prob=False)

        return result
