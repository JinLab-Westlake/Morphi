import json, os

CONFIG_FILE = "config.json"

def load_params():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # 默认值
    return {
        "seg_score_thr": 0.3,
        "seg_nms_iou": 0.1,
        "seg_min_area": 0,
        "seg_min_side": 0,
        "seg_output_dir": None,
    }

def save_params(params: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(params, f, indent=2)
