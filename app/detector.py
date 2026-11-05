"""YOLOv8 ONNX sudoku-grid detector wrapper.

Loading / pre / post processing is adapted from
``det_model/load_onnx_ir9_det.py`` (the model is a single-class YOLOv8
detector trained on sudoku grids).
"""

import ast
import os
import threading

import cv2
import numpy as np
import onnxruntime as ort

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DEFAULT_CLASS_NAMES = {0: "sudoku"}


def letterbox(img, new_shape=640):
    """Resize keeping the aspect ratio and pad to new_shape x new_shape."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, (left, top)


def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return np.ascontiguousarray(np.expand_dims(img_chw, axis=0))


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def _parse_names(meta):
    raw = meta.get("names")
    if raw:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict) and parsed:
                return {int(k): str(v) for k, v in parsed.items()}
        except (ValueError, SyntaxError):
            pass
    return dict(DEFAULT_CLASS_NAMES)


class GridDetector:
    """Thread-safe, lazily-created ONNX session for the sudoku detector."""

    def __init__(self, model_path, providers=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"detector model not found: '{model_path}'")
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = _parse_names(self.session.get_modelmeta().custom_metadata_map)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def detect(self, img_bgr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD):
        """Return a list of dicts: {box:[x1,y1,x2,y2], score, class_id, label}."""
        h, w = img_bgr.shape[:2]
        padded, ratio, pad = letterbox(img_bgr, INPUT_SIZE)
        tensor = preprocess(padded)
        with self._lock:
            outputs = self.session.run(None, {self.input_name: tensor})
        return self.postprocess(outputs[0], (h, w), ratio, pad, conf, iou)

    # ------------------------------------------------------------------ #
    def postprocess(self, output, orig_shape, ratio, pad, conf_thres, iou_thres):
        preds = np.asarray(output)
        if preds.ndim == 3:
            preds = preds[0]
        if preds.shape[0] < preds.shape[1]:      # (84, N) -> (N, 84)
            preds = preds.T
        if preds.shape[0] == 0:
            return []

        scores = np.max(preds[:, 4:], axis=1)
        keep = scores > conf_thres
        preds, scores = preds[keep], scores[keep]
        if len(preds) == 0:
            return []
        class_ids = np.argmax(preds[:, 4:], axis=1)

        boxes = xywh2xyxy(preds[:, :4])
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                   conf_thres, iou_thres)

        results = []
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x1, y1, x2, y2 = boxes[i]
                x1 = (x1 - pad[0]) / ratio
                y1 = (y1 - pad[1]) / ratio
                x2 = (x2 - pad[0]) / ratio
                y2 = (y2 - pad[1]) / ratio
                x1 = max(0, min(float(x1), orig_shape[1]))
                y1 = max(0, min(float(y1), orig_shape[0]))
                x2 = max(0, min(float(x2), orig_shape[1]))
                y2 = max(0, min(float(y2), orig_shape[0]))
                cls_id = int(class_ids[i])
                results.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(scores[i]),
                    "class_id": cls_id,
                    "label": self.class_names.get(cls_id, str(cls_id)),
                })
        results.sort(key=lambda d: -d["score"])
        return results
