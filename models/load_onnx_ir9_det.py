import cv2
import numpy as np
import onnxruntime as ort

# ================= 配置参数 =================
MODEL_PATH = "det_model/best.onnx-quant-ir9.onnx"
IMAGE_PATH = ".test_input/15.jpg"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# COCO 80 类名称（YOLOv8 默认）
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def letterbox(img, new_shape=640):
    """等比例缩放并填充到 new_shape x new_shape，返回处理后的图、缩放比和 padding。"""
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

def preprocess(img):
    """BGR -> RGB, HWC -> CHW, 归一化, 增加 batch 维。"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    return np.ascontiguousarray(img_batch)

def xywh2xyxy(x):
    """中心点宽高 -> 左上右下。"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def postprocess(output, orig_shape, ratio, pad, conf_thres, iou_thres):
    """
    YOLOv8 输出形状通常为 (1, 84, 8400)：
    前 4 为 xywh，后 80 为类别分数（无 objectness）。
    量化模型输出可能仍为 float，需确认。
    """
    # (1, 84, 8400) -> (8400, 84)
    preds = output[0].T
    
    # 过滤低置信度
    scores = np.max(preds[:, 4:], axis=1)
    mask = scores > conf_thres
    preds = preds[mask]
    scores = scores[mask]
    class_ids = np.argmax(preds[:, 4:], axis=1)
    
    if len(preds) == 0:
        return []
    
    # xywh -> xyxy
    boxes = xywh2xyxy(preds[:, :4])
    
    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_thres, iou_thres
    )
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, x2, y2 = boxes[i]
            # 去除 padding 并还原到原图尺寸
            x1 = (x1 - pad[0]) / ratio
            y1 = (y1 - pad[1]) / ratio
            x2 = (x2 - pad[0]) / ratio
            y2 = (y2 - pad[1]) / ratio
            
            # 裁剪到原图边界
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))
            
            results.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(scores[i]),
                "class_id": int(class_ids[i])
            })
    return results

def main():
    # 1. 加载 ONNX 模型
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # 2. 读取图片
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {IMAGE_PATH}")
    orig_h, orig_w = img.shape[:2]
    
    # 3. 预处理
    padded, ratio, pad = letterbox(img, INPUT_SIZE)
    input_tensor = preprocess(padded)
    
    # 4. 推理
    outputs = session.run(None, {input_name: input_tensor})
    
    # 5. 后处理
    detections = postprocess(
        outputs[0], (orig_h, orig_w), ratio, pad,
        CONF_THRESHOLD, IOU_THRESHOLD
    )
    
    # 6. 绘制检测框
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["score"]
        cls_id = det["class_id"]
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        
        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 画标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 7. 显示与保存
    # cv2.imshow("YOLOv8 ONNX Detection", img)
    cv2.imwrite("output_detection.jpg", img)
    print(f"检测到 {len(detections)} 个目标，结果已保存为 output_detection.jpg")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()