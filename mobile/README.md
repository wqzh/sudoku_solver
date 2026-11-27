# 数独识别求解 · 移动端（mobile）

本目录是「数独识别求解」的手机端实现工作区，功能与当前 `app/` 网页版保持一致：
**拍照/选图 → 检测数独网格 → 识别数字 → 自动求解 → 把答案叠回原图**。

**当前进度：阶段 1（目录骨架 + 模型权重）已完成，应用源码尚未开始编写。**
后续阶段会在本目录实现移动端源码，并输出一个可交互的 HTML 展示页面。

---

## 1. 模型权重（已完成）

| 用途 | 文件 | 大小 | 来源 | SHA256 |
|---|---|---|---|---|
| 网格检测（YOLOv8 单类） | `models/grid_detector_yolov8n_640_int8.onnx` | 3.28 MB | `det_model/best.onnx-quant-ir9.onnx` | `67A1E0DF…42FA7` |
| 数字识别（1–9） | `models/digit_classifier_28x28.onnx` | 132 KB | `src_onnx/model.onnx` | `448A017C…6F16A0` |

机器可读的完整规格（输入/输出、量化方式、前后处理参数、基线结果）见
`models/manifest.json`。

选用这两个文件的原因：它们正是 `app/server.py` 线上使用的同一对权重
（`DIGIT_MODEL_PATH` / `DET_MODEL_PATH`），体积小（合计约 3.4 MB），
且检测模型已做动态 INT8 量化，适合随移动端包体分发。

`det_model/yolov8-best.onnx` 是 FP32 原始模型，且带外部权重文件
（`.onnx.data`，合计约 24 MB），不适合手机端，**未复制**。

## 2. 推理规格（实现时必须对齐）

### 2.1 网格检测器

- 输入 `images`：`float32[1,3,640,640]`，NCHW，取值范围 0–1。
- 预处理：等比缩放到 640 内 → 用 (114,114,114) 填充到 640×640 → RGB → `/255`。
- 输出 `output0`：`float32[1,5,8400]`（YOLOv8 head，4 个框参数 + 1 个类别分数，无 objectness）。
- 后处理：转置为 `8400×5`；`conf ≥ 0.25`；IoU NMS `0.45`；再把坐标反 letterbox 回原图。

### 2.2 数字识别器

- 输入 `input`：`float32[batch,28,28,1]`，NHWC，**0–255，不做归一化**。
- 图像格式：**白字黑底**的 28×28 掩膜。
- 输出 `output`：`float32[batch,9]`，对应数字 1–9（没有空白类），预测值 = `argmax + 1`。
- 空白格在推理前由图像处理过滤，不送进模型。

### 2.3 与现有管线对应的常量

| 常量 | 值 | 出处 |
|---|---|---|
| 工作宽度 | 1000 px | `app/pipeline.py: WORK_WIDTH` |
| 网格边长（透视校正后） | 720 px（9 × 80） | `sudoku_utils: CELL_SIZE=80` |
| 白边 | 输入图 24 px，校正图 10 px | `IMG_PAD` / `WARP_PAD` |
| 最少已知数字 | 17 | `MIN_GIVENS` |
| 单格边距擦除比例 | 8% | `CELL_BORDER_FRACTION` |
| 新增数字颜色 | 红 (230, 20, 20) | `pipeline.FILL_COLOR` |
| 判定为数字 | 连通域面积 ≥ 1.5% 且高度 ≥ 30% 格高 | `_process_cell` |

## 3. 参考基线（用于后续一致性校验）

用 Python 版管线（ONNX Runtime CPU）跑 `../.test_input/15.jpg` 的结果已记录：

- `ok=true`，网格得分 `0.9935`，旋转 `0°`，已知数字 `25`，耗时 `1.14 s`。
- 完整盘面与解答见 `docs/baseline.json`。

移动端实现完成后应复现同样的盘面与解答，这是最直接的验收标准。

## 4. 移动端技术路线（下一阶段执行）

推荐 **纯前端 PWA（HTML/CSS/JS + onnxruntime-web WASM）**，无后端：

- 两个模型都能在浏览器内直接跑，拍照/选图用 `<input capture>` 或 `getUserMedia`，答案直接画在 `<canvas>` 上，整条链路可离线。
- 后续的「HTML 展示页」可以和 App 界面共用同一套核心代码，展示页里就是真实推理，而不是假动画。
- 关键点：`app/pipeline.py` 里的透视变换、单元格切分、网格线擦除、求解与纠错重试需要用 JS 重写；
  OpenCV.js 约 8 MB，除非必须，否则**不引入**，用 Canvas + 类型化数组实现
  （YOLO 已经给出四边形，经典轮廓搜索可以省略，这是移动端最大的简化点）。

需要提前验证的兼容性风险：

1. 检测模型是 **opset 22 / IR 10**，需确认目标 onnxruntime-web 版本能加载；不行则在桌面端转存为 opset 17。
2. INT8 量化算子（`ConvInteger` / `MatMulInteger`）在 WASM 后端的支持度，失败则回退到 FP32 模型（先导出 320/640 单文件版）。
3. 手机端大图要先缩到 1000 px 宽再处理，避免 WASM 内存与耗时超限；单张目标 < 3 s。

## 5. 目录规划（后续实现时填充）

```
mobile/
├─ models/                  # 模型权重 + manifest（已完成）
├─ docs/                    # 规格、基线、验收记录
├─ src/
│  ├─ core/                 # 纯算法，无 UI 依赖
│  │  ├─ detector.js        # YOLOv8 前后处理 + NMS
│  │  ├─ warp.js            # 四边形排序、透视变换
│  │  ├─ cells.js           # 81 格切分、网格线擦除、28×28 掩膜
│  │  ├─ digit.js           # 批量识别 + 候选分数
│  │  ├─ solver.js          # 位运算回溯、最少候选优先、纠错重试
│  │  └─ pipeline.js        # 串起上述步骤，含 4 个旋转尝试
│  └─ app/                  # 界面与状态（拍照、裁剪、结果叠加、下载）
├─ index.html               # 移动端应用入口
└─ demo.html                # 可交互展示页（示例图 / 相册 / 截图体验）
```

## 6. 验收口径

- 对 `../.test_input/15.jpg` 的结果与 `docs/baseline.json` 完全一致（盘面、解答、25 个已知数字）。
- 展示页在手机尺寸视口下无横向滚动，上传→求解→查看答案全流程可点通。
- 加载 3.4 MB 权重不阻塞首屏；推理过程有明确进度与失败原因提示（未检测到网格 / 识别数字不足 / 无解）。
