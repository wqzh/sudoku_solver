[中文](README.md)|[English](README_EN.md)

# Sudoku Solver · 数独识别与求解

用两个本地小模型完成 **「上传/拍摄图片 → 定位数独棋盘 → 识别数字 → 求解 → 回填答案」** 的完整流程：

| 环节 | 模型 | 说明 |
| --- | --- | --- |
| 棋盘检测定位 | `models/best.onnx-quant-ir9.onnx` | YOLOv8 单类检测器，已量化为 ONNX（IR9），CPU 上即可快速推理 |
| 数字识别 | `models/digit_recognize_model.onnx` | 28×28 单字分类，输出 1~9 的类别分数 |

两个模型都经过量化，依赖 `onnxruntime` 即可运行；装有 `onnxruntime-gpu` 时数字识别会自动使用 CUDA，检测器固定留在 CPU（量化模型在 CPU 上更快更稳）。

除模型外，网格定位、透视矫正、格子切分、数独求解全部是纯 OpenCV + NumPy 实现，不依赖 YOLO 的 Python 包，也不需要深度学习框架。

## 目录结构

```
sudoku_solver/
├── app/                        # Web 版（FastAPI + 静态页面）
│   ├── server.py               # 服务入口：/、/api/health、/api/solve
│   ├── pipeline.py             # Web 侧流水线：检测框 + 经典轮廓双路候选，结果回填到原图
│   ├── detector.py             # YOLOv8 ONNX 检测器封装（letterbox / NMS / 坐标还原）
│   ├── sudoku_utils.py         # 核心算法：网格候选、透视矫正、格子切分、数字识别、求解器
│   ├── static/                 # 前端页面（index.html / app.js / style.css，无需构建）
│   ├── requirements.txt        # app 目录所需 Python 依赖
│   └── website_demo.png        # 界面截图
├── main.py                     # 命令行批量版：遍历目录内图片，逐张求解并保存结果
├── utils.py                    # 命令行版使用的算法实现（与 app/sudoku_utils.py 同源）
├── models/                  # 模型与导出/量化脚本
│   ├── best.onnx-quant-ir9.onnx    # 棋盘检测（量化，部署用）
│   ├── digit_recognize_model.onnx  # 数字识别（部署用）
│   ├── export_yolopt_to_onnx.py    # .pt -> .onnx 导出
│   ├── onnx_quant_ir9.py           # ONNX 量化（IR9）
│   └── load_onnx_ir9_det.py        # 量化检测模型的加载/推理参考实现
├── misc/                       # 历史版本与实验代码（src_tf / src_onnx、YOLO 导出笔记）
└── mobile/                     # Capacitor 手机端（未纳入版本库，见 .gitignore）
```

> `mobile/` 为独立的手机端工程（Capacitor + Android），体积较大且不属于本仓库提交内容，已在 `.gitignore` 中整体排除。

## 环境要求

- Python 3.9+（推荐 3.10 / 3.11）
- 无需 GPU；有 NVIDIA 显卡时可安装 `onnxruntime-gpu` 加速数字识别

## 安装

```bash
# 建议使用独立环境
conda create -n sudoku python=3.11 -y
conda activate sudoku

pip install -r app/requirements.txt
# GPU 版（可选，与 onnxruntime 二选一）：
# pip uninstall -y onnxruntime && pip install onnxruntime-gpu
```

## Web 版运行

```bash
python app/server.py --host 127.0.0.1 --port 8000
# 开发时自动重载：
# python app/server.py --host 127.0.0.1 --port 8000 --reload
```

打开 <http://127.0.0.1:8000> ，拖入或选择一张数独图片即可。页面无需构建步骤，`app/static/` 直接被服务。

<img src="app/website_demo.png" alt="Web 界面截图" style="display: block; margin: 0 auto;">

### 接口说明

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 前端页面 |
| GET | `/api/health` | 健康检查，返回模型是否加载成功 |
| POST | `/api/solve` | 表单字段 `file` 上传图片，返回 JSON |

`/api/solve` 返回字段：

```jsonc
{
  "ok": true,                 // 是否成功求解
  "message": "已求解：网格得分 0.93，旋转 0°，原始数字 28 个，新增 53 个",
  "seconds": 1.27,            // 服务端耗时
  "width": 1000, "height": 1000,
  "boxes": [                  // 检测到的棋盘框
    {"x": 12, "y": 8, "w": 976, "h": 980, "score": 0.94, "label": "sudoku"}
  ],
  "givens": 28,               // 识别出的原始数字个数
  "filled": 53,               // 求解补充的数字个数
  "board": [[...9x9...]],     // 识别出的题面（0 表示空格）
  "solved_board": [[...9x9...]], // 求解结果
  "image": "data:image/png;base64,..." // 回填答案后的图片（与原图同尺寸）
}
```

失败时 `ok=false`，`message` 给出原因（未检测到网格 / 识别数字不足 17 个 / 无解 / 超时等），HTTP 状态码仍可能为 200，前端按 `ok` 判断即可。

## 命令行批量版

```bash
python main.py --input_dir ./samples --output_dir ./out \
               --model_fpath models/digit_recognize_model.onnx
```

| 参数 | 说明 |
| --- | --- |
| `--input_dir` | 待处理图片目录（jpg/png/bmp/tiff/webp） |
| `--output_dir` | 结果图片保存目录 |
| `--model_fpath` | 数字识别模型路径，默认 `src_onnx/model.onnx`（**旧路径，请显式指定为 `models/digit_recognize_model.onnx`**） |
| `--debug_dir` | 可选，求解失败的图片额外输出调试图 |
| `--time_budget` | 单张图片最大耗时（秒），默认 20 |

> 注意：`main.py` 中默认值仍是旧的 `src_onnx/model.onnx`，且带有 `if i==5: break` 的调试用提前退出，正式批量处理前请自行调整。

## 工作流程

1. **候选网格**：图像四周先补 24px 白边，保证贴边的棋盘有闭合轮廓；随后用 5 种二值化策略（自适应均值/高斯、CLAHE、Otsu、Canny）+ 轮廓/凸包/最小外接矩形生成候选四边形，并把整图作为兜底候选。
2. **候选打分**：每个候选透视变换到 720×720，统计横竖线投影与 9×9 等分线的吻合程度得到网格得分，保留得分最高的若干候选。
3. **格子切分**：在矫正后的棋盘上检测 10 条横线 / 10 条竖线（检测不到时回退到等分），据此切出 81 个格子，并抹掉网格线避免被当成数字。
4. **数字识别**：按连通域判断格子是否有数字，规范化为 28×28 后送入 ONNX 分类模型，得到 9 路分数。
5. **求解**：位运算 + MRV 启发式回溯求解（带节点数与时间上限）。识别数字少于 17 个视为无效；无解时按置信度把最低的几个数字换成次优类别重试，并尝试 90°/180°/270° 旋转。
6. **回填**：只在原本为空的格子里用红色绘制答案，Web 版回填到与原图同样尺寸的画布上。

## 已知问题

- 手机拍摄电脑屏幕产生的**摩尔纹**会导致棋盘检测失败，需要额外做去摩尔纹预处理。
- 手写体、低分辨率、强透视/弯曲书页的数字识别率有限，必要时可先做去畸变。
- 求解器对“识别错一个数字”敏感，目前靠置信度回退策略兜底，复杂题可能重试较久。

## 参考

- 模型导出与量化细节：`misc/yolo-export.md`、`misc/src_onnx/README.md`、`misc/src_tf/README.md`
