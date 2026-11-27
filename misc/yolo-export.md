



```
conda activate zwq-torch-gpu
python.exe main.py  --input_dir E:\ChromeDownload\sudoku--1102  --output_dir  E:\ChromeDownload\sudoku--1102-1104-2
# 1/8, OK


python.exe main.py  --input_dir  E:\ChromeDownload\1002--crop --output_dir  E:\ChromeDownload\1002--crop-1104

python.exe main.py --input_dir E:\ChromeDownload\Sudoku--all\img-cropped_output --output_dir E:\ChromeDownload\Sudoku--all\img-cropped_output-det --debug_dir E:\ChromeDownload\Sudoku--all\img-cropped_output-debug-1105
```

```

# yolo pt -> onnx
yolo export model=path/to/your_model.pt format=onnx imgsz=640 simplify=True dynamic=True
yolo export model=/home/xhai/wqzh/A100/PdfPageCheck/tmp_wgts/best.pt format=onnx imgsz=640 simplify=True dynamic=True
imgsz=640：指定导出模型的输入分辨率。
simplify=True：调用 ONNX Simplifier 优化计算图。
dynamic=True：允许模型接受动态输入尺寸（适用于视频流或分辨率不固定的场景）。
导出成功后，ONNX 文件通常会自动保存在原 .pt 文件所在的同级目录下。


from ultralytics import YOLO
# 加载一个模型，路径为 YOLO 模型的 .pt 文件
model = YOLO("/home/xhai/wqzh/A100/PdfPageCheck/tmp_wgts/best.pt")
# 导出模型，格式为 ONNX
model.export(format="onnx")



# 导出模型，设置多种参数
model.export(
    format="onnx",      # 导出格式为 ONNX
    imgsz=(640, 640),   # 设置输入图像的尺寸
    keras=False,        # 不导出为 Keras 格式
    optimize=False,     # 不进行优化 False, 移动设备优化的参数，用于在导出为TorchScript 格式时进行模型优化
    half=False,         # 不启用 FP16 量化
    int8=False,         # 不启用 INT8 量化
    dynamic=False,      # 不启用动态输入尺寸
    simplify=True,      # 简化 ONNX 模型
    opset=None,         # 使用最新的 opset 版本
    workspace=4.0,      # 为 TensorRT 优化设置最大工作区大小（GiB）
    nms=False,          # 不添加 NMS（非极大值抑制）
    batch=1,            # 指定批处理大小
    device="cpu"        # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"
)

====

12M     model-tmp/yolov8-best.onnx
12M     model-tmp/yolov8-best.onnx.data
6.0M    model-tmp/yolov8-best.pt



```

## QAs

### simplify=True dynamic=True 由什么优缺点？
`simplify=True` 和 `dynamic=True` 这两个参数各有明确的适用场景与取舍，它们解决的问题不同，优缺点也相互独立。

### 🔧 `simplify=True`：优化计算图

这个参数会在导出后调用 ONNX Simplifier（或类似工具）对计算图进行优化，主要手段是常量折叠和删除冗余算子。

**优点：**
*   **减小模型体积**：通过消除不必要的节点和常量，模型文件通常会变小。
*   **潜在的性能提升**：更简洁的计算图在某些推理引擎上可能跑得更快，因为减少了计算开销。
*   **提高兼容性**：简化的图结构有时能避免特定推理后端（如某些版本的 TensorRT）的解析问题。

**缺点与注意点：**
*   **收益可能有限**：有测试指出，对于某些模型，`simplify` 对体积的缩减和速度的提升“极其微不足道”，甚至可能因为优化过程本身而“轻微降低运行速度”。
*   **依赖额外包**：启用此功能需要环境中安装了 `onnxsim` 或 `onnxslim` 包，否则可能报错。
*   **并非默认开启**：在部分 Ultralytics 版本中，`simplify` 的默认值为 `False`，需要手动指定。

### 📐 `dynamic=True`：启用动态输入

这个参数允许导出的 ONNX 模型接受非固定的输入尺寸（例如可变的分辨率或批大小），而不是将输入形状“锁死”在导出时指定的 `imgsz` 上。

**优点：**
*   **灵活性高**：模型可以处理不同尺寸的图片或可变的批大小（Batch Size），非常适合输入分辨率不固定的视频流或需要动态批处理的场景。
*   **精度对比更公平**：在进行 PyTorch 与 ONNX 的精度对齐测试时，`dynamic=True` 能避免因固定尺寸导出时采用的填充（Padding）策略不同而引入的微小差异，使对比结果更纯粹（差值可低至 0.0000）。

**缺点与注意点：**
*   **推理延迟可能增加**：动态形状意味着推理引擎无法在编译/加载阶段针对特定尺寸做极致的优化（如算子融合）。对于**输入尺寸固定**的场景，`dynamic=False` 通常能获得更低的推理延迟。
*   **并非所有后端都完美支持**：某些推理框架或硬件后端对动态形状的支持可能不完善，或者优化效果不如静态形状。
*   **可能影响模型体积**：有测试显示，单独开启 `dynamic` 可能会轻微减小模型体积，但同时也可能轻微降低运行速度。

### 💎 如何选择

你可以根据部署场景来决定：

1.  **输入尺寸固定且追求极致性能**：使用 **`dynamic=False`**（默认值）。这是最稳妥、通常也是延迟最低的选择。如果模型在特定后端解析报错，可以尝试加上 `simplify=True`。
2.  **输入尺寸或批大小不固定**：必须使用 **`dynamic=True`**。此时为了兼容性和潜在的图优化，可以同时开启 `simplify=True`。需要注意的是，在动态形状下，`simplify` 的优化效果可能不如静态形状下显著。


## Errors
运行的时候，
python.exe main.py  --input_dir  E:\ChromeDownload\1002--crop --output_dir  E:\ChromeDownload\1002--crop-1104

有一些没有检测到sudoku的网格，可能是utils.py 代码的检测的鲁棒性不够。而且在处理第7张图片的时候，卡死了。

请基于以上的信息，生成更加鲁棒的代码

```
[2/103] Processing '06785fdcabf11fae.jpg'...
  [ERROR] Unexpected error: Unable to find the required number of cells in image.
[3/103] Processing '07bb01c8d955513c.jpg'...
  [ERROR] Unexpected error: Unable to find the required number of cells in image.
[4/103] Processing '08b8baef75c1921d.jpg'...
  [ERROR] Unexpected error: No grid contour candidates were found in image
[5/103] Processing '0a8375eee99cf647.jpg'...
  [ERROR] Unexpected error: Unable to find the required number of cells in image.
[6/103] Processing '0ab69e46f6d35df7.jpg'...
  [ERROR] Unexpected error: Unable to find the required number of cells in image.
[7/103] Processing '0cf5ea0b7e3a8d4d.jpg'...
Traceback (most recent call last):
  File "C:\HF-Models\sudoku_solver\main.py", line 125, in <module>
    main()
  File "C:\HF-Models\sudoku_solver\main.py", line 106, in main
    annotated_img = solve_sudoku_puzzle(loaded_model, img_fpath)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\HF-Models\sudoku_solver\main.py", line 45, in solve_sudoku_puzzle
    solver.solve()
  File "C:\HF-Models\sudoku_solver\utils.py", line 122, in solve
    if self.solve():
       ^^^^^^^^^^^^
  File "C:\HF-Models\sudoku_solver\utils.py", line 122, in solve
    if self.solve():
       ^^^^^^^^^^^^
  File "C:\HF-Models\sudoku_solver\utils.py", line 122, in solve
    if self.solve():
       ^^^^^^^^^^^^
  [Previous line repeated 40 more times]
  File "C:\HF-Models\sudoku_solver\utils.py", line 107, in solve
    next_empty_pos = self.find_next_empty()
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\HF-Models\sudoku_solver\utils.py", line 57, in find_next_empty
    if self.board[row][col] == empty_val:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

