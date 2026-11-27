from onnxruntime.quantization import quantize_dynamic, QuantType

# 输入是你现有的 .onnx 文件（确保 .onnx.data 在同目录）
model_fp32 = "/home/xhai/wqzh/A100/ultralytics-sudoku/models/best.onnx"
# 输出是期望的量化版文件
model_quant = "/home/xhai/wqzh/A100/ultralytics-sudoku/models/best.onnx-quant-ir9.onnx"

# 执行动态量化，将权重转为 INT8
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_quant,
    weight_type=QuantType.QInt8,
)

print("量化完成，输出文件：", model_quant)