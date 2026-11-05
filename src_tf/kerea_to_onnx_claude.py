"""不依赖 TensorFlow：把 Keras 模型文件（.keras zip 格式 或 旧版 HDF5 格式）转换为 .onnx
依赖: pip install numpy h5py onnx onnxruntime(可选，仅验证用)
用法: python keras_to_onnx.py model.keras model.onnx
"""
import json, os, re, sys, tempfile, zipfile
import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# ============================================================ 读取模型文件
def _dec(x):
    return x.decode("utf8") if isinstance(x, bytes) else str(x)


def _snake(name):
    s = re.sub("(.)([A-Z][a-z0-9]+)", r"\1_\2", name)
    return re.sub("([a-z])([A-Z])", r"\1_\2", s).lower()


def _load_legacy_h5(path):
    """旧版 Keras(2.x / tf.keras) HDF5：model_config 在根属性里，权重在 model_weights/<层名>/..."""
    with h5py.File(path, "r") as f:
        cfg = f.attrs.get("model_config")
        if cfg is None:
            raise ValueError("文件里没有 model_config（可能是纯权重文件），需要提供模型结构")
        cfg = json.loads(_dec(cfg))
        root = f["model_weights"] if "model_weights" in f else f
        weights = {}
        for lname in [_dec(n) for n in root.attrs["layer_names"]]:
            g = root[lname]
            names = [_dec(n) for n in g.attrs["weight_names"]]
            weights[lname] = [np.array(g[n]) for n in names]
    return cfg, weights


def _load_keras3_zip(path):
    """Keras 3 的 .keras zip 包：config.json + model.weights.h5"""
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as td:
        cfg = json.loads(z.read("config.json"))
        z.extract("model.weights.h5", td)
        used, weights = {}, {}
        with h5py.File(os.path.join(td, "model.weights.h5"), "r") as f:
            for L in cfg["config"]["layers"]:
                lname, base = L["config"]["name"], _snake(L["class_name"])
                if base in used:
                    used[base] += 1; key = f"{base}_{used[base]}"
                else:
                    used[base] = 0; key = base
                arrs = []
                for k in (key, lname):
                    p = f"layers/{k}/vars"
                    if p in f:
                        g = f[p]
                        arrs = [np.array(g[i]) for i in sorted(g.keys(), key=int)]
                        break
                weights[lname] = arrs
    return cfg, weights


def load_keras_file(path):
    if zipfile.is_zipfile(path):
        return _load_keras3_zip(path)
    if h5py.is_hdf5(path):
        return _load_legacy_h5(path)
    raise ValueError("既不是 zip 也不是 HDF5 文件")


# ============================================================ ONNX 构图工具
class _Builder:
    def __init__(self):
        self.nodes, self.inits, self._n = [], [], 0

    def _uid(self, p):
        self._n += 1
        return f"{p}_{self._n}"

    def const(self, arr):
        name = self._uid("const")
        self.inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    def op(self, op_type, inputs, **attrs):
        out = self._uid(op_type.lower())
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=out, **attrs))
        return out


def _act_name(a):
    if a is None:
        return "linear"
    if isinstance(a, dict):
        c = a.get("config")
        return c if isinstance(c, str) else a.get("class_name", "linear")
    return a


def _activation(b, x, name):
    name, f32 = _act_name(name), np.float32
    if name == "linear":  return x
    if name == "relu":    return b.op("Relu", [x])
    if name == "relu6":   return b.op("Clip", [x, b.const(np.array(0, f32)), b.const(np.array(6, f32))])
    if name == "sigmoid": return b.op("Sigmoid", [x])
    if name == "tanh":    return b.op("Tanh", [x])
    if name == "softmax": return b.op("Softmax", [x], axis=-1)
    if name == "softplus":return b.op("Softplus", [x])
    if name == "elu":     return b.op("Elu", [x], alpha=1.0)
    if name in ("swish", "silu"):
        return b.op("Mul", [x, b.op("Sigmoid", [x])])
    raise NotImplementedError(f"暂不支持的激活函数: {name}")


def _to_nchw(b, x): return b.op("Transpose", [x], perm=[0, 3, 1, 2])
def _to_nhwc(b, x): return b.op("Transpose", [x], perm=[0, 2, 3, 1])


def _dims(shape):
    return ["batch" if (i == 0 and d is None) else (f"d{i}" if d is None else int(d))
            for i, d in enumerate(shape)]


def _inbound_names(L):
    """同时兼容 Keras3 (__keras_tensor__) 与旧版 ([[name, node, tensor, {}], ...]) 的 inbound_nodes"""
    names = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("class_name") == "__keras_tensor__":
                names.append(o["config"]["keras_history"][0]); return
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            if len(o) >= 3 and isinstance(o[0], str) and isinstance(o[1], int):
                names.append(o[0]); return
            for v in o: walk(v)

    nodes = L.get("inbound_nodes", [])
    if len(nodes) > 1:
        raise NotImplementedError(f"层 {L['name'] if 'name' in L else L['config']['name']} 被多次调用（共享层），暂不支持")
    walk(nodes)
    return names


# ============================================================ 层映射
def _convert_layer(b, cls, cfg, w, xs):
    x = xs[0]

    if cls == "Dense":
        y = b.op("MatMul", [x, b.const(w[0])])
        if cfg.get("use_bias", True):
            y = b.op("Add", [y, b.const(w[1])])
        return _activation(b, y, cfg.get("activation"))

    if cls in ("Conv2D", "DepthwiseConv2D"):
        assert cfg.get("data_format", "channels_last") == "channels_last"
        if cls == "Conv2D":
            W, group = np.transpose(w[0], (3, 2, 0, 1)), cfg.get("groups", 1)   # HWIO -> OIHW
        else:
            assert cfg.get("depth_multiplier", 1) == 1
            W = np.transpose(w[0], (2, 3, 0, 1)); group = W.shape[0]            # HWC1 -> C1HW
        ins = [_to_nchw(b, x), b.const(W)]
        if cfg.get("use_bias", True):
            ins.append(b.const(w[1]))
        y = b.op("Conv", ins, kernel_shape=list(cfg["kernel_size"]),
                 strides=list(cfg["strides"]),
                 dilations=list(cfg.get("dilation_rate", (1, 1))), group=group,
                 auto_pad="SAME_UPPER" if cfg["padding"] == "same" else "NOTSET")
        return _activation(b, _to_nhwc(b, y), cfg.get("activation"))

    if cls in ("MaxPooling2D", "AveragePooling2D"):
        pool = list(cfg["pool_size"])
        y = b.op("MaxPool" if cls == "MaxPooling2D" else "AveragePool", [_to_nchw(b, x)],
                 kernel_shape=pool, strides=list(cfg.get("strides") or pool),
                 auto_pad="SAME_UPPER" if cfg["padding"] == "same" else "NOTSET")
        return _to_nhwc(b, y)

    if cls in ("GlobalAveragePooling2D", "GlobalMaxPooling2D"):
        return b.op("ReduceMean" if "Average" in cls else "ReduceMax", [x],
                    axes=[1, 2], keepdims=int(cfg.get("keepdims", False)))

    if cls == "BatchNormalization":
        eps, i, gamma, beta = cfg.get("epsilon", 1e-3), 0, None, None
        if cfg.get("scale", True):  gamma = w[i]; i += 1
        if cfg.get("center", True): beta = w[i]; i += 1
        mean, var = w[i], w[i + 1]
        gamma = np.ones_like(mean) if gamma is None else gamma
        beta = np.zeros_like(mean) if beta is None else beta
        scale = gamma / np.sqrt(var + eps)
        shift = beta - mean * scale
        y = b.op("Mul", [x, b.const(scale.astype(np.float32))])
        return b.op("Add", [y, b.const(shift.astype(np.float32))])

    if cls == "Flatten":
        return b.op("Flatten", [x], axis=1)
    if cls == "Reshape":
        shape = [0] + [int(s) for s in cfg["target_shape"]]
        return b.op("Reshape", [x, b.const(np.array(shape, np.int64))])
    if cls in ("Dropout", "SpatialDropout1D", "SpatialDropout2D", "GaussianNoise"):
        return x
    if cls == "Activation":
        return _activation(b, x, cfg["activation"])
    if cls == "ReLU":
        if cfg.get("max_value") is None and not cfg.get("negative_slope") and not cfg.get("threshold"):
            return b.op("Relu", [x])
        raise NotImplementedError("带参数的 ReLU 层暂不支持")
    if cls == "LeakyReLU":
        return b.op("LeakyRelu", [x], alpha=float(cfg.get("negative_slope", cfg.get("alpha", 0.3))))
    if cls == "Softmax":
        return b.op("Softmax", [x], axis=cfg.get("axis", -1))
    if cls == "Rescaling":
        y = b.op("Mul", [x, b.const(np.array(cfg["scale"], np.float32))])
        return b.op("Add", [y, b.const(np.array(cfg["offset"], np.float32))])
    if cls in ("Add", "Multiply"):
        y = xs[0]
        for t in xs[1:]:
            y = b.op("Add" if cls == "Add" else "Mul", [y, t])
        return y
    if cls == "Concatenate":
        return b.op("Concat", xs, axis=cfg.get("axis", -1))

    raise NotImplementedError(f"暂不支持的层类型: {cls}")


# ============================================================ 主函数
def keras_to_onnx(keras_path, onnx_path, opset=13):
    model_cfg, weights = load_keras_file(keras_path)
    kind = model_cfg["class_name"]
    layers = model_cfg["config"]["layers"] if isinstance(model_cfg["config"], dict) else model_cfg["config"]
    b = _Builder()
    graph_inputs = []

    def add_input(name, shape):
        graph_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, _dims(shape)))
        return name

    if kind == "Sequential":
        prev = None
        for L in layers:
            cls, cfg = L["class_name"], L["config"]
            shape = cfg.get("batch_input_shape") or cfg.get("batch_shape")
            if prev is None:
                if shape is None:
                    raise ValueError("找不到输入形状")
                prev = add_input("input", shape)
            if cls == "InputLayer":
                continue
            prev = _convert_layer(b, cls, cfg, weights.get(cfg["name"], []), [prev])
        outputs = [prev]

    elif kind in ("Functional", "Model"):
        tensors = {}
        for L in layers:
            cls, cfg = L["class_name"], L["config"]
            name = cfg["name"]
            if cls == "InputLayer":
                tensors[name] = add_input(name, cfg.get("batch_shape") or cfg.get("batch_input_shape"))
                continue
            xs = [tensors[n] for n in _inbound_names(L)]
            tensors[name] = _convert_layer(b, cls, cfg, weights.get(name, []), xs)
        out_layers = model_cfg["config"]["output_layers"]
        if out_layers and isinstance(out_layers[0], str):
            out_layers = [out_layers]
        outputs = [tensors[o[0]] for o in out_layers]
    else:
        raise NotImplementedError(f"不支持的模型类型: {kind}")

    graph_outputs = []
    for i, t in enumerate(outputs):
        oname = "output" if len(outputs) == 1 else f"output_{i}"
        b.nodes.append(helper.make_node("Identity", [t], [oname], name=f"out_{i}"))
        graph_outputs.append(helper.make_tensor_value_info(oname, TensorProto.FLOAT, None))

    graph = helper.make_graph(b.nodes, "keras_model", graph_inputs, graph_outputs, initializer=b.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)   # 自动补全输出的 shape
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print("已保存:", onnx_path)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "models/model_15_epochs_font_mnist.keras"
    dst = sys.argv[2] if len(sys.argv) > 2 else "model.onnx"
    keras_to_onnx(src, dst)
    import onnxruntime as ort
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.run(None, {inp.name: np.random.rand(1, 28, 28, 1).astype(np.float32)})[0]
    print("输入:", inp.name, inp.shape, "输出形状:", out.shape, "softmax 和:", out.sum())