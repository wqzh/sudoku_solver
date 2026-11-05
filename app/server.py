"""FastAPI backend for the sudoku web app.

Run from the project root:

    conda activate zwq-torch-gpu
    python app/server.py --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000
"""

import argparse
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import detector as detector_mod          # noqa: E402
import pipeline                           # noqa: E402
import sudoku_utils as sutils             # noqa: E402

DIGIT_MODEL_PATH = os.path.join(PROJECT_ROOT, "src_onnx", "model.onnx")
DET_MODEL_PATH = os.path.join(PROJECT_ROOT, "det_model", "best.onnx-quant-ir9.onnx")
STATIC_DIR = os.path.join(APP_DIR, "static")

app = FastAPI(title="Sudoku Solver", version="1.0.0")

_state = {
    "digit_model": None,
    "detector": None,
    "error": None,
    "loaded_at": None,
}
_load_lock = threading.Lock()


def _load_models():
    """Load both ONNX models once (CUDA when available, CPU otherwise)."""
    if _state["digit_model"] is not None:
        return
    with _load_lock:
        if _state["digit_model"] is not None:
            return
        try:
            providers = None
            try:
                import onnxruntime as ort
                avail = ort.get_available_providers()
                wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                providers = [p for p in wanted if p in avail]
                # quantised detector: keep it on CPU, it is faster & robust there
                _state["detector"] = detector_mod.GridDetector(
                    DET_MODEL_PATH, providers=["CPUExecutionProvider"])
                _state["digit_model"] = sutils.OnnxModel(DIGIT_MODEL_PATH,
                                                         providers=providers or None)
            except Exception:
                _state["detector"] = detector_mod.GridDetector(DET_MODEL_PATH)
                _state["digit_model"] = sutils.OnnxModel(
                    DIGIT_MODEL_PATH, providers=["CPUExecutionProvider"])
            _state["loaded_at"] = time.time()
            _state["error"] = None
        except Exception as exc:
            _state["error"] = f"{exc.__class__.__name__}: {exc}"
            raise


@app.on_event("startup")
def _startup():
    try:
        _load_models()
    except Exception:
        traceback.print_exc()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    return {
        "ok": _state["digit_model"] is not None,
        "error": _state["error"],
        "digit_model": os.path.basename(DIGIT_MODEL_PATH),
        "det_model": os.path.basename(DET_MODEL_PATH),
    }


@app.post("/api/solve")
async def api_solve(file: UploadFile = File(...)):
    try:
        _load_models()
    except Exception as exc:
        return JSONResponse(status_code=500, content={
            "ok": False, "message": f"模型加载失败：{exc}"})

    raw = await file.read()
    if not raw:
        return JSONResponse(status_code=400, content={
            "ok": False, "message": "上传的文件为空"})

    buf = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=400, content={
            "ok": False, "message": "无法解析该图片，请换一张常见格式（jpg/png/bmp/webp）"})
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    t0 = time.monotonic()
    try:
        result = pipeline.solve_image(
            img_rgb, _state["digit_model"], _state["detector"],
            time_budget=25.0, use_det_candidates=True)
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "ok": False, "message": f"处理失败：{exc.__class__.__name__}: {exc}"})

    payload = {
        "ok": bool(result["ok"]),
        "message": result["message"],
        "seconds": round(time.monotonic() - t0, 2),
        "width": w,
        "height": h,
        "boxes": [
            {
                "x": b["box"][0], "y": b["box"][1],
                "w": max(1, b["box"][2] - b["box"][0]),
                "h": max(1, b["box"][3] - b["box"][1]),
                "score": round(b["score"], 4),
                "label": b["label"],
            }
            for b in result["boxes"]
        ],
        "givens": result["givens"],
        "filled": 0 if not result["ok"] else 81 - result["givens"],
        "board": result["board"],
        "solved_board": result["solved_board"],
    }

    annotated = result["annotated"] if result["ok"] else img_rgb
    payload["image"] = pipeline.encode_png_data_url(annotated)
    return payload


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _parse_args():
    ap = argparse.ArgumentParser(description="Sudoku solver web server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"数字识别模型: {DIGIT_MODEL_PATH}")
    print(f"YOLO 检测模型: {DET_MODEL_PATH}")
    print(f"打开 http://{args.host}:{args.port}")
    uvicorn.run("server:app" if args.reload else app, host=args.host,
                port=args.port, reload=args.reload)
