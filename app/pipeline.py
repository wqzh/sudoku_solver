"""Sudoku detection / recognition / solving pipeline used by the web app.

The heavy lifting (grid search, warping, cell slicing, digit recognition,
solver) comes from ``app/sudoku_utils.py`` which is a copy of the project's
top-level ``utils.py``.

Differences from the batch script:
  * the YOLO detector is used both for drawing boxes and for proposing grid
    quads (the classical contour search is still the main path),
  * work is done on a 1000px-wide copy of the image (as in ``main.py``) but the
    digits are rendered back onto the ORIGINAL image, so the returned picture
    always has exactly the same size as the uploaded one.
"""

import time
from typing import Optional

import cv2
import numpy as np

import sudoku_utils as sutils


WORK_WIDTH = 1000

# Digits that were filled in by the solver are drawn in this colour (RGB).
FILL_COLOR = (230, 20, 20)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_rgb(img):
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _scale_quad(quad, scale, pad):
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    return np.float32([[x * scale + pad, y * scale + pad] for x, y in q])


def _dedupe(candidates, diag):
    out = []
    for cand in candidates:
        if all(not sutils._same_quad(cand.quad, c.quad, 0.02 * diag)
               for c in out):
            out.append(cand)
    out.sort(key=lambda c: -c.score)
    return out


def _render_digits(img_rgb, cells, solved_board, M_full, color=FILL_COLOR):
    """Draw the solved digits into the ORIGINAL image (same size / canvas)."""
    out = img_rgb.copy()
    inv = np.linalg.inv(M_full)
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_h = max(1, cv2.getTextSize("8", font, 1.0, 1)[0][1])
    flat = np.asarray(solved_board).reshape(-1)

    for idx, cell in enumerate(cells):
        if cell["contains_digit"]:
            continue
        cx, cy = float(cell["x_centroid"]), float(cell["y_centroid"])

        # local scale of the warped grid in original-image pixels
        probe = np.float32([[[cx - 40, cy], [cx + 40, cy],
                             [cx, cy - 40], [cx, cy + 40]]])
        q = cv2.perspectiveTransform(probe, inv)[0]
        fx = np.linalg.norm(q[1] - q[0]) / 80.0
        fy = np.linalg.norm(q[3] - q[2]) / 80.0
        factor = max(1e-3, (fx + fy) / 2.0)
        cell_px = sutils.CELL_SIZE * factor

        scale = cell_px * 0.58 / base_h
        thickness = max(1, int(round(cell_px / 26)))
        text = str(int(flat[idx]))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

        centre = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), inv)[0][0]
        org = (int(round(centre[0] - tw / 2.0)), int(round(centre[1] + th / 2.0)))
        cv2.putText(out, text, org, font, scale, color, thickness, cv2.LINE_AA)
    return out


def _box_to_quad(box, scale, pad):
    x1, y1, x2, y2 = box
    return np.float32([[x1 * scale + pad, y1 * scale + pad],
                       [x2 * scale + pad, y1 * scale + pad],
                       [x2 * scale + pad, y2 * scale + pad],
                       [x1 * scale + pad, y2 * scale + pad]])


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def solve_image(img_rgb, digit_model, detector=None, time_budget=25.0,
                use_det_candidates=True):
    """Detect, read, solve a sudoku from an RGB image.

    Returns a dict:
        ok, message, annotated (RGB, same shape as input), boxes,
        givens, board (9x9 list), solved_board (9x9 list), grid_score, rotation
    """
    img_rgb = _as_rgb(np.ascontiguousarray(img_rgb))
    orig_h, orig_w = img_rgb.shape[:2]
    t0 = time.monotonic()
    t_end = t0 + time_budget

    result = {
        "ok": False,
        "message": "",
        "annotated": None,
        "boxes": [],
        "givens": 0,
        "board": None,
        "solved_board": None,
        "grid_score": 0.0,
        "rotation": 0,
        "seconds": 0.0,
    }

    # --- detector (drawing + extra grid candidates) ------------------------- #
    dets = []
    if detector is not None:
        try:
            dets = detector.detect(img_rgb)
        except Exception as exc:                      # never fail the solve
            result["message"] = f"detector skipped: {exc}"
    result["boxes"] = dets

    # --- work copy (same as main.py: resize to 1000 px wide) ---------------- #
    work = sutils.resize_and_maintain_aspect_ratio(img_rgb, WORK_WIDTH)
    work_w = work.shape[1]
    s = work_w / float(orig_w)
    p = sutils.IMG_PAD
    work_pad = cv2.copyMakeBorder(work, p, p, p, p, cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))
    S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    T = np.array([[1, 0, p], [0, 1, p], [0, 0, 1]], dtype=np.float64)

    # --- candidates: classical search + YOLO boxes -------------------------- #
    candidates, best_seen = sutils.find_grid_candidates(work_pad)

    if use_det_candidates and dets:
        extra = []
        for det in dets:
            quad = _box_to_quad(det["box"], s, p)
            try:
                quad, score = sutils.refine_quad(work_pad, quad)
            except Exception:
                continue
            if score >= sutils.MIN_GRID_SCORE:
                extra.append(sutils.GridCandidate(quad, float(score)))
        # keep every candidate, best-scoring quad wins on duplicates
        diag = float(np.hypot(*work_pad.shape[:2]))
        candidates = _dedupe(sorted(extra + candidates, key=lambda c: -c.score), diag)

    if not candidates:
        result["message"] = result["message"] or (
            f"未检测到数独网格（最佳网格得分 {best_seen:.2f}）")
        result["seconds"] = time.monotonic() - t0
        return result

    # --- try candidates / rotations ---------------------------------------- #
    notes = []
    debug = None
    for ci, cand in enumerate(candidates):
        for k in range(4):
            if time.monotonic() > t_end:
                result["message"] = "处理超时（time budget exceeded）"
                result["seconds"] = time.monotonic() - t0
                return result

            view = sutils.analyse_grid(work_pad, cand.quad, k)
            cells = sutils.extract_cells(view)
            board, digit_idx, scores = sutils.read_board(digit_model, cells)
            if ci == 0 and k == 0:
                debug = sutils.draw_debug_image(view, cells, board)

            if len(digit_idx) < sutils.MIN_GIVENS:
                notes.append(f"候选{ci}/旋转{90 * k}°：仅识别到 {len(digit_idx)} 个数字")
                continue

            solved = sutils.solve_with_corrections(board, digit_idx, scores, t_end)
            if solved is None:
                notes.append(f"候选{ci}/旋转{90 * k}°：识别到 {len(digit_idx)} 个数字但无解")
                continue

            M_full = view.M @ T @ S          # original image -> warped grid
            annotated = _render_digits(img_rgb, cells, solved, M_full)

            result.update({
                "ok": True,
                "message": (f"已求解：网格得分 {cand.score:.2f}，"
                            f"旋转 {90 * k}°，原始数字 {len(digit_idx)} 个，"
                            f"新增 {81 - len(digit_idx)} 个"),
                "annotated": annotated,
                "givens": len(digit_idx),
                "board": board.tolist(),
                "solved_board": np.asarray(solved).astype(int).tolist(),
                "grid_score": float(cand.score),
                "rotation": 90 * k,
                "debug": debug,
            })
            result["seconds"] = time.monotonic() - t0
            return result

    result["message"] = "找到了网格，但无法求解：" + "；".join(notes[:4])
    result["debug"] = debug
    result["seconds"] = time.monotonic() - t0
    return result


def encode_png_data_url(img_rgb):
    """Encode an RGB numpy image to a data URL (PNG, lossless colours)."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encode failed")
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
