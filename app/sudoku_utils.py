# Robust utilities for locating, reading and solving sudoku puzzles in images.
#
# Pipeline overview
#   1. Pad the image with a white frame (so grids touching the image border
#      still have a closed outline).
#   2. Generate many grid-outline candidates (several threshold strategies,
#      hull / minAreaRect fallbacks, whole-image fallback).
#   3. Warp every candidate to a square, measure how much it "looks like" a
#      9x9 grid (line-profile score), tighten the quad, and rank candidates.
#   4. Slice the warped grid into 81 cells using the detected grid lines
#      (no dependency on per-cell contours any more).
#   5. Classify digits with the ONNX model, then solve with a fast bit-mask
#      solver that has a time/node limit. If the board is inconsistent, retry
#      with the least-confident digits swapped for their 2nd-best class, and
#      with the grid rotated by 90/180/270 degrees.

import math
import os
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CELL_SIZE = 80                       # px per cell in the warped grid
GRID_SIZE = 9 * CELL_SIZE            # 720
WARP_PAD = 10                        # margin around the grid in the warped image
SIDE = GRID_SIZE + 2 * WARP_PAD      # side of the warped (square) image
IMG_PAD = 24                         # white frame added around the input image

MIN_GRID_FRACTION = 0.04             # min area of a candidate quad (fraction of image)
MIN_GRID_SCORE = 0.20                # min "looks like a sudoku" score
MIN_GIVENS = 17                      # a valid sudoku has at least 17 clues
CELL_BORDER_FRACTION = 0.08          # border blanked inside every cell

_POPCOUNT = [bin(i).count("1") for i in range(1024)]


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class SudokuSolver:
    """Bit-mask backtracking solver with MRV heuristic and a time/node limit.

    solve() returns True and replaces self.board with the solution on success;
    on failure (invalid board, no solution, or limit hit) self.board is left
    untouched and False is returned.
    """

    def __init__(self, board, max_nodes=400_000, timeout=3.0):
        self.board = board
        self.max_nodes = max_nodes
        self.timeout = timeout
        self.nodes = 0
        self.timed_out = False

    def set_new_board(self, new_board):
        self.board = new_board

    def print_board(self):
        for r in range(9):
            if r and r % 3 == 0:
                print("- " * 11)
            row = ""
            for c in range(9):
                if c and c % 3 == 0:
                    row += "| "
                row += f"{self.board[r][c]} "
            print(row)

    @staticmethod
    def is_board_valid(board):
        """True if the board has no duplicate digit in any row/column/box."""
        b = np.asarray(board)
        if b.shape != (9, 9) or b.min() < 0 or b.max() > 9:
            return False
        for i in range(9):
            br, bc = 3 * (i // 3), 3 * (i % 3)
            units = (b[i, :], b[:, i], b[br:br + 3, bc:bc + 3].ravel())
            for u in units:
                u = u[u != 0]
                if len(np.unique(u)) != len(u):
                    return False
        return True

    def solve(self):
        board = np.array(self.board, dtype=int)
        if not self.is_board_valid(board):
            return False

        grid = board.tolist()
        rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
        empties = []
        for r in range(9):
            for c in range(9):
                v = grid[r][c]
                if v:
                    bit = 1 << v
                    rows[r] |= bit
                    cols[c] |= bit
                    boxes[3 * (r // 3) + c // 3] |= bit
                else:
                    empties.append((r, c))

        self.nodes = 0
        self.timed_out = False
        deadline = time.monotonic() + self.timeout
        FULL = 0x3FE  # bits 1..9
        pc = _POPCOUNT

        def search():
            self.nodes += 1
            if self.nodes > self.max_nodes or (
                (self.nodes & 255) == 0 and time.monotonic() > deadline
            ):
                self.timed_out = True
                return False

            # Pick the empty cell with the fewest candidates (MRV)
            best, best_mask, best_cnt = None, 0, 10
            for (r, c) in empties:
                if grid[r][c]:
                    continue
                mask = FULL & ~(rows[r] | cols[c] | boxes[3 * (r // 3) + c // 3])
                cnt = pc[mask]
                if cnt == 0:
                    return False  # dead end
                if cnt < best_cnt:
                    best, best_mask, best_cnt = (r, c), mask, cnt
                    if cnt == 1:
                        break
            if best is None:
                return True  # no empty cell left

            r, c = best
            b = 3 * (r // 3) + c // 3
            m = best_mask
            while m:
                bit = m & -m
                m ^= bit
                grid[r][c] = bit.bit_length() - 1
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit
                if search():
                    return True
                rows[r] ^= bit
                cols[c] ^= bit
                boxes[b] ^= bit
                grid[r][c] = 0
                if self.timed_out:
                    return False
            return False

        if search():
            self.board = np.array(grid, dtype=int)
            return True
        return False


# --------------------------------------------------------------------------- #
# ONNX model wrapper
# --------------------------------------------------------------------------- #
class OnnxModel:
    """Minimal onnxruntime wrapper that mimics Keras model.predict.

    Input : (N, 28, 28, 1) float array (same format as before, 0..255 values)
    Output: (N, 9) class scores for digits 1..9
    """

    def __init__(self, model_fpath, providers=None):
        if not os.path.exists(model_fpath):
            raise FileNotFoundError(f"Model file not found: '{model_fpath}'")
        if providers is None:
            wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            available = ort.get_available_providers()
            providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_fpath, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        return self.session.run(None, {self.input_name: x})[0]


# --------------------------------------------------------------------------- #
# Basic image helpers
# --------------------------------------------------------------------------- #
def resize_and_maintain_aspect_ratio(input_image, new_width):
    """Resize to `new_width` keeping the aspect ratio."""
    h, w = input_image.shape[:2]
    ratio = new_width / float(w)
    new_height = max(1, int(round(h * ratio)))
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
    return cv2.resize(input_image, (new_width, new_height), interpolation=interp)


def order_corners(pts):
    """Order 4 points clockwise, starting from the one closest to top-left.

    Uses the angle around the centroid, so it also works for rotated quads.
    Result order: TL, TR, BR, BL for an upright quad.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]            # clockwise in image coordinates
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0).astype(np.float32)


def _find_contours(binary, mode):
    res = cv2.findContours(binary, mode, cv2.CHAIN_APPROX_SIMPLE)
    return list(res[0] if len(res) == 2 else res[1])


# --------------------------------------------------------------------------- #
# Warping / grid analysis
# --------------------------------------------------------------------------- #
@dataclass
class GridView:
    M: np.ndarray            # source image -> warped image
    warped: np.ndarray       # RGB, SIDE x SIDE
    binary: np.ndarray       # white-on-black thresholded warped image
    h_lines: np.ndarray      # horizontal line mask
    v_lines: np.ndarray      # vertical line mask
    hp: np.ndarray           # horizontal line coverage per row
    vp: np.ndarray           # vertical line coverage per column


def warp_grid(img, quad, k=0):
    """Perspective-warp `quad` to a SIDE x SIDE square.

    k rotates the corner assignment by k*90 degrees (used to try other
    orientations of the puzzle).
    """
    quad = np.roll(order_corners(quad), -k, axis=0).astype(np.float32)
    dst = np.float32([
        [WARP_PAD, WARP_PAD],
        [WARP_PAD + GRID_SIZE, WARP_PAD],
        [WARP_PAD + GRID_SIZE, WARP_PAD + GRID_SIZE],
        [WARP_PAD, WARP_PAD + GRID_SIZE],
    ])
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(
        img, M, (SIDE, SIDE), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return M, warped


def binarize_warped(warped_rgb):
    """Illumination-normalised adaptive threshold. Ink -> 255, paper -> 0."""
    gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))
    norm = cv2.divide(gray, bg, scale=255)
    return cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 91, 7)


def _line_masks_and_profiles(binary):
    k = int(CELL_SIZE * 0.9)
    h_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    v_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    side = binary.shape[0]
    hp = h_lines.sum(axis=1) / (255.0 * side)
    vp = v_lines.sum(axis=0) / (255.0 * side)
    return h_lines, v_lines, hp, vp


def analyse_grid(img, quad, k=0):
    M, warped = warp_grid(img, quad, k)
    binary = binarize_warped(warped)
    h_lines, v_lines, hp, vp = _line_masks_and_profiles(binary)
    return GridView(M, warped, binary, h_lines, v_lines, hp, vp)


def _grid_score(hp, vp):
    """How well do the line profiles match 10 + 10 evenly spaced lines? (0..1)"""
    s = []
    for prof in (hp, vp):
        for i in range(10):
            e = WARP_PAD + i * CELL_SIZE
            lo = max(e - CELL_SIZE // 4, 0)
            hi = min(e + CELL_SIZE // 4 + 1, len(prof))
            s.append(min(1.0, float(prof[lo:hi].max()) / 0.45))
    return float(np.mean(s))


def _line_centres(profile):
    if profile.max() <= 0.05:
        return []
    idx = np.flatnonzero(profile >= max(0.5 * profile.max(), 0.1))
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
    centres = []
    for g in groups:
        w = profile[g]
        centres.append(float((g * w).sum() / w.sum()))
    merged = []
    for c in centres:
        if merged and c - merged[-1] < 0.3 * CELL_SIZE:
            merged[-1] = (merged[-1] + c) / 2
        else:
            merged.append(c)
    return merged


def _outer_box(hp, vp):
    """Position (warped coords) of the outermost grid lines, or None."""
    cy, cx = _line_centres(hp), _line_centres(vp)
    if len(cy) < 6 or len(cx) < 6:
        return None

    def span_ok(c):
        sp = np.median(np.diff(c))
        return sp > 0 and 7.5 <= (c[-1] - c[0]) / sp <= 10.5

    if not (span_ok(cy) and span_ok(cx)):
        return None
    return cx[0], cy[0], cx[-1], cy[-1]


def refine_quad(img, quad, iters=2):
    """Tighten/loosen a quad so that its edges sit on the outer grid lines.
    Returns (best_quad, best_score)."""
    q = np.asarray(quad, np.float32)
    best_q, best_score = q.copy(), -1.0
    for it in range(iters + 1):
        view = analyse_grid(img, q)
        score = _grid_score(view.hp, view.vp)
        if score > best_score:
            best_q, best_score = q.copy(), score
        if it == iters:
            break
        box = _outer_box(view.hp, view.vp)
        if box is None:
            break
        x0, y0, x1, y1 = box
        lo, hi = WARP_PAD, WARP_PAD + GRID_SIZE
        if max(abs(x0 - lo), abs(y0 - lo), abs(x1 - hi), abs(y1 - hi)) <= 3:
            break
        dst = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        q = cv2.perspectiveTransform(dst[None], np.linalg.inv(view.M))[0].astype(np.float32)
    return best_q, best_score


def _refine_positions(prof):
    """Positions of the 10 grid lines along one axis (snapped to detected
    lines when possible, otherwise the nominal position)."""
    nominal = np.array([WARP_PAD + i * CELL_SIZE for i in range(10)], dtype=float)
    pos = []
    for i in range(10):
        e = int(nominal[i])
        lo = max(e - CELL_SIZE // 4, 0)
        hi = min(e + CELL_SIZE // 4 + 1, len(prof))
        seg = prof[lo:hi]
        j = int(np.argmax(seg))
        if seg[j] < 0.2:
            pos.append(float(e))
            continue
        c = lo + j
        a, b = max(c - 4, 0), min(c + 5, len(prof))
        w = prof[a:b]
        pos.append(float((np.arange(a, b) * w).sum() / w.sum()))
    pos = np.array(pos)
    d = np.diff(pos)
    if np.any(d < 0.6 * CELL_SIZE) or np.any(d > 1.4 * CELL_SIZE):
        return nominal
    return pos


# --------------------------------------------------------------------------- #
# Grid candidate search
# --------------------------------------------------------------------------- #
@dataclass
class GridCandidate:
    quad: np.ndarray
    score: float


def _threshold_variants(gray):
    blur3 = cv2.GaussianBlur(gray, (3, 3), 0)
    blur5 = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(blur3)

    out = [
        cv2.adaptiveThreshold(blur3, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY_INV, 41, 8),       # original setting
        cv2.adaptiveThreshold(blur5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 31, 6),
        cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 51, 10),
        cv2.threshold(blur5, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        cv2.dilate(cv2.Canny(blur5, 30, 100), np.ones((3, 3), np.uint8)),
    ]
    k5 = np.ones((5, 5), np.uint8)
    # Closing bridges small gaps in broken grid lines
    return [cv2.morphologyEx(v, cv2.MORPH_CLOSE, k5) for v in out]


def _quad_from_contour(contour):
    hull = cv2.convexHull(contour)
    if len(hull) < 4:
        return None
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.045):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    # Fallback: minimum-area rectangle if the hull is nearly rectangular
    hull_area = cv2.contourArea(hull)
    rect = cv2.minAreaRect(hull)
    rw, rh = rect[1]
    if rw * rh > 0 and hull_area / (rw * rh) > 0.85:
        return cv2.boxPoints(rect).astype(np.float32)
    return None


def _quad_ok(q, img_area):
    pts = q.reshape(-1, 1, 2).astype(np.float32)
    if cv2.contourArea(pts) < MIN_GRID_FRACTION * img_area:
        return False
    if not cv2.isContourConvex(pts):
        return False
    d = [float(np.linalg.norm(q[i] - q[(i + 1) % 4])) for i in range(4)]
    if min(d) < 10:
        return False
    w, h = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
    return 0.5 < w / h < 2.0


def _same_quad(a, b, tol):
    return float(np.max(np.linalg.norm(a - b, axis=1))) < tol


def find_grid_candidates(img_pad, max_candidates=4, max_eval=12):
    """Return (candidates sorted by score, best_score_seen)."""
    h, w = img_pad.shape[:2]
    img_area = float(h * w)
    diag = math.hypot(h, w)
    gray = cv2.cvtColor(img_pad, cv2.COLOR_RGB2GRAY)

    raw = []
    for bin_img in _threshold_variants(gray):
        contours = _find_contours(bin_img, cv2.RETR_LIST)
        areas = [(cv2.contourArea(c), c) for c in contours if len(c) >= 4]
        areas.sort(key=lambda t: t[0], reverse=True)
        for area, c in areas[:10]:
            if area < MIN_GRID_FRACTION * img_area:
                break
            quad = _quad_from_contour(c)
            if quad is None:
                continue
            quad = order_corners(quad)
            if _quad_ok(quad, img_area):
                raw.append(quad)

    # Whole-image fallbacks: the picture may be a tight crop of the grid
    p = IMG_PAD
    W0, H0 = w - 2 * p, h - 2 * p
    for inset in (0.0, 0.02):
        dx, dy = inset * W0, inset * H0
        raw.append(np.float32([
            [p + dx, p + dy], [p + W0 - 1 - dx, p + dy],
            [p + W0 - 1 - dx, p + H0 - 1 - dy], [p + dx, p + H0 - 1 - dy]]))

    uniq = []
    for q in raw:
        if all(not _same_quad(q, u, 0.02 * diag) for u in uniq):
            uniq.append(q)
    uniq.sort(key=lambda q: cv2.contourArea(q.reshape(-1, 1, 2)), reverse=True)

    scored = [(sc, rq) for rq, sc in (refine_quad(img_pad, q) for q in uniq[:max_eval])]
    scored.sort(key=lambda t: -t[0])
    best_seen = scored[0][0] if scored else 0.0

    out = []
    for sc, q in scored:
        if sc < MIN_GRID_SCORE:
            break
        if any(_same_quad(q, c.quad, 0.02 * diag) for c in out):
            continue
        out.append(GridCandidate(q, sc))
        if len(out) >= max_candidates:
            break
    return out, best_seen


# --------------------------------------------------------------------------- #
# Cell extraction and digit recognition
# --------------------------------------------------------------------------- #
def _process_cell(crop):
    """Return (contains_digit, 28x28 uint8 image: white digit on black)."""
    blank = np.zeros((28, 28), np.uint8)
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return False, blank

    m = max(2, int(round(CELL_BORDER_FRACTION * min(h, w))))
    cell = crop.copy()
    cell[:m, :] = 0
    cell[h - m:, :] = 0
    cell[:, :m] = 0
    cell[:, w - m:] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cell, connectivity=8)
    keep = np.zeros_like(cell)
    best_area, best_h = 0, 0
    for lab in range(1, n):
        x, y, cw, ch, area = stats[lab]
        if area < 0.004 * h * w:                       # noise
            continue
        touches = x <= m or y <= m or x + cw >= w - m or y + ch >= h - m
        fill = area / float(cw * ch)
        aspect = max(cw / float(ch), ch / float(cw))
        if touches and (fill < 0.2 or aspect > 4.0):   # leftover grid-line pieces
            continue
        keep[labels == lab] = 255
        if area > best_area:
            best_area, best_h = area, ch

    present = best_area >= 0.015 * h * w and best_h >= 0.3 * h
    if not present:
        return False, blank

    keep = cv2.erode(keep, np.ones((3, 3), np.uint8), iterations=1)
    return True, cv2.resize(keep, (28, 28), interpolation=cv2.INTER_AREA)


def extract_cells(view):
    """Slice the warped grid into 81 cells (row-major order)."""
    xs = _refine_positions(view.vp)
    ys = _refine_positions(view.hp)

    # Erase the grid lines so they cannot be mistaken for digits
    lines = cv2.dilate(cv2.bitwise_or(view.h_lines, view.v_lines),
                       np.ones((3, 3), np.uint8))
    clean = view.binary.copy()
    clean[lines > 0] = 0

    cells = []
    for i in range(9):
        for j in range(9):
            y0, y1 = int(round(ys[i])), int(round(ys[i + 1]))
            x0, x1 = int(round(xs[j])), int(round(xs[j + 1]))
            present, img28 = _process_cell(clean[y0:y1, x0:x1])
            cells.append({
                "img": img28,
                "contains_digit": present,
                "x_centroid": (xs[j] + xs[j + 1]) / 2.0,
                "y_centroid": (ys[i] + ys[i + 1]) / 2.0,
            })
    return cells


def read_board(model, cells):
    """Classify digit cells. Returns (board 9x9, digit_cell_indices, scores)."""
    digit_idx = [i for i, c in enumerate(cells) if c["contains_digit"]]
    board = np.zeros(81, dtype=int)
    scores = None
    if digit_idx:
        x = np.array([cells[i]["img"][..., None] for i in digit_idx], dtype=np.float32)
        scores = np.asarray(model.predict(x))
        board[digit_idx] = np.argmax(scores, axis=1) + 1
    return board.reshape(9, 9), digit_idx, scores


# Kept for backwards compatibility with the previous API
def get_predicted_sudoku_grid(model, cells):
    return read_board(model, cells)[0]


def solve_with_corrections(board, digit_idx, scores, t_end, max_weak=12, pair_pool=6):
    """Solve `board`. If it is inconsistent / unsolvable, retry after replacing
    the least-confident digits by their 2nd-best class (1 or 2 at a time)."""

    def attempt(b, per_try):
        if not SudokuSolver.is_board_valid(b):
            return None
        remaining = t_end - time.monotonic()
        if remaining <= 0:
            return None
        s = SudokuSolver(b.copy(), timeout=min(per_try, remaining))
        return s.board if s.solve() else None

    res = attempt(board, 4.0)
    if res is not None or scores is None or len(digit_idx) == 0:
        return res

    scores = np.asarray(scores)
    if scores.shape[1] < 2:
        return None
    order = np.argsort(scores, axis=1)
    rows = np.arange(len(scores))
    best, second = order[:, -1], order[:, -2]
    margin = scores[rows, best] - scores[rows, second]
    weak = [int(i) for i in np.argsort(margin)[:max_weak]]
    flat = board.reshape(-1)

    def with_changes(changes):
        b = flat.copy()
        for w in changes:
            b[digit_idx[w]] = second[w] + 1
        return b.reshape(9, 9)

    for w in weak:
        res = attempt(with_changes([w]), 1.5)
        if res is not None:
            return res
    for a, b in combinations(weak[:pair_pool], 2):
        res = attempt(with_changes([a, b]), 1.5)
        if res is not None:
            return res
    return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def generate_solution_image(full_image, cells, solved_board, M, color=(255, 15, 0)):
    """Draw the solved digits (only in originally empty cells) onto the image.

    M must map full_image coordinates -> warped grid coordinates.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_h = cv2.getTextSize("8", font, 1.0, 1)[0][1]
    scale = CELL_SIZE * 0.55 / base_h
    thickness = max(2, int(round(CELL_SIZE / 28)))

    mask = np.zeros((SIDE, SIDE), dtype=np.uint8)
    flat = np.asarray(solved_board).reshape(-1)
    for i, cell in enumerate(cells):
        if cell["contains_digit"]:
            continue
        text = str(int(flat[i]))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        org = (int(round(cell["x_centroid"] - tw / 2.0)),
               int(round(cell["y_centroid"] + th / 2.0)))
        cv2.putText(mask, text, org, font, scale, 255, thickness, cv2.LINE_AA)

    h, w = full_image.shape[:2]
    unwarped = cv2.warpPerspective(mask, M, (w, h),
                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = (unwarped.astype(np.float32) / 255.0)[..., None]
    out = full_image.astype(np.float32) * (1 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_debug_image(view, cells, board):
    """Warped grid with detected lines and the digits that were read."""
    img = view.warped.copy()
    for x in _refine_positions(view.vp):
        cv2.line(img, (int(x), 0), (int(x), SIDE - 1), (0, 200, 0), 1)
    for y in _refine_positions(view.hp):
        cv2.line(img, (0, int(y)), (SIDE - 1, int(y)), (0, 200, 0), 1)
    for idx, c in enumerate(cells):
        if c["contains_digit"]:
            r, cc = divmod(idx, 9)
            cv2.putText(img, str(int(board[r][cc])),
                        (int(c["x_centroid"]) - 12, int(c["y_centroid"]) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# High-level entry point
# --------------------------------------------------------------------------- #
@dataclass
class SolveResult:
    annotated: Optional[np.ndarray] = None
    message: str = ""
    debug_image: Optional[np.ndarray] = None

    @property
    def ok(self):
        return self.annotated is not None


def solve_sudoku_image(model, img_rgb, time_budget=20.0):
    """Detect, read and solve the sudoku in an RGB image.

    Never raises for "normal" failures; inspect the returned SolveResult.
    """
    t_end = time.monotonic() + time_budget
    p = IMG_PAD
    img_pad = cv2.copyMakeBorder(img_rgb, p, p, p, p, cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255))

    candidates, best_seen = find_grid_candidates(img_pad)
    if not candidates:
        return SolveResult(message=f"no sudoku grid found (best grid score {best_seen:.2f})")

    # original image -> padded image
    T = np.array([[1, 0, p], [0, 1, p], [0, 0, 1]], dtype=np.float64)

    debug = None
    notes = []
    for ci, cand in enumerate(candidates):
        for k in range(4):                      # 0/90/180/270 degree orientations
            if time.monotonic() > t_end:
                return SolveResult(message="time budget exceeded", debug_image=debug)

            view = analyse_grid(img_pad, cand.quad, k)
            cells = extract_cells(view)
            board, digit_idx, scores = read_board(model, cells)
            if ci == 0 and k == 0:
                debug = draw_debug_image(view, cells, board)

            if len(digit_idx) < MIN_GIVENS:
                notes.append(f"cand{ci}/rot{90 * k}: only {len(digit_idx)} digits")
                continue

            solved = solve_with_corrections(board, digit_idx, scores, t_end)
            if solved is None:
                notes.append(f"cand{ci}/rot{90 * k}: {len(digit_idx)} digits, unsolvable")
                continue

            M_orig = view.M @ T
            annotated = generate_solution_image(img_rgb, cells, solved, M_orig)
            return SolveResult(
                annotated=annotated,
                message=f"solved (grid score {cand.score:.2f}, rotation {90 * k} deg, "
                        f"{len(digit_idx)} givens)")

    return SolveResult(
        message="grid found but puzzle could not be solved; " + "; ".join(notes[:4]),
        debug_image=debug)