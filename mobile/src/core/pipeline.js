/* End-to-end pipeline: detect -> warp -> read digits -> solve -> render. */
import { applyH } from "./math.js";
import {
  analyseGrid, CELL_SIZE, IMG_PAD, MIN_GIVENS, MIN_GRID_SCORE, refineQuad,
} from "./grid.js";
import { extractCells } from "./cells.js";
import { readBoard } from "./digit.js";
import { solveWithCorrections } from "./solver.js";

export const WORK_WIDTH = 1000;

export async function createEngine({ detectorUrl, digitUrl, progress }) {
  const step = (m) => progress && progress(m);
  step("加载网格检测模型…");
  const detSession = await ort.InferenceSession.create(detectorUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  step("加载数字识别模型…");
  const digitSession = await ort.InferenceSession.create(digitUrl, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  const { GridDetector } = await import("./detect.js");
  const { DigitClassifier } = await import("./digit.js");
  return {
    detector: new GridDetector(detSession),
    digits: new DigitClassifier(digitSession),
  };
}

/** Draw `source` scaled to at most WORK_WIDTH and surround it with white padding. */
function prepareWorkCanvas(source) {
  const sw = source.width, sh = source.height;
  const scale = Math.min(1, WORK_WIDTH / sw);
  const w = Math.round(sw * scale);
  const h = Math.round(sh * scale);
  const work = document.createElement("canvas");
  work.width = w; work.height = h;
  const wctx = work.getContext("2d", { willReadFrequently: true });
  wctx.imageSmoothingQuality = "high";
  wctx.drawImage(source, 0, 0, w, h);

  const padded = document.createElement("canvas");
  padded.width = w + 2 * IMG_PAD;
  padded.height = h + 2 * IMG_PAD;
  const pctx = padded.getContext("2d", { willReadFrequently: true });
  pctx.fillStyle = "#fff";
  pctx.fillRect(0, 0, padded.width, padded.height);
  pctx.drawImage(work, IMG_PAD, IMG_PAD);

  return {
    canvas: work,
    padded,
    imageData: pctx.getImageData(0, 0, padded.width, padded.height).data,
    scale,
    width: w,
    height: h,
    padWidth: padded.width,
    padHeight: padded.height,
  };
}

function boxToQuad(box, pad) {
  const [x1, y1, x2, y2] = box;
  return [
    [x1 + pad, y1 + pad],
    [x2 + pad, y1 + pad],
    [x2 + pad, y2 + pad],
    [x1 + pad, y2 + pad],
  ];
}

function sameQuad(a, b, tol) {
  for (let i = 0; i < 4; i++) {
    if (Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]) > tol) return false;
  }
  return true;
}

function drawSolution(ctx, cells, solved, view, scale, pad) {
  const font = '600 %SPX "Segoe UI", system-ui, -apple-system, sans-serif';
  const flat = solved.flat();
  ctx.save();
  ctx.fillStyle = "#e61414";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < cells.length; i++) {
    if (cells[i].present) continue;
    const [px, py] = applyH(view.inv, cells[i].cx, cells[i].cy);
    const [qx, qy] = applyH(view.inv, cells[i].cx + CELL_SIZE, cells[i].cy);
    const ox = (px - pad) / scale;
    const oy = (py - pad) / scale;
    const cellPx = Math.hypot(qx - px, qy - py) / scale;
    const fontPx = Math.max(10, cellPx * 0.62);
    ctx.font = font.replace("%SPX", fontPx.toFixed(1) + "px");
    ctx.fillText(String(flat[i]), ox, oy + fontPx * 0.02);
  }
  ctx.restore();
}

/**
 * Solve a sudoku image.
 * @param {HTMLCanvasElement|ImageBitmap} source  full-resolution source image
 * @param {object} engine  from createEngine()
 * @param {object} opts  {timeBudget, onProgress}
 */
export async function solveImage(source, engine, opts = {}) {
  const t0 = performance.now();
  const budget = opts.timeBudget || 25000;
  const deadline = t0 + budget;
  const tick = (m) => opts.onProgress && opts.onProgress(m);

  tick("准备图像…");
  const prep = prepareWorkCanvas(source);
  const W = prep.padWidth, H = prep.padHeight;
  const img = prep.imageData;

  tick("检测数独网格…");
  let dets = [];
  try {
    dets = await engine.detector.detect(prep.canvas);
  } catch (err) {
    dets = [];
  }

  const raw = dets.slice(0, 4).map((d) => boxToQuad(d.box, IMG_PAD));
  // Fallback: the picture may already be a tight crop of the grid.
  raw.push([
    [IMG_PAD, IMG_PAD],
    [IMG_PAD + prep.width - 1, IMG_PAD],
    [IMG_PAD + prep.width - 1, IMG_PAD + prep.height - 1],
    [IMG_PAD, IMG_PAD + prep.height - 1],
  ]);

  tick("校正网格…");
  const tol = 0.02 * Math.hypot(W, H);
  const candidates = [];
  for (const quad of raw) {
    const { quad: q, score } = refineQuad(img, W, H, quad, dets.length ? 1 : 2);
    if (score < MIN_GRID_SCORE) continue;
    if (candidates.some((c) => sameQuad(c.quad, q, tol))) continue;
    candidates.push({ quad: q, score });
    if (candidates.length >= 3) break;
  }
  candidates.sort((a, b) => b.score - a.score);

  if (!candidates.length) {
    return {
      ok: false,
      message: "未检测到数独网格，请把网格拍完整、避免强反光或倾斜过大。",
      boxes: dets.map((d) => d.box),
      seconds: (performance.now() - t0) / 1000,
    };
  }

  const notes = [];
  const attempts = [];
  let debug = null;
  for (let ci = 0; ci < candidates.length; ci++) {
    const cand = candidates[ci];
    for (let k = 0; k < 4; k++) {
      if (performance.now() > deadline) break;
      tick(`识别数字（候选 ${ci + 1}，旋转 ${k * 90}°）…`);
      const view = analyseGrid(img, W, H, cand.quad, k);
      if (!view) continue;
      const cells = extractCells(view);
      const present = cells.filter((c) => c.present);
      const scores = await engine.digits.predict(present);
      const { board, digitIdx, scores: rows } = readBoard(cells, scores);
      if (ci === 0 && k === 0) debug = { view, cells, board };
      attempts.push({ ci, k, givens: digitIdx.length, board });

      if (digitIdx.length < MIN_GIVENS) {
        notes.push(`候选${ci + 1}/旋转${k * 90}°：仅识别到 ${digitIdx.length} 个数字`);
        continue;
      }

      tick(`求解（${digitIdx.length} 个已知数字）…`);
      const solved = solveWithCorrections(board, digitIdx, rows, deadline);
      if (!solved) {
        notes.push(`候选${ci + 1}/旋转${k * 90}°：${digitIdx.length} 个数字但无解`);
        continue;
      }

      const out = document.createElement("canvas");
      out.width = source.width;
      out.height = source.height;
      const ctx = out.getContext("2d");
      ctx.drawImage(source, 0, 0);
      drawSolution(ctx, cells, solved, view, prep.scale, IMG_PAD);

      return {
        ok: true,
        message: `已求解：网格得分 ${cand.score.toFixed(2)}，旋转 ${k * 90}°，` +
                 `已知 ${digitIdx.length} 个数字，填入 ${81 - digitIdx.length} 个`,
        canvas: out,
        board,
        solved,
        givens: digitIdx.length,
        filled: 81 - digitIdx.length,
        score: cand.score,
        rotation: k * 90,
        boxes: dets.map((d) => d.box),
        seconds: (performance.now() - t0) / 1000,
      };
    }
  }

  return {
    ok: false,
    message: "找到了网格，但无法求解：" + notes.slice(0, 3).join("；"),
    boxes: dets.map((d) => d.box),
    attempts,
    debug,
    seconds: (performance.now() - t0) / 1000,
  };
}
