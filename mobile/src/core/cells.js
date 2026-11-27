/* Slice the warped grid into 81 cells and build 28x28 digit masks. */
import { erode3, maxFilter, resizeArea } from "./image.js";
import { refinePositions, SIDE } from "./grid.js";

/** 8-connected component labelling of a small binary crop. */
function connectedComponents(mask, w, h) {
  const labels = new Int32Array(w * h).fill(-1);
  const stats = [];
  const stack = new Int32Array(w * h);
  let next = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || labels[i] >= 0) continue;
    const id = next++;
    let sp = 0;
    stack[sp++] = i;
    labels[i] = id;
    let minX = w, minY = h, maxX = -1, maxY = -1, area = 0;
    while (sp > 0) {
      const p = stack[--sp];
      const x = p % w, y = (p / w) | 0;
      area++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          const q = ny * w + nx;
          if (mask[q] && labels[q] < 0) {
            labels[q] = id;
            stack[sp++] = q;
          }
        }
      }
    }
    stats.push({ x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1, area });
  }
  return { labels, stats };
}

const BLANK28 = () => new Float32Array(28 * 28);

/** One cell crop -> (contains_digit, 28x28 white-on-black mask). */
export function processCell(crop, cw, ch) {
  if (ch < 8 || cw < 8) return { present: false, img: BLANK28() };
  const m = Math.max(2, Math.round(0.08 * Math.min(cw, ch)));
  const cell = new Uint8ClampedArray(crop);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      if (x < m || y < m || x >= cw - m || y >= ch - m) cell[y * cw + x] = 0;
    }
  }

  const { labels, stats } = connectedComponents(cell, cw, ch);
  const keep = new Uint8ClampedArray(cw * ch);
  let bestArea = 0, bestH = 0;
  const cellArea = cw * ch;
  for (let id = 0; id < stats.length; id++) {
    const s = stats[id];
    if (s.area < 0.004 * cellArea) continue;
    const touches = s.x <= m || s.y <= m || s.x + s.w >= cw - m || s.y + s.h >= ch - m;
    const fill = s.area / (s.w * s.h);
    const aspect = Math.max(s.w / s.h, s.h / s.w);
    if (touches && (fill < 0.2 || aspect > 4.0)) continue;
    for (let i = 0; i < labels.length; i++) if (labels[i] === id) keep[i] = 255;
    if (s.area > bestArea) { bestArea = s.area; bestH = s.h; }
  }

  const present = bestArea >= 0.015 * cellArea && bestH >= 0.3 * ch;
  if (!present) return { present: false, img: BLANK28() };
  return { present: true, img: resizeArea(erode3(keep, cw, ch), cw, ch, 28, 28) };
}

/** Slice the warped grid into 81 cells in row-major order. */
export function extractCells(view) {
  const xs = refinePositions(view.vp);
  const ys = refinePositions(view.hp);

  // erase grid lines (dilated 3x3) so they are not mistaken for digits
  const lines = maxFilter(
    maxFilter(
      new Uint8ClampedArray(view.binary.length).map((_, i) =>
        view.hLines[i] || view.vLines[i] ? 255 : 0),
      SIDE, SIDE, 1, true),
    SIDE, SIDE, 1, false);
  const clean = new Uint8ClampedArray(view.binary);
  for (let i = 0; i < clean.length; i++) if (lines[i]) clean[i] = 0;

  const cells = [];
  for (let row = 0; row < 9; row++) {
    const y0 = Math.round(ys[row]);
    const y1 = Math.round(ys[row + 1]);
    for (let col = 0; col < 9; col++) {
      const x0 = Math.round(xs[col]);
      const x1 = Math.round(xs[col + 1]);
      const cw = Math.max(1, x1 - x0);
      const chh = Math.max(1, y1 - y0);
      const crop = new Uint8ClampedArray(cw * chh);
      for (let y = 0; y < chh; y++) {
        const srcRow = (y0 + y) * SIDE + x0;
        for (let x = 0; x < cw; x++) crop[y * cw + x] = clean[srcRow + x] || 0;
      }
      const { present, img } = processCell(crop, cw, chh);
      cells.push({
        img,
        present,
        cx: (xs[col] + xs[col + 1]) / 2,
        cy: (ys[row] + ys[row + 1]) / 2,
        row, col,
      });
    }
  }
  return cells;
}
