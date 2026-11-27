/* Perspective warping + grid analysis (port of app/sudoku_utils.py). */
import { applyH, homography, invert3, orderCorners } from "./math.js";
import {
  adaptiveInk, closeRect, gaussian3, maxFilter, minFilter, toGray,
} from "./image.js";

export const CELL_SIZE = 80;
export const GRID_SIZE = 9 * CELL_SIZE;          // 720
export const WARP_PAD = 10;
export const SIDE = GRID_SIZE + 2 * WARP_PAD;    // 740
export const IMG_PAD = 24;
export const MIN_GRID_SCORE = 0.2;
export const MIN_GIVENS = 17;
export const CELL_BORDER_FRACTION = 0.08;

const DST = [
  [WARP_PAD, WARP_PAD],
  [WARP_PAD + GRID_SIZE, WARP_PAD],
  [WARP_PAD + GRID_SIZE, WARP_PAD + GRID_SIZE],
  [WARP_PAD, WARP_PAD + GRID_SIZE],
];

/** Bilinear-sample an RGBA image; outside -> white. */
function sampleBilinear(src, w, h, x, y, out, o) {
  if (x < 0 || y < 0 || x > w - 1 || y > h - 1) {
    out[o] = 255; out[o + 1] = 255; out[o + 2] = 255; out[o + 3] = 255;
    return;
  }
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;
  for (let c = 0; c < 3; c++) {
    const p00 = src[(y0 * w + x0) * 4 + c];
    const p10 = src[(y0 * w + x1) * 4 + c];
    const p01 = src[(y1 * w + x0) * 4 + c];
    const p11 = src[(y1 * w + x1) * 4 + c];
    const top = p00 + (p10 - p00) * fx;
    const bot = p01 + (p11 - p01) * fx;
    out[o + c] = (top + (bot - top) * fy) | 0;
  }
  out[o + 3] = 255;
}

/**
 * Warp `quad` (in source coords) to a SIDE x SIDE square.
 * k rotates the corner assignment by k*90 degrees.
 */
export function warpGrid(src, w, h, quad, k = 0) {
  const q = orderCorners(quad);
  const rolled = [];
  for (let i = 0; i < 4; i++) rolled.push(q[(i + k) % 4]);
  const M = homography(rolled, DST);
  if (!M) return null;
  const inv = invert3(M);
  if (!inv) return null;

  const warped = new Uint8ClampedArray(SIDE * SIDE * 4);
  for (let y = 0; y < SIDE; y++) {
    for (let x = 0; x < SIDE; x++) {
      const [sx, sy] = applyH(inv, x, y);
      sampleBilinear(src, w, h, sx, sy, warped, (y * SIDE + x) * 4);
    }
  }
  return { M, inv, warped, side: SIDE, k };
}

/** Illumination-normalised binarisation: ink -> 255, paper -> 0. */
export function binarizeWarped(rgba, side = SIDE) {
  const gray = gaussian3(toGray(rgba, side, side), side, side);
  const bg = closeRect(gray, side, side, 41);
  const norm = new Uint8ClampedArray(side * side);
  for (let i = 0; i < norm.length; i++) {
    const b = bg[i] || 1;
    norm[i] = Math.min(255, (gray[i] * 255) / b);
  }
  return adaptiveInk(norm, side, side, 91, 7);
}

function lineProfiles(binary, side) {
  const r = Math.floor((CELL_SIZE * 0.9) / 2);
  const hLines = maxFilter(minFilter(binary, side, side, r, true), side, side, r, true);
  const vLines = maxFilter(minFilter(binary, side, side, r, false), side, side, r, false);
  const hp = new Float32Array(side);
  const vp = new Float32Array(side);
  const norm = 255 * side;
  for (let y = 0; y < side; y++) {
    let s = 0;
    for (let x = 0; x < side; x++) s += hLines[y * side + x];
    hp[y] = s / norm;
  }
  for (let x = 0; x < side; x++) {
    let s = 0;
    for (let y = 0; y < side; y++) s += vLines[y * side + x];
    vp[x] = s / norm;
  }
  return { hLines, vLines, hp, vp };
}

export function analyseGrid(src, w, h, quad, k = 0) {
  const view = warpGrid(src, w, h, quad, k);
  if (!view) return null;
  const binary = binarizeWarped(view.warped, SIDE);
  return { ...view, binary, ...lineProfiles(binary, SIDE) };
}

/** How well do profiles match 10 + 10 evenly spaced lines? 0..1 */
export function gridScore(hp, vp) {
  let total = 0;
  for (const prof of [hp, vp]) {
    for (let i = 0; i < 10; i++) {
      const e = WARP_PAD + i * CELL_SIZE;
      const lo = Math.max(e - CELL_SIZE / 4, 0);
      const hi = Math.min(e + CELL_SIZE / 4, prof.length - 1);
      let mx = 0;
      for (let j = Math.floor(lo); j <= Math.ceil(hi); j++) mx = Math.max(mx, prof[j] || 0);
      total += Math.min(1, mx / 0.45);
    }
  }
  return total / 20;
}

function lineCentres(profile) {
  let mx = 0;
  for (let i = 0; i < profile.length; i++) mx = Math.max(mx, profile[i]);
  if (mx <= 0.05) return [];
  const thr = Math.max(0.5 * mx, 0.1);
  const centres = [];
  let i = 0;
  while (i < profile.length) {
    if (profile[i] < thr) { i++; continue; }
    let j = i;
    while (j + 1 < profile.length && profile[j + 1] >= thr) j++;
    let sw = 0, sx = 0;
    for (let t = i; t <= j; t++) { sw += profile[t]; sx += t * profile[t]; }
    centres.push(sx / sw);
    i = j + 1;
  }
  const merged = [];
  for (const c of centres) {
    if (merged.length && c - merged[merged.length - 1] < 0.3 * CELL_SIZE) {
      merged[merged.length - 1] = (merged[merged.length - 1] + c) / 2;
    } else merged.push(c);
  }
  return merged;
}

function outerBox(hp, vp) {
  const cy = lineCentres(hp);
  const cx = lineCentres(vp);
  if (cy.length < 6 || cx.length < 6) return null;
  const spanOk = (c) => {
    const gaps = [];
    for (let i = 1; i < c.length; i++) gaps.push(c[i] - c[i - 1]);
    gaps.sort((a, b) => a - b);
    const sp = gaps[gaps.length >> 1];
    return sp > 0 && (c[c.length - 1] - c[0]) / sp >= 7.5 && (c[c.length - 1] - c[0]) / sp <= 10.5;
  };
  if (!spanOk(cy) || !spanOk(cx)) return null;
  return [cx[0], cy[0], cx[cx.length - 1], cy[cy.length - 1]];
}

/** Tighten/loosen a quad so its edges sit on the outer grid lines. */
export function refineQuad(src, w, h, quad, iters = 2) {
  let q = orderCorners(quad);
  let bestQ = q;
  let bestScore = -1;
  for (let it = 0; it <= iters; it++) {
    const view = analyseGrid(src, w, h, q);
    if (!view) break;
    const score = gridScore(view.hp, view.vp);
    if (score > bestScore) { bestQ = q; bestScore = score; }
    if (it === iters) break;
    const box = outerBox(view.hp, view.vp);
    if (!box) break;
    const [x0, y0, x1, y1] = box;
    const lo = WARP_PAD, hi = WARP_PAD + GRID_SIZE;
    if (Math.max(Math.abs(x0 - lo), Math.abs(y0 - lo), Math.abs(x1 - hi), Math.abs(y1 - hi)) <= 3) break;
    const dst = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    q = dst.map(([x, y]) => {
      const p = applyH(view.inv, x, y);
      return [p[0], p[1]];
    });
  }
  return { quad: bestQ, score: bestScore };
}

/** Positions of the 10 grid lines along one axis. */
export function refinePositions(prof) {
  const nominal = [];
  for (let i = 0; i < 10; i++) nominal.push(WARP_PAD + i * CELL_SIZE);
  const pos = [];
  for (let i = 0; i < 10; i++) {
    const e = Math.round(nominal[i]);
    const lo = Math.max(e - CELL_SIZE / 4, 0);
    const hi = Math.min(e + CELL_SIZE / 4, prof.length - 1);
    let j = -1, best = -1;
    for (let t = Math.floor(lo); t <= Math.ceil(hi); t++) {
      if (prof[t] > best) { best = prof[t]; j = t; }
    }
    if (j < 0 || best < 0.2) { pos.push(e); continue; }
    const a = Math.max(j - 4, 0), b = Math.min(j + 5, prof.length);
    let sw = 0, sx = 0;
    for (let t = a; t < b; t++) { sw += prof[t]; sx += t * prof[t]; }
    pos.push(sw > 0 ? sx / sw : e);
  }
  for (let i = 1; i < 10; i++) {
    const d = pos[i] - pos[i - 1];
    if (d < 0.6 * CELL_SIZE || d > 1.4 * CELL_SIZE) return nominal;
  }
  return pos;
}
