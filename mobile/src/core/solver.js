/* Sudoku validity check, bit-mask solver with MRV, and digit-correction retries. */
import { MIN_GIVENS } from "./grid.js";

const FULL = 0x3fe; // bits 1..9
const popcount = new Uint8Array(1024);
for (let i = 0; i < 1024; i++) popcount[i] = (i & 1) + popcount[i >> 1];

export function isBoardValid(grid) {
  const flat = Array.isArray(grid[0]) ? grid.flat() : grid;
  const rows = new Int32Array(9), cols = new Int32Array(9), boxes = new Int32Array(9);
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const v = flat[r * 9 + c];
      if (!v) continue;
      const bit = 1 << v;
      const b = 3 * ((r / 3) | 0) + ((c / 3) | 0);
      if (rows[r] & bit || cols[c] & bit || boxes[b] & bit) return false;
      rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit;
    }
  }
  return true;
}

/**
 * Solve in place on a copy. Returns a 9x9 flat array or null.
 * MRV ordering + node/time limits, same strategy as the Python version.
 */
export function solve(grid9, { maxNodes = 400000, deadline = Infinity } = {}) {
  if (!isBoardValid(grid9)) return null;
  const grid = Int8Array.from(Array.isArray(grid9[0]) ? grid9.flat() : grid9);
  const rows = new Int32Array(9), cols = new Int32Array(9), boxes = new Int32Array(9);
  const empties = [];
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const v = grid[r * 9 + c];
      if (v) {
        const bit = 1 << v;
        rows[r] |= bit; cols[c] |= bit; boxes[3 * ((r / 3) | 0) + ((c / 3) | 0)] |= bit;
      } else empties.push(r * 9 + c);
    }
  }
  let nodes = 0;
  let timedOut = false;

  const search = () => {
    if (++nodes > maxNodes) { timedOut = true; return false; }
    if ((nodes & 255) === 0 && performance.now() > deadline) { timedOut = true; return false; }

    let bestIdx = -1, bestMask = 0, bestCnt = 10;
    for (let e = 0; e < empties.length; e++) {
      const idx = empties[e];
      if (grid[idx]) continue;
      const r = (idx / 9) | 0, c = idx % 9;
      const mask = FULL & ~(rows[r] | cols[c] | boxes[3 * ((r / 3) | 0) + ((c / 3) | 0)]);
      const cnt = popcount[mask];
      if (cnt === 0) return false;
      if (cnt < bestCnt) { bestIdx = idx; bestMask = mask; bestCnt = cnt; if (cnt === 1) break; }
    }
    if (bestIdx < 0) return true;

    const r = (bestIdx / 9) | 0, c = bestIdx % 9;
    const b = 3 * ((r / 3) | 0) + ((c / 3) | 0);
    let m = bestMask;
    while (m) {
      const bit = m & -m;
      m ^= bit;
      grid[bestIdx] = 31 - Math.clz32(bit);
      rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit;
      if (search()) return true;
      rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit;
      grid[bestIdx] = 0;
      if (timedOut) return false;
    }
    return false;
  };

  if (!search()) return null;
  const out = [];
  for (let r = 0; r < 9; r++) out.push(Array.from(grid.slice(r * 9, r * 9 + 9)));
  return out;
}

/**
 * Try to solve; on failure swap the least-confident digits for their 2nd-best
 * class (1 or 2 at a time). Returns 9x9 array or null.
 */
export function solveWithCorrections(board, digitIdx, scores, deadline = Infinity) {
  const flat = board.flat();
  const attempt = (b, perTryMs) => {
    if (!isBoardValid(b)) return null;
    const dl = Math.min(deadline, performance.now() + perTryMs);
    if (dl <= performance.now()) return null;
    return solve(Array.from({ length: 9 }, (_, r) => b.slice(r * 9, r * 9 + 9)), { deadline: dl });
  };

  const first = attempt(flat, 4000);
  if (first) return first;
  if (!scores || !scores.length || !digitIdx.length) return null;

  const rows = []; // {pos, best, second, margin}
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    let best = 0, second = -1;
    for (let k = 1; k < s.length; k++) {
      if (s[k] > s[best]) { second = best; best = k; }
      else if (second < 0 || s[k] > s[second]) second = k;
    }
    rows.push({ pos: digitIdx[i], second: second + 1, margin: s[best] - s[second] });
  }
  rows.sort((a, b) => a.margin - b.margin);
  const weak = rows.slice(0, 12);

  const withChanges = (changes) => {
    const b = flat.slice();
    for (const w of changes) b[w.pos] = w.second;
    return b;
  };

  for (const w of weak) {
    const solved = attempt(withChanges([w]), 1500);
    if (solved) return solved;
  }
  const pool = weak.slice(0, 6);
  for (let i = 0; i < pool.length; i++) {
    for (let j = i + 1; j < pool.length; j++) {
      const solved = attempt(withChanges([pool[i], pool[j]]), 1500);
      if (solved) return solved;
    }
  }
  return null;
}

export { MIN_GIVENS };
