/* Low-level grayscale image ops (typed arrays, no Canvas dependency). */

/** RGBA (Uint8ClampedArray) -> grayscale Uint8ClampedArray (Rec.601 like cv2). */
export function toGray(rgba, w, h) {
  const out = new Uint8ClampedArray(w * h);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
  }
  return out;
}

/** 3x3 gaussian with kernel [1,2,1] (separable, edges replicated). */
export function gaussian3(src, w, h) {
  const tmp = new Uint8ClampedArray(w * h);
  const out = new Uint8ClampedArray(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const l = x > 0 ? src[row + x - 1] : src[row];
      const c = src[row + x];
      const r = x < w - 1 ? src[row + x + 1] : src[row + x];
      tmp[row + x] = (l + 2 * c + r) >> 2;
    }
  }
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const up = y > 0 ? row - w : row;
    const dn = y < h - 1 ? row + w : row;
    for (let x = 0; x < w; x++) {
      out[row + x] = (tmp[up + x] + 2 * tmp[row + x] + tmp[dn + x]) >> 2;
    }
  }
  return out;
}

/** Sliding-window max (horizontal if horizontal=true) with clamped borders. */
export function maxFilter(src, w, h, radius, horizontal = true) {
  const out = new Uint8ClampedArray(w * h);
  const len = horizontal ? w : h;
  const lines = horizontal ? h : w;
  const k = radius * 2 + 1;
  const dq = new Int32Array(len);
  for (let line = 0; line < lines; line++) {
    let head = 0, tail = 0;
    const idx = (i) => (horizontal ? line * w + i : i * w + line);
    for (let i = 0; i < len + radius; i++) {
      if (i < len) {
        const v = src[idx(i)];
        while (tail > head && src[idx(dq[tail - 1])] <= v) tail--;
        dq[tail++] = i;
      }
      const outIdx = i - radius;
      if (outIdx >= 0) {
        while (dq[head] < outIdx - radius) head++;
        out[idx(outIdx)] = src[idx(dq[head])];
      }
    }
  }
  return out;
}

export function minFilter(src, w, h, radius, horizontal = true) {
  const out = new Uint8ClampedArray(w * h);
  const len = horizontal ? w : h;
  const lines = horizontal ? h : w;
  const dq = new Int32Array(len);
  for (let line = 0; line < lines; line++) {
    let head = 0, tail = 0;
    const idx = (i) => (horizontal ? line * w + i : i * w + line);
    for (let i = 0; i < len + radius; i++) {
      if (i < len) {
        const v = src[idx(i)];
        while (tail > head && src[idx(dq[tail - 1])] >= v) tail--;
        dq[tail++] = i;
      }
      const outIdx = i - radius;
      if (outIdx >= 0) {
        while (dq[head] < outIdx - radius) head++;
        out[idx(outIdx)] = src[idx(dq[head])];
      }
    }
  }
  return out;
}

/** Rect dilate (max) / erode (min) with a square kernel. */
export function dilateRect(src, w, h, size) {
  const r = Math.floor(size / 2);
  return maxFilter(maxFilter(src, w, h, r, true), w, h, r, false);
}
export function erodeRect(src, w, h, size) {
  const r = Math.floor(size / 2);
  return minFilter(minFilter(src, w, h, r, true), w, h, r, false);
}
export function closeRect(src, w, h, size) {
  return erodeRect(dilateRect(src, w, h, size), w, h, size);
}
export function openRect(src, w, h, size) {
  return dilateRect(erodeRect(src, w, h, size), w, h, size);
}

/** Erode with a 3x3 square (used to thin digit strokes). */
export function erode3(src, w, h) {
  return erodeRect(src, w, h, 3);
}

/** Integral image (Float64Array) for O(1) box sums. */
export function integralImage(src, w, h) {
  const ii = new Float64Array((w + 1) * (h + 1));
  const stride = w + 1;
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      rowSum += src[y * w + x];
      ii[(y + 1) * stride + (x + 1)] = ii[y * stride + (x + 1)] + rowSum;
    }
  }
  return ii;
}

/**
 * Adaptive mean threshold on `norm`, ink -> 255.
 * Mirrors cv2.adaptiveThreshold(..., MEAN_C, BINARY_INV, block, C).
 */
export function adaptiveInk(norm, w, h, block = 91, C = 7) {
  const ii = integralImage(norm, w, h);
  const stride = w + 1;
  const r = Math.floor(block / 2);
  const out = new Uint8ClampedArray(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      const sum = ii[y1 * stride + x1] - ii[y0 * stride + x1] -
                  ii[y1 * stride + x0] + ii[y0 * stride + x0];
      const mean = sum / ((x1 - x0) * (y1 - y0));
      out[y * w + x] = norm[y * w + x] < mean - C ? 255 : 0;
    }
  }
  return out;
}

/** Area-average downscale of a Uint8 mask. */
export function resizeArea(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  const sx = sw / dw, sy = sh / dh;
  for (let y = 0; y < dh; y++) {
    const y0 = y * sy, y1 = Math.min(sh, (y + 1) * sy);
    for (let x = 0; x < dw; x++) {
      const x0 = x * sx, x1 = Math.min(sw, (x + 1) * sx);
      let sum = 0, weight = 0;
      const iy0 = Math.floor(y0), iy1 = Math.ceil(y1);
      const ix0 = Math.floor(x0), ix1 = Math.ceil(x1);
      for (let yy = iy0; yy < iy1; yy++) {
        const wy = Math.min(y1, yy + 1) - Math.max(y0, yy);
        if (wy <= 0) continue;
        for (let xx = ix0; xx < ix1; xx++) {
          const wx = Math.min(x1, xx + 1) - Math.max(x0, xx);
          if (wx <= 0) continue;
          const wgt = wx * wy;
          sum += src[yy * sw + xx] * wgt;
          weight += wgt;
        }
      }
      out[y * dw + x] = weight > 0 ? sum / weight : 0;
    }
  }
  return out;
}
