/* YOLOv8 single-class grid detector (onnxruntime-web, WASM). */

export const INPUT_SIZE = 640;
export const CONF_THRESHOLD = 0.25;
export const IOU_THRESHOLD = 0.45;

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter || 1);
}

function nms(boxes, scores, thr) {
  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const keep = [];
  const removed = new Uint8Array(boxes.length);
  for (const i of order) {
    if (removed[i]) continue;
    keep.push(i);
    for (const j of order) {
      if (removed[j] || j === i) continue;
      if (iou(boxes[i], boxes[j]) > thr) removed[j] = 1;
    }
  }
  return keep;
}

export class GridDetector {
  constructor(session) {
    this.session = session;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
  }

  /** Detect sudoku grids in `canvas`; returns boxes in canvas pixel coords. */
  async detect(canvas, conf = CONF_THRESHOLD, iouThr = IOU_THRESHOLD) {
    const w = canvas.width, h = canvas.height;
    const size = INPUT_SIZE;
    const scale = Math.min(size / w, size / h);
    const nw = Math.round(w * scale), nh = Math.round(h * scale);
    const dx = Math.floor((size - nw) / 2), dy = Math.floor((size - nh) / 2);

    const c = document.createElement("canvas");
    c.width = size; c.height = size;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0, 0, size, size);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(canvas, dx, dy, nw, nh);
    const rgba = ctx.getImageData(0, 0, size, size).data;

    const area = size * size;
    const input = new Float32Array(3 * area);
    for (let p = 0, i = 0; i < area; i++, p += 4) {
      input[i] = rgba[p] / 255;
      input[area + i] = rgba[p + 1] / 255;
      input[2 * area + i] = rgba[p + 2] / 255;
    }

    const tensor = new ort.Tensor("float32", input, [1, 3, size, size]);
    const feeds = {};
    feeds[this.inputName] = tensor;
    const out = await this.session.run(feeds);
    return this.postprocess(out[this.outputName].data, w, h, scale, dx, dy, conf, iouThr);
  }

  postprocess(data, origW, origH, scale, padX, padY, conf, iouThr) {
    const n = 8400, nc = 5;
    const boxes = [];
    const scores = [];
    for (let i = 0; i < n; i++) {
      let score = 0;
      for (let k = 4; k < nc; k++) score = Math.max(score, data[k * n + i]);
      if (score <= conf) continue;
      const cx = data[i], cy = data[n + i], bw = data[2 * n + i], bh = data[3 * n + i];
      const x1 = Math.max(0, Math.min(origW, (cx - bw / 2 - padX) / scale));
      const y1 = Math.max(0, Math.min(origH, (cy - bh / 2 - padY) / scale));
      const x2 = Math.max(0, Math.min(origW, (cx + bw / 2 - padX) / scale));
      const y2 = Math.max(0, Math.min(origH, (cy + bh / 2 - padY) / scale));
      if (x2 - x1 < 20 || y2 - y1 < 20) continue;
      boxes.push([x1, y1, x2, y2]);
      scores.push(score);
    }
    return nms(boxes, scores, iouThr)
      .map((i) => ({ box: boxes[i], score: scores[i] }))
      .sort((a, b) => b.score - a.score);
  }
}
