/* Digit classifier wrapper: batches all present cells into one ONNX call. */

export class DigitClassifier {
  constructor(session) {
    this.session = session;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
  }

  /** cells: [{img: Float32Array(784)}] -> scores: Float32Array(9)[] */
  async predict(cells) {
    if (!cells.length) return [];
    const n = cells.length;
    const data = new Float32Array(n * 28 * 28);
    for (let i = 0; i < n; i++) data.set(cells[i].img, i * 784);
    const tensor = new ort.Tensor("float32", data, [n, 28, 28, 1]);
    const feeds = {};
    feeds[this.inputName] = tensor;
    const out = await this.session.run(feeds);
    const res = out[this.outputName].data;
    const scores = [];
    for (let i = 0; i < n; i++) scores.push(res.subarray(i * 9, i * 9 + 9));
    return scores;
  }
}

/** board 9x9 + scores -> {board, digitIdx, scores}. */
export function readBoard(cells, scores) {
  const board = Array.from({ length: 9 }, () => new Array(9).fill(0));
  const digitIdx = [];
  const scoreRows = [];
  let s = 0;
  for (let i = 0; i < cells.length; i++) {
    if (!cells[i].present) continue;
    const row = (i / 9) | 0, col = i % 9;
    const sc = scores[s++];
    let best = 0;
    for (let k = 1; k < sc.length; k++) if (sc[k] > sc[best]) best = k;
    board[row][col] = best + 1;
    digitIdx.push(i);
    scoreRows.push(sc);
  }
  return { board, digitIdx, scores: scoreRows };
}
