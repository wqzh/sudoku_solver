import { solve, isBoardValid } from "../src/core/solver.js";
const b = [
  [0, 0, 0, 0, 0, 0, 0, 0, 8],
  [0, 0, 0, 7, 8, 0, 4, 0, 1],
  [0, 0, 1, 0, 0, 2, 0, 5, 0],
  [2, 6, 0, 0, 5, 0, 0, 0, 0],
  [3, 0, 0, 0, 0, 0, 8, 0, 4],
  [8, 1, 0, 0, 4, 0, 0, 0, 0],
  [0, 0, 5, 0, 0, 9, 0, 8, 0],
  [0, 0, 0, 6, 3, 0, 7, 0, 9],
  [0, 0, 0, 0, 0, 0, 0, 0, 3],
];
console.log("valid(flat):", isBoardValid(b.flat()));
console.log("valid(nested):", isBoardValid(b));
const t0 = Date.now();
const s = solve(b, {});
console.log("solved:", !!s, Date.now() - t0, "ms");
if (s) console.log(s.map((r) => r.join("")).join("\n"));
