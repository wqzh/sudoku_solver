// Regression test: solve ../.test_input/15.jpg in Chromium and compare to docs/baseline.json.
// Usage: node test/solve.mjs [imagePath]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire("C:/Users/F-117/AppData/Roaming/NextWork/SKILLs/pptx/node_modules/");
const { chromium } = require("playwright");

const here = path.dirname(fileURLToPath(import.meta.url));
const imgPath = process.argv[2] || path.join(here, "..", "..", ".test_input", "15.jpg");
const baseline = JSON.parse(fs.readFileSync(path.join(here, "..", "docs", "baseline.json"), "utf8"));
const b64 = "data:image/jpeg;base64," + fs.readFileSync(imgPath).toString("base64");

const browser = await chromium.launch({
  executablePath: "C:/Users/F-117/AppData/Local/ms-playwright/chromium-1140/chrome-win/chrome.exe",
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("console", (m) => { if (m.type() === "error") console.log("[console]", m.text()); });
await page.goto("http://127.0.0.1:8765/test/solve.html?threads=" + (process.env.THREADS || 1));
await page.waitForFunction(() => window.__status === "READY", null, { timeout: 120000 });
console.log("engine loaded in", await page.evaluate(() => window.__ready), "ms");

const t0 = Date.now();
const res = await page.evaluate((d) => window.__run(d), b64);
console.log("wall:", ((Date.now() - t0) / 1000).toFixed(2), "s; pipeline:", res.seconds.toFixed(2), "s");
console.log("ok:", res.ok, "|", res.message);
console.log("score:", res.score, "rotation:", res.rotation, "givens:", res.givens);

const show = (b) => b.map((r) => r.map((v) => (v || ".")).join("")).join("\n");
for (const a of res.attempts || []) {
  console.log(`\n--- attempt cand${a.ci} rot${a.k * 90} givens=${a.givens}`);
  console.log(show(a.board));
  const diff = [];
  for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) {
    const exp = baseline.board[r][c], got = a.board[r][c];
    if (exp !== got) diff.push(`r${r + 1}c${c + 1}: ${got || "."} != ${exp || "."}`);
  }
  console.log("diff vs baseline:", diff.length ? diff.join(", ") : "none");
}

if (res.warpUrl) {
  fs.writeFileSync(path.join(here, "warp.png"), Buffer.from(res.warpUrl.split(",")[1], "base64"));
  console.log("wrote test/warp.png");
}
if (res.outUrl) {
  fs.writeFileSync(path.join(here, "out-web.png"), Buffer.from(res.outUrl.split(",")[1], "base64"));
  console.log("wrote test/out-web.png");
}

if (res.ok) {
  const sameBoard = JSON.stringify(res.board) === JSON.stringify(baseline.board);
  const sameSolved = JSON.stringify(res.solved) === JSON.stringify(baseline.solved_board);
  console.log("board matches baseline: ", sameBoard);
  console.log("solved matches baseline:", sameSolved);
  if (!sameBoard || !sameSolved) process.exitCode = 2;
} else {
  process.exitCode = 1;
}
await browser.close();
