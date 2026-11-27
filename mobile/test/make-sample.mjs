// Generates assets/sample.png (a clean sample sudoku puzzle for the demo page).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire("C:/Users/F-117/AppData/Roaming/NextWork/SKILLs/pptx/node_modules/");
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));

const PUZZLE = [
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

const browser = await chromium.launch({
  executablePath: "C:/Users/F-117/AppData/Local/ms-playwright/chromium-1140/chrome-win/chrome.exe",
});
const page = await browser.newPage();
await page.setContent("<html><body></body></html>");
const dataUrl = await page.evaluate((puzzle) => {
  const margin = 56, cell = 92, side = margin * 2 + cell * 9;
  const c = document.createElement("canvas");
  c.width = side; c.height = side;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, side, side);

  ctx.strokeStyle = "#141414"; ctx.lineCap = "square";
  for (let i = 0; i <= 9; i++) {
    const p = margin + i * cell;
    ctx.lineWidth = i % 3 === 0 ? 6 : 2;
    ctx.beginPath();
    ctx.moveTo(margin, p); ctx.lineTo(margin + cell * 9, p);
    ctx.moveTo(p, margin); ctx.lineTo(p, margin + cell * 9);
    ctx.stroke();
  }
  ctx.fillStyle = "#141414";
  ctx.font = '700 58px Verdana, "DejaVu Sans", sans-serif';
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (let r = 0; r < 9; r++) {
    for (let col = 0; col < 9; col++) {
      const v = puzzle[r][col];
      if (!v) continue;
      ctx.fillText(String(v), margin + col * cell + cell / 2, margin + r * cell + cell / 2 + 3);
    }
  }
  return c.toDataURL("image/png");
}, PUZZLE);

fs.writeFileSync(path.join(here, "..", "assets", "sample.png"),
  Buffer.from(dataUrl.split(",")[1], "base64"));
console.log("wrote assets/sample.png");
await browser.close();
