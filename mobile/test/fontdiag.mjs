import { createRequire } from "node:module";
const require = createRequire("C:/Users/F-117/AppData/Roaming/NextWork/SKILLs/pptx/node_modules/");
const { chromium } = require("playwright");
const browser = await chromium.launch({
  executablePath: "C:/Users/F-117/AppData/Local/ms-playwright/chromium-1140/chrome-win/chrome.exe",
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("console", (m) => console.log("[console]", m.text()));
await page.goto("http://127.0.0.1:8765/test/fonts.html");
await page.waitForFunction(() => window.__ready === true);
const diag = await page.evaluate(async () => {
  const { analyseGrid, IMG_PAD } = await import("/src/core/grid.js");
  const { extractCells } = await import("/src/core/cells.js");
  const margin = 46, cell = 84, side = margin * 2 + cell * 9;
  const c = document.createElement("canvas");
  c.width = side; c.height = side;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, side, side);
  ctx.strokeStyle = "#111"; ctx.lineWidth = 2;
  for (let i = 0; i <= 9; i++) {
    const p = margin + i * cell;
    ctx.beginPath(); ctx.moveTo(margin, p); ctx.lineTo(margin + cell * 9, p);
    ctx.moveTo(p, margin); ctx.lineTo(p, margin + cell * 9); ctx.stroke();
  }
  ctx.fillStyle = "#111"; ctx.font = '700 48px "Arial"';
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (let d = 1; d <= 9; d++) ctx.fillText(String(d), margin + (d - 1) * cell + cell / 2, margin + cell / 2 + 2);

  const pad = document.createElement("canvas");
  pad.width = side + 48; pad.height = side + 48;
  const pctx = pad.getContext("2d");
  pctx.fillStyle = "#fff"; pctx.fillRect(0, 0, pad.width, pad.height);
  pctx.drawImage(c, IMG_PAD, IMG_PAD);
  const data = pctx.getImageData(0, 0, pad.width, pad.height).data;
  const quad = [[IMG_PAD, IMG_PAD], [IMG_PAD + side - 1, IMG_PAD],
                [IMG_PAD + side - 1, IMG_PAD + side - 1], [IMG_PAD, IMG_PAD + side - 1]];
  const view = analyseGrid(data, pad.width, pad.height, quad, 0);
  if (!view) return { error: "analyse null" };
  let ink = 0; for (const v of view.binary) if (v) ink++;
  const cells = extractCells(view);
  let maxHp = 0, maxHpIdx = -1, maxVp = 0, maxVpIdx = -1;
  for (let i = 0; i < view.hp.length; i++) if (view.hp[i] > maxHp) { maxHp = view.hp[i]; maxHpIdx = i; }
  for (let i = 0; i < view.vp.length; i++) if (view.vp[i] > maxVp) { maxVp = view.vp[i]; maxVpIdx = i; }
  const bin = document.createElement("canvas");
  bin.width = view.side; bin.height = view.side;
  const bctx = bin.getContext("2d");
  const bid = bctx.createImageData(view.side, view.side);
  for (let i = 0; i < view.binary.length; i++) {
    const v = view.binary[i]; bid.data[i * 4] = v; bid.data[i * 4 + 1] = v; bid.data[i * 4 + 2] = v; bid.data[i * 4 + 3] = 255;
  }
  bctx.putImageData(bid, 0, 0);
  return {
    ink, side,
    maxHp, maxHpIdx, maxVp, maxVpIdx,
    binaryUrl: bin.toDataURL("image/png"),
    cellPresence: cells.map((x) => (x.present ? 1 : 0)).join(""),
    cellSums: cells.slice(0, 12).map((x) => Array.from(x.img).reduce((a, b) => a + b, 0)),
    hpSample: Array.from(view.hp).filter((_, i) => i % 80 === 0).map((v) => v.toFixed(2)),
  };
});
console.log(JSON.stringify({ ...diag, binaryUrl: diag.binaryUrl ? "<png>" : null }, null, 2));
if (diag.binaryUrl) {
  const fs = await import("node:fs");
  fs.writeFileSync("test/binary.png", Buffer.from(diag.binaryUrl.split(",")[1], "base64"));
  console.log("wrote test/binary.png");
}
await browser.close();
