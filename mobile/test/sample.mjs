import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire("C:/Users/F-117/AppData/Roaming/NextWork/SKILLs/pptx/node_modules/");
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));

const browser = await chromium.launch({
  executablePath: "C:/Users/F-117/AppData/Local/ms-playwright/chromium-1140/chrome-win/chrome.exe",
});
const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
page.on("console", (m) => { if (m.type() === "error") console.log("[console]", m.text()); });
await page.goto("http://127.0.0.1:8765/index.html");
await page.waitForFunction(() => document.getElementById("status").textContent.includes("就绪"), null, { timeout: 120000 });
await page.locator("#sample").click();

const res = await page.evaluate(async () => {
  const { solveImage } = await import("/src/core/pipeline.js");
  const logs = [];
  const r = await solveImage(window.__app.source, window.__app.engine, {
    timeBudget: 40000, onProgress: (m) => logs.push(m),
  });
  return {
    ok: r.ok, message: r.message, seconds: r.seconds, logs,
    attempts: r.attempts,
    boxes: r.boxes,
    score: r.score,
  };
});
console.log(JSON.stringify(res, null, 2));
await browser.close();
