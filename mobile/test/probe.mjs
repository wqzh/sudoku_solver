// Probe: load both ONNX models inside Chromium (onnxruntime-web WASM).
// Usage: node test/probe.mjs
import { createRequire } from "node:module";
const require = createRequire("C:/Users/F-117/AppData/Roaming/NextWork/SKILLs/pptx/node_modules/");
const { chromium } = require("playwright");

const browser = await chromium.launch({
  executablePath: "C:/Users/F-117/AppData/Local/ms-playwright/chromium-1140/chrome-win/chrome.exe",
  args: ["--enable-features=SharedArrayBuffer"],
});
const page = await browser.newPage();
page.on("console", (m) => console.log("[console]", m.type(), m.text()));
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto("http://127.0.0.1:8765/test/probe.html");
await page.waitForFunction(() => document.title.startsWith("PROBE_"), null, { timeout: 180000 });
console.log(await page.locator("#out").innerText());
console.log("title:", await page.title());
await browser.close();
