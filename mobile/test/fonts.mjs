import { createRequire } from "node:module";
const require = createRequire("C:/Users/F-117/AppData/Roaming/NextWork/SKILLs/pptx/node_modules/");
const { chromium } = require("playwright");
const browser = await chromium.launch({
  executablePath: "C:/Users/F-117/AppData/Local/ms-playwright/chromium-1140/chrome-win/chrome.exe",
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto("http://127.0.0.1:8765/test/fonts.html");
await page.waitForFunction(() => window.__ready === true);
const res = await page.evaluate(() => window.__run());
console.log(res.map((r) => `${r.ok ? "OK  " : "FAIL"} ${r.font} ${r.weight}: ${r.pred}`).join("\n"));
await browser.close();
