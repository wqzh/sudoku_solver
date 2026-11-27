// UI smoke test: mobile viewport, sample puzzle + real photo, screenshots.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire("C:/Users/F-117/AppData/Roaming/NextWork/SKILLs/pptx/node_modules/");
const { chromium } = require("playwright");

const here = path.dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch({
  executablePath: "C:/Users/F-117/AppData/Local/ms-playwright/chromium-1140/chrome-win/chrome.exe",
});
const context = await browser.newContext({
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
  isMobile: true,
  hasTouch: true,
});
const page = await context.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("favicon")) errors.push("console: " + m.text()); });

await page.goto("http://127.0.0.1:8765/index.html");
await page.waitForFunction(() => document.getElementById("status").textContent.includes("就绪"), null, { timeout: 120000 });
console.log("status:", await page.locator("#status").innerText());
console.log("engine note:", await page.locator("#engineNote").innerText());

// 1) sample puzzle
const t0 = Date.now();
await page.locator("#sample").click();
await page.locator("#solve").click();
await page.waitForSelector("#resultCard:not([hidden])", { timeout: 120000 });
console.log("sample solved in", ((Date.now() - t0) / 1000).toFixed(2), "s");
console.log("badge:", await page.locator("#resultBadge").innerText());
console.log("stats:", (await page.locator("#stats").innerText()).replace(/\n/g, " | "));
await page.screenshot({ path: path.join(here, "ui-sample.png"), fullPage: true });

// board tab
await page.locator('.tabs button[data-tab="board"]').click();
await page.waitForTimeout(200);
await page.screenshot({ path: path.join(here, "ui-board.png"), fullPage: false });

// 2) real photo through the file input
const imgPath = path.join(here, "..", "..", ".test_input", "15.jpg");
await page.locator('.tabs button[data-tab="image"]').click();
await page.locator("#file").setInputFiles(imgPath);
await page.waitForTimeout(800);
const t1 = Date.now();
await page.locator("#solve").click();
await page.waitForSelector("#resultCard:not([hidden])", { timeout: 180000 });
console.log("photo solved in", ((Date.now() - t1) / 1000).toFixed(2), "s");
console.log("photo stats:", (await page.locator("#stats").innerText()).replace(/\n/g, " | "));
await page.screenshot({ path: path.join(here, "ui-photo.png"), fullPage: true });

// horizontal overflow check
const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
console.log("horizontal overflow px:", overflow);
console.log("errors:", errors.length ? errors : "none");

await browser.close();
