import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire("C:/Users/F-117/AppData/Roaming/NextWork/SKILLs/pptx/node_modules/");
const { chromium } = require("playwright");
const here = path.dirname(fileURLToPath(import.meta.url));

const browser = await chromium.launch({
  executablePath: "C:/Users/F-117/AppData/Local/ms-playwright/chromium-1140/chrome-win/chrome.exe",
});
const errors = [];

const desktop = await browser.newPage({ viewport: { width: 1440, height: 980 } });
desktop.on("pageerror", (e) => errors.push("pageerror: " + e.message));
desktop.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
await desktop.goto("http://127.0.0.1:8765/demo.html");
await desktop.waitForTimeout(4000);
await desktop.screenshot({ path: path.join(here, "demo-desktop.png"), fullPage: true });

const frame = desktop.frameLocator("#appFrame");
await desktop.locator("#trySample").click();
await frame.locator("#solve").click();
await frame.locator("#resultCard").waitFor({ state: "visible", timeout: 120000 });
console.log("iframe solve stats:", (await frame.locator("#stats").innerText()).replace(/\n/g, " | "));
await desktop.screenshot({ path: path.join(here, "demo-tried.png"), fullPage: false });

const mobile = await browser.newPage({
  viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
});
mobile.on("pageerror", (e) => errors.push("mobile pageerror: " + e.message));
await mobile.goto("http://127.0.0.1:8765/demo.html");
await mobile.waitForTimeout(2500);
await mobile.screenshot({ path: path.join(here, "demo-mobile.png"), fullPage: true });
const overflow = await mobile.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
console.log("mobile overflow:", overflow, "errors:", errors.length ? errors : "none");

await browser.close();
