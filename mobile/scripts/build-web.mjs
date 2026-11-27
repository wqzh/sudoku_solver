// Copies the runnable web app into ./dist for Capacitor (does not touch src/).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");

const entries = [
  "index.html",
  "manifest.webmanifest",
  "assets",
  "src",
  "vendor",
  "models",
];

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(dist, { recursive: true });

function copy(src, dst) {
  const st = fs.statSync(src);
  if (st.isDirectory()) {
    fs.mkdirSync(dst, { recursive: true });
    for (const name of fs.readdirSync(src)) copy(path.join(src, name), path.join(dst, name));
  } else {
    fs.copyFileSync(src, dst);
  }
}

for (const e of entries) {
  const src = path.join(root, e);
  if (!fs.existsSync(src)) {
    console.warn("skip missing:", e);
    continue;
  }
  copy(src, path.join(dist, e));
}

let bytes = 0;
let files = 0;
(function walk(dir) {
  for (const name of fs.readdirSync(dir)) {
    const p = path.join(dir, name);
    const st = fs.statSync(p);
    if (st.isDirectory()) walk(p);
    else { bytes += st.size; files++; }
  }
})(dist);

console.log(`dist ready: ${files} files, ${(bytes / 1024 / 1024).toFixed(2)} MB`);
