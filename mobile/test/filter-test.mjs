import { maxFilter, minFilter } from "../src/core/image.js";

const w = 20, h = 1;
const a = new Uint8ClampedArray(w);
a[10] = 255;
const mx = maxFilter(a, w, h, 2, true);
const mn = minFilter(a, w, h, 2, true);
console.log("max:", Array.from(mx).join(","));
console.log("min:", Array.from(mn).join(","));

const b = new Uint8ClampedArray(w * 3);
for (let x = 5; x < 15; x++) b[x] = 255;
const mn2 = minFilter(b, w, 3, 2, true);
console.log("row0 min:", Array.from(mn2.slice(0, w)).join(","));
