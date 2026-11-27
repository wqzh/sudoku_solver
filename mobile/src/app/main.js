/* Mobile app controller: pick/capture -> solve -> show answer. */
import { solveImage } from "../core/pipeline.js";
import { bootEngine, MODEL_URLS } from "./boot.js";

const $ = (id) => document.getElementById(id);

const els = {
  status: $("status"),
  drop: $("drop"),
  file: $("file"),
  preview: $("preview"),
  pick: $("pick"),
  sample: $("sample"),
  solve: $("solve"),
  progress: $("progress"),
  progressText: $("progressText"),
  result: $("resultCard"),
  tabs: document.querySelectorAll(".tabs button"),
  paneImage: $("paneImage"),
  paneBoard: $("paneBoard"),
  out: $("out"),
  board: $("board"),
  stats: $("stats"),
  download: $("download"),
  badge: $("resultBadge"),
  msg: $("msg"),
  toast: $("toast"),
  engineNote: $("engineNote"),
  notice: $("notice"),
};

let source = null;          // full-resolution canvas of the chosen photo
let answer = null;          // canvas with the solved digits
let busy = false;
let engine = null;

/* ---------------------------------- utils --------------------------------- */

let toastTimer = 0;
function toast(text, isError = false) {
  els.toast.textContent = text;
  els.toast.className = "toast show" + (isError ? " err" : "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (els.toast.className = "toast"), 2600);
}

function setStatus(text, kind = "") {
  els.status.textContent = text;
  els.status.className = "pill" + (kind ? " " + kind : "");
}

function showMessage(text, isError = false) {
  const el = els.notice;
  if (!el) return;
  el.textContent = text || "";
  el.className = "msg" + (isError ? " err" : "");
  el.hidden = !text;
}

function setProgress(text) {
  if (text == null) {
    els.progress.hidden = true;
    return;
  }
  els.progressText.textContent = text;
  els.progress.hidden = false;
}

/* -------------------------------- image input ----------------------------- */

function canvasFromImage(img) {
  const MAX = 2800;
  const scale = Math.min(1, MAX / Math.max(img.width, img.height));
  const c = document.createElement("canvas");
  c.width = Math.round(img.width * scale);
  c.height = Math.round(img.height * scale);
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, c.width, c.height);
  return c;
}

function useSource(canvas) {
  source = canvas;
  answer = null;
  els.drop.classList.add("has-image");
  const maxW = 1200;
  const scale = Math.min(1, maxW / canvas.width);
  els.preview.width = Math.round(canvas.width * scale);
  els.preview.height = Math.round(canvas.height * scale);
  const ctx = els.preview.getContext("2d");
  ctx.drawImage(canvas, 0, 0, els.preview.width, els.preview.height);
  els.solve.disabled = false;
  els.result.hidden = true;
  showMessage("");
  setStatus(engine ? "模型就绪" : "模型加载中", engine ? "ok" : "");
}

async function loadFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    toast("请选择图片文件", true);
    return;
  }
  try {
    let img;
    if ("createImageBitmap" in window) {
      img = await createImageBitmap(file, { imageOrientation: "from-image" });
    } else {
      img = await new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => res(im);
        im.onerror = rej;
        im.src = URL.createObjectURL(file);
      });
    }
    useSource(canvasFromImage(img));
  } catch (err) {
    toast("这张图片无法读取，请换 JPG/PNG 格式", true);
  }
}

els.pick.addEventListener("click", (e) => {
  e.stopPropagation();
  els.file.click();
});
els.drop.addEventListener("click", () => els.file.click());
els.file.addEventListener("change", (e) => {
  loadFile(e.target.files && e.target.files[0]);
  els.file.value = "";
});

["dragenter", "dragover"].forEach((ev) =>
  els.drop.addEventListener(ev, (e) => {
    e.preventDefault();
    els.drop.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  els.drop.addEventListener(ev, (e) => {
    e.preventDefault();
    els.drop.classList.remove("dragover");
  })
);
els.drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  loadFile(f);
});
window.addEventListener("paste", (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const it of items) {
    if (it.type.startsWith("image/")) {
      loadFile(it.getAsFile());
      break;
    }
  }
});

/* ------------------------------- sample puzzle ---------------------------- */

async function loadSample() {
  const img = await new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = "/assets/sample.png";
  });
  useSource(canvasFromImage(img));
  toast("已载入示例题，点「求解」试试");
}

els.sample.addEventListener("click", (e) => {
  e.stopPropagation();
  loadSample().catch(() => toast("示例题载入失败", true));
});

/* ---------------------------------- solve --------------------------------- */

function renderBoard(board, solved) {
  els.board.innerHTML = "";
  for (let r = 0; r < 9; r++) {
    for (let col = 0; col < 9; col++) {
      const d = document.createElement("div");
      const given = board[r][col] > 0;
      d.className = "cell" + (given ? " given" : " filled");
      if ((col + 1) % 3 === 0 && col !== 8) d.classList.add("bx");
      if ((r + 1) % 3 === 0 && r !== 8) d.classList.add("by");
      d.textContent = String(given ? board[r][col] : solved[r][col]);
      els.board.appendChild(d);
    }
  }
}

function renderStats(res) {
  const items = [
    ["已知数字", res.givens],
    ["填入", res.filled],
    ["网格得分", res.score ? res.score.toFixed(2) : "—"],
    ["耗时", res.seconds.toFixed(1) + "s"],
  ];
  els.stats.innerHTML = items
    .map(([k, v]) => `<span class="stat">${k}<b>${v}</b></span>`)
    .join("");
}

function showResult(res) {
  answer = res.canvas;
  els.result.hidden = false;
  els.out.width = res.canvas.width;
  els.out.height = res.canvas.height;
  els.out.getContext("2d").drawImage(res.canvas, 0, 0);
  renderBoard(res.board, res.solved);
  renderStats(res);
  els.badge.textContent = "求解成功";
  els.badge.className = "badge";
  showMessage(res.message);
  setStatus("已求解", "ok");
  els.result.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function solve() {
  if (!source || busy) return;
  busy = true;
  els.solve.disabled = true;
  els.sample.disabled = true;
  els.result.hidden = true;
  showMessage("");
  setStatus("处理中…");
  setProgress("准备中…");

  try {
    const res = await solveImage(source, engine, {
      timeBudget: 30000,
      onProgress: setProgress,
    });
    if (res.ok) {
      showResult(res);
    } else {
      els.badge.textContent = "求解失败";
      els.result.hidden = true;
      setStatus("未求解", "err");
      showMessage(res.message, true);
      toast(res.message, true);
    }
  } catch (err) {
    setStatus("出错", "err");
    showMessage("处理失败：" + (err && err.message ? err.message : err), true);
    toast("处理失败，请重试", true);
  } finally {
    busy = false;
    els.solve.disabled = !source;
    els.sample.disabled = false;
    setProgress(null);
  }
}

els.solve.addEventListener("click", solve);

els.tabs.forEach((btn) => {
  btn.addEventListener("click", () => {
    els.tabs.forEach((b) => b.classList.toggle("active", b === btn));
    const boardTab = btn.dataset.tab === "board";
    els.paneImage.hidden = boardTab;
    els.paneBoard.hidden = !boardTab;
  });
});

els.download.addEventListener("click", () => {
  if (!answer) return;
  answer.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "sudoku_solved.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, "image/png");
});

/* ---------------------------------- boot ---------------------------------- */

(async () => {
  try {
    const t0 = performance.now();
    engine = await bootEngine(MODEL_URLS);
    const ms = Math.round(performance.now() - t0);
    setStatus("模型就绪", "ok");
    if (els.engineNote) {
      els.engineNote.textContent = `两个模型共 3.4 MB，已在浏览器内加载（${ms} ms），推理全程在本机完成。`;
    }
  } catch (err) {
    setStatus("模型加载失败", "err");
    showMessage("模型加载失败：" + (err && err.message ? err.message : err), true);
    toast("模型加载失败", true);
  }
})();

window.__app = { solve, useSource, get source() { return source; }, get engine() { return engine; } };

// demo page controls this embedded app
window.addEventListener("message", (e) => {
  if (e.data && e.data.type === "nextwork:sample") els.sample.click();
});
