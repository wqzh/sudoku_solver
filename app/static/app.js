(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const dropZone = $("dropZone");
  const fileInput = $("fileInput");
  const dzEmpty = $("dzEmpty");
  const dzPreview = $("dzPreview");
  const previewImg = $("previewImg");
  const fileMeta = $("fileMeta");
  const solveBtn = $("solveBtn");
  const clearBtn = $("clearBtn");
  const loading = $("loading");
  const placeholder = $("placeholder");
  const stage = $("stage");
  const resultImg = $("resultImg");
  const boxLayer = $("boxLayer");
  const boxToggle = $("boxToggle");
  const downloadBtn = $("downloadBtn");
  const badge = $("resultBadge");
  const stats = $("stats");
  const alertBox = $("alert");
  const boardBox = $("boardBox");
  const boardEl = $("board");
  const modelStatus = $("modelStatus");

  let currentFile = null;
  let currentUrl = null;
  let resultDataUrl = null;
  let boxes = [];
  let busy = false;

  /* ---------------- helpers ---------------- */
  const humanSize = (n) =>
    n > 1024 * 1024 ? (n / 1024 / 1024).toFixed(2) + " MB"
    : n > 1024 ? (n / 1024).toFixed(0) + " KB" : n + " B";

  function setBadge(text, kind) {
    badge.textContent = text;
    badge.className = "badge" + (kind ? " " + kind : "");
  }

  function showAlert(msg, isError) {
    if (!msg) {
      alertBox.hidden = true;
      return;
    }
    alertBox.textContent = msg;
    alertBox.className = "alert" + (isError ? " err" : "");
    alertBox.hidden = false;
  }

  function resetResult() {
    resultDataUrl = null;
    boxes = [];
    boxLayer.innerHTML = "";
    resultImg.removeAttribute("src");
    stage.hidden = true;
    placeholder.hidden = false;
    downloadBtn.disabled = true;
    stats.innerHTML = "";
    setBadge("等待上传");
    boardBox.hidden = true;
    showAlert("");
  }

  /* ---------------- model status ---------------- */
  fetch("/api/health")
    .then((r) => r.json())
    .then((d) => {
      if (d.ok) {
        modelStatus.textContent = "模型已就绪 · " + d.digit_model;
        modelStatus.className = "status ok";
      } else {
        modelStatus.textContent = "模型未就绪" + (d.error ? "：" + d.error : "");
        modelStatus.className = "status err";
      }
    })
    .catch(() => {
      modelStatus.textContent = "无法连接后端";
      modelStatus.className = "status err";
    });

  /* ---------------- file selection ---------------- */
  function acceptFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      showAlert("请选择图片文件（jpg / png / bmp / webp）", true);
      return;
    }
    currentFile = file;
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = URL.createObjectURL(file);
    previewImg.src = currentUrl;
    dzEmpty.hidden = true;
    dzPreview.hidden = false;

    fileMeta.hidden = false;
    fileMeta.innerHTML = `<span><b>${file.name}</b></span>
      <span>大小 ${humanSize(file.size)}</span>
      <span id="dimText">读取尺寸…</span>`;

    const probe = new Image();
    probe.onload = () => {
      const el = $("dimText");
      if (el) el.textContent = `尺寸 ${probe.naturalWidth} × ${probe.naturalHeight}`;
    };
    probe.src = currentUrl;

    solveBtn.disabled = false;
    clearBtn.disabled = false;
    resetResult();
  }

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", (e) => {
    acceptFile(e.target.files && e.target.files[0]);
    fileInput.value = "";
  });

  ["dragenter", "dragover"].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    })
  );
  dropZone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    acceptFile(f);
  });

  window.addEventListener("paste", (e) => {
    const items = e.clipboardData && e.clipboardData.items;
    if (!items) return;
    for (const it of items) {
      if (it.type && it.type.startsWith("image/")) {
        acceptFile(it.getAsFile());
        break;
      }
    }
  });

  clearBtn.addEventListener("click", () => {
    if (busy) return;
    currentFile = null;
    if (currentUrl) URL.revokeObjectURL(currentUrl);
    currentUrl = null;
    previewImg.removeAttribute("src");
    dzEmpty.hidden = false;
    dzPreview.hidden = true;
    fileMeta.hidden = true;
    fileMeta.innerHTML = "";
    solveBtn.disabled = true;
    clearBtn.disabled = true;
    resetResult();
  });

  /* ---------------- solve ---------------- */
  function renderBoxes() {
    boxLayer.innerHTML = "";
    if (!boxToggle.checked || !boxes.length) return;
    for (const b of boxes) {
      const d = document.createElement("div");
      d.className = "bbox";
      d.style.left = (b.x / b.imgW) * 100 + "%";
      d.style.top = (b.y / b.imgH) * 100 + "%";
      d.style.width = (b.w / b.imgW) * 100 + "%";
      d.style.height = (b.h / b.imgH) * 100 + "%";
      const tag = document.createElement("span");
      tag.textContent = `${b.label} ${(b.score * 100).toFixed(0)}%`;
      d.appendChild(tag);
      boxLayer.appendChild(d);
    }
  }

  boxToggle.addEventListener("change", renderBoxes);

  function renderBoard(board, solved) {
    if (!board || !solved) {
      boardBox.hidden = true;
      return;
    }
    boardEl.innerHTML = "";
    for (let r = 0; r < 9; r++) {
      for (let c = 0; c < 9; c++) {
        const cell = document.createElement("div");
        const given = board[r][c] > 0;
        const val = given ? board[r][c] : solved[r][c];
        cell.className = "cell" + (given ? "" : " fill");
        if ((c + 1) % 3 === 0 && c !== 8) cell.classList.add("border-r");
        if ((r + 1) % 3 === 0 && r !== 8) cell.classList.add("border-b");
        cell.textContent = val || "";
        boardEl.appendChild(cell);
      }
    }
    boardBox.hidden = false;
  }

  async function solve() {
    if (!currentFile || busy) return;
    busy = true;
    solveBtn.disabled = true;
    clearBtn.disabled = true;
    loading.hidden = false;
    placeholder.hidden = true;
    showAlert("");
    setBadge("处理中…");

    const fd = new FormData();
    fd.append("file", currentFile, currentFile.name);

    try {
      const resp = await fetch("/api/solve", { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.message || `HTTP ${resp.status}`);

      resultDataUrl = data.image;
      resultImg.src = data.image;
      stage.hidden = false;
      placeholder.hidden = true;
      downloadBtn.disabled = false;

      boxes = (data.boxes || []).map((b) => ({ ...b, imgW: data.width, imgH: data.height }));
      renderBoxes();

      const dims = `${data.width} × ${data.height}`;
      const sameSize = true;
      stats.innerHTML =
        `<span>尺寸 <b>${dims}</b>${sameSize ? "（与原图一致）" : ""}</span>` +
        `<span>检测框 <b>${(data.boxes || []).length}</b></span>` +
        (data.ok
          ? `<span>原有数字 <b>${data.givens}</b></span><span>填入 <b>${data.filled}</b></span>`
          : "") +
        `<span>耗时 <b>${data.seconds}s</b></span>`;

      if (data.ok) {
        setBadge("求解成功", "ok");
      } else if ((data.boxes || []).length) {
        setBadge("检测到网格但未求解", "warn");
      } else {
        setBadge("未检测到数独", "err");
      }
      showAlert(data.message || "", !data.ok);
      renderBoard(data.board, data.solved_board);
    } catch (err) {
      stage.hidden = true;
      placeholder.hidden = false;
      setBadge("处理失败", "err");
      showAlert(err.message || String(err), true);
    } finally {
      loading.hidden = true;
      busy = false;
      solveBtn.disabled = !currentFile;
      clearBtn.disabled = !currentFile;
    }
  }

  solveBtn.addEventListener("click", solve);
  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") solve();
  });

  downloadBtn.addEventListener("click", () => {
    if (!resultDataUrl) return;
    const base = currentFile ? currentFile.name.replace(/\.[^.]+$/, "") : "sudoku";
    const a = document.createElement("a");
    a.href = resultDataUrl;
    a.download = `${base}_solved.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  });
})();
