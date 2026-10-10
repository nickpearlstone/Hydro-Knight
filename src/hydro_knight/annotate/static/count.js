// Hydro-Knight labeler, Swimmer count tab: one point per person on still frames.
// Points are pixels in the original frame. Zoom/pan works like the Rescue timeline tab;
// the frame always opens fitted to the stage, so nothing starts off-screen.

const $ = (s) => document.querySelector(s);

const img = $("#frame");
const overlay = $("#overlay");
const ctx = overlay.getContext("2d");
const stage = $("#stage");
const zoomer = $("#zoomer");

const S = {
  frames: [],
  idx: -1,
  w: 1, h: 1,
  history: [],       // [{idx, water, deck}] snapshots for undo
  zoom: { k: 1, tx: 0, ty: 0, bx: 0, by: 0, bw: 0, bh: 0 },
  saveTimer: null,
};

const cur = () => S.frames[S.idx];

// ---------------------------------------------------------------- feedback

function toast(msg, kind = "ok") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 2600);
}

function setSaveState(state, text) {
  const pill = $("#saveState");
  pill.className = `pill ${state}`;
  pill.textContent = text;
}

// ---------------------------------------------------------------- frames

async function loadFrames() {
  const res = await fetch("/api/count");
  if (!res.ok) { $("#emptyState").hidden = false; setSaveState("", "—"); return; }
  S.frames = await res.json();
  if (!S.frames.length) { $("#emptyState").hidden = false; return; }
  const saved = Number(localStorage.getItem("hk.countIdx"));
  await openFrame(saved >= 0 && saved < S.frames.length ? saved : 0);
}

async function openFrame(i) {
  if (i < 0 || i >= S.frames.length) return;
  await flushSave();
  S.idx = i;
  localStorage.setItem("hk.countIdx", i);
  const f = cur();
  $("#clipId").textContent = `${f.clip_id} · frame ${f.frame}`;
  $("#clipNotes").textContent = f.label || "";
  img.src = `/api/count/${i}/image`;
  await img.decode().catch(() => toast("Couldn't load this frame", "error"));
  S.w = img.naturalWidth || 1;
  S.h = img.naturalHeight || 1;
  overlay.width = S.w;
  overlay.height = S.h;
  fitStage(true);
  setSaveState("saved", "Saved ✓");
  render();
}

// ---------------------------------------------------------------- edits + save

function edit(fn) {
  const f = cur();
  S.history.push({ idx: S.idx, water: [...f.water], deck: [...f.deck] });
  if (S.history.length > 500) S.history.shift();
  fn(f);
  render();
  scheduleSave();
}

async function undo() {
  const prev = S.history.pop();
  if (!prev) { toast("Nothing to undo", "warn"); return; }
  if (prev.idx !== S.idx) await openFrame(prev.idx);
  Object.assign(cur(), { water: prev.water, deck: prev.deck });
  render();
  scheduleSave();
  toast("Undone");
}

function scheduleSave() {
  clearTimeout(S.saveTimer);
  setSaveState("saving", "Saving…");
  const idx = S.idx;
  S.saveTimer = setTimeout(() => { S.saveTimer = null; save(idx); }, 300);
}

async function flushSave() {
  if (!S.saveTimer) return;
  clearTimeout(S.saveTimer);
  S.saveTimer = null;
  await save(S.idx);
}

async function save(idx) {
  const f = S.frames[idx];
  try {
    const res = await fetch(`/api/count/${idx}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ water: f.water, deck: f.deck }),
    });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    if (idx === S.idx) setSaveState("saved", "Saved ✓");
    renderList();
  } catch (e) {
    setSaveState("error", "Save failed");
    toast(`Save failed: ${e.message}`, "error");
  }
}

// ---------------------------------------------------------------- clicks

function toFrame(clientX, clientY) {
  const r = img.getBoundingClientRect();
  return [((clientX - r.left) / r.width) * S.w, ((clientY - r.top) / r.height) * S.h];
}

function addPoint(e) {
  const [x, y] = toFrame(e.clientX, e.clientY);
  if (x < 0 || y < 0 || x > S.w || y > S.h) return;
  const kind = e.shiftKey ? "deck" : "water";
  edit((f) => f[kind].push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]));
}

function removeNearest(e) {
  const [x, y] = toFrame(e.clientX, e.clientY);
  const reach = 14 * unit(); // about 14 screen pixels at any zoom
  let best = null;
  for (const kind of ["water", "deck"]) {
    cur()[kind].forEach(([px, py], j) => {
      const d = Math.hypot(px - x, py - y);
      if (d <= reach && (!best || d < best.d)) best = { kind, j, d };
    });
  }
  if (!best) { toast("No dot here to remove", "warn"); return; }
  edit((f) => f[best.kind].splice(best.j, 1));
}

// ---------------------------------------------------------------- zoom + pan

function fitStage(reset) {
  const sw = stage.clientWidth, sh = stage.clientHeight;
  const scale = Math.min(sw / S.w, sh / S.h);
  const z = S.zoom;
  z.bw = S.w * scale; z.bh = S.h * scale;
  z.bx = (sw - z.bw) / 2; z.by = (sh - z.bh) / 2;
  if (reset) { z.k = 1; z.tx = 0; z.ty = 0; }
  applyZoom();
}

function applyZoom() {
  const z = S.zoom;
  z.tx = Math.min(0, Math.max(z.bw - z.bw * z.k, z.tx));
  z.ty = Math.min(0, Math.max(z.bh - z.bh * z.k, z.ty));
  Object.assign(zoomer.style, {
    left: `${z.bx}px`, top: `${z.by}px`, width: `${z.bw}px`, height: `${z.bh}px`,
    transform: `translate(${z.tx}px, ${z.ty}px) scale(${z.k})`,
  });
  $("#zoomBadge").hidden = z.k <= 1.001;
  $("#zoomBadge").textContent = `${z.k.toFixed(1)}×`;
  drawOverlay();
}

// Frame pixels per screen pixel at the current fit and zoom.
const unit = () => S.w / (S.zoom.bw * S.zoom.k || 1);

stage.addEventListener("wheel", (e) => {
  e.preventDefault();
  const z = S.zoom;
  const rect = stage.getBoundingClientRect();
  const px = e.clientX - rect.left - z.bx, py = e.clientY - rect.top - z.by;
  const lx = (px - z.tx) / z.k, ly = (py - z.ty) / z.k;
  const k = Math.min(12, Math.max(1, z.k * Math.exp(-e.deltaY * 0.0015)));
  z.tx = px - lx * k; z.ty = py - ly * k; z.k = k;
  applyZoom();
}, { passive: false });

let drag = null;
stage.addEventListener("mousedown", (e) => {
  if (e.button !== 0) return;
  drag = { x: e.clientX, y: e.clientY, tx: S.zoom.tx, ty: S.zoom.ty, moved: false };
});
window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 4) return;
  drag.moved = true;
  if (S.zoom.k > 1) {
    stage.classList.add("panning");
    S.zoom.tx = drag.tx + dx; S.zoom.ty = drag.ty + dy;
    applyZoom();
  }
});
window.addEventListener("mouseup", (e) => {
  if (!drag) return;
  const wasClick = !drag.moved;
  drag = null;
  stage.classList.remove("panning");
  if (wasClick && stage.contains(e.target) && S.idx >= 0) addPoint(e);
});
stage.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (S.idx >= 0) removeNearest(e);
});
window.addEventListener("resize", () => fitStage(false));

// ---------------------------------------------------------------- rendering

function drawOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const f = cur();
  if (!f) return;
  const u = unit();
  const colors = { water: "#ef4444", deck: "#3b82f6" };
  for (const kind of ["water", "deck"]) {
    ctx.fillStyle = colors[kind];
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5 * u;
    for (const [x, y] of f[kind]) {
      ctx.beginPath();
      ctx.arc(x, y, 5 * u, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}

function renderList() {
  $("#frameList").innerHTML = S.frames.map((f, i) => {
    const n = f.water.length + f.deck.length;
    return `<li data-i="${i}" class="${i === S.idx ? "current" : ""} ${n ? "" : "blank"}">
      <span class="mono">${f.clip_id}</span>
      <span class="muted">${n ? `${f.water.length} + ${f.deck.length}` : "not started"}</span>
    </li>`;
  }).join("");
}

function render() {
  const f = cur();
  $("#nWater").textContent = f.water.length;
  $("#nDeck").textContent = f.deck.length;
  $("#counter").textContent = `Frame ${S.idx + 1} of ${S.frames.length}`;
  $("#prev").disabled = S.idx <= 0;
  $("#next").disabled = S.idx >= S.frames.length - 1;
  renderList();
  drawOverlay();
}

// ---------------------------------------------------------------- wiring

$("#prev").onclick = () => openFrame(S.idx - 1);
$("#next").onclick = () => openFrame(S.idx + 1);
$("#zoomReset").onclick = () => fitStage(true);
$("#undoBtn").onclick = undo;
$("#frameList").addEventListener("click", (e) => {
  const li = e.target.closest("[data-i]");
  if (li) openFrame(Number(li.dataset.i));
});

document.addEventListener("keydown", (e) => {
  if (S.idx < 0) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); openFrame(S.idx - 1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); openFrame(S.idx + 1); }
  else if (e.key === "0") fitStage(true);
  else if (e.key.toLowerCase() === "z" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); undo(); }
});

// Closing the tab with a save still waiting on its timer: send it anyway.
window.addEventListener("beforeunload", () => {
  if (!S.saveTimer) return;
  const f = cur();
  navigator.sendBeacon(`/api/count/${S.idx}`,
    new Blob([JSON.stringify({ water: f.water, deck: f.deck })], { type: "application/json" }));
});

loadFrames();
