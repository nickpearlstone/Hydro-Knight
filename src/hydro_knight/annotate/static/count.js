// Hydro-Knight labeler, Swimmer count tab: one point per person on still frames, and in
// head-box mode one box per visible head. Points and boxes are pixels in the original frame.
// Zoom/pan works like the Rescue timeline tab; the frame always opens fitted to the stage,
// so nothing starts off-screen. Two frame sets share the tab: the benchmark (test) frames
// and the head validation frames (?set=val).

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
  set: localStorage.getItem("hk.countSet") === "val" ? "val" : "bench",
  mode: localStorage.getItem("hk.countMode") === "heads" ? "heads" : "dots",
  history: [],       // [{idx, snap}] snapshots for undo
  zoom: { k: 1, tx: 0, ty: 0, bx: 0, by: 0, bw: 0, bh: 0 },
  saveTimer: null,
  space: false,      // space held: drag pans in head-box mode
  box: null,         // head box being drawn: {x0, y0, x1, y1, deck}
};

const cur = () => S.frames[S.idx];
const api = (path) => `/api/count${path}?set=${S.set}`;
const idxKey = () => (S.set === "bench" ? "hk.countIdx" : "hk.countIdx.val");
const heads = (f) => (f.heads ||= { water: [], deck: [], no_head: [] });
const snap = (f) => JSON.parse(JSON.stringify({
  water: f.water, deck: f.deck, hidden: f.hidden ?? [],
  heads: f.heads ?? null, heads_reviewed: f.heads_reviewed ?? null,
}));

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
  await flushSave();
  S.frames = [];
  S.idx = -1;
  S.history = [];
  $("#emptyState").hidden = true;
  const res = await fetch(api(""));
  if (!res.ok) {
    $("#emptyState").textContent = res.status === 404
      ? (S.set === "val"
        ? "No validation frames yet. Run: uv run hydroknight heads frames"
        : "No count frames yet. Run: uv run hydroknight benchmark pose frames")
      : "Couldn't load the frames.";
    $("#emptyState").hidden = false;
    img.removeAttribute("src");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    $("#frameList").innerHTML = "";
    setSaveState("", "—");
    return;
  }
  S.frames = await res.json();
  if (!S.frames.length) { $("#emptyState").hidden = false; return; }
  const saved = Number(localStorage.getItem(idxKey()));
  await openFrame(saved >= 0 && saved < S.frames.length ? saved : 0);
}

async function openFrame(i) {
  if (i < 0 || i >= S.frames.length) return;
  await flushSave();
  S.idx = i;
  localStorage.setItem(idxKey(), i);
  const f = cur();
  $("#clipId").textContent = `${f.clip_id} · frame ${f.frame}`;
  $("#clipNotes").textContent = f.label || "";
  img.src = api(`/${i}/image`);
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
  S.history.push({ idx: S.idx, snap: snap(f) });
  if (S.history.length > 500) S.history.shift();
  fn(f);
  render();
  scheduleSave();
}

async function undo() {
  const prev = S.history.pop();
  if (!prev) { toast("Nothing to undo", "warn"); return; }
  if (prev.idx !== S.idx) await openFrame(prev.idx);
  const f = cur(), s = prev.snap;
  Object.assign(f, { water: s.water, deck: s.deck, hidden: s.hidden });
  // A frame that had no head boxes yet gets empty lists (not none), so the save clears them.
  f.heads = s.heads ?? (f.heads ? { water: [], deck: [], no_head: [] } : undefined);
  f.heads_reviewed = s.heads_reviewed ?? (typeof f.heads_reviewed === "boolean" ? false : undefined);
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

// The dots always; head boxes and the reviewed flag only once the frame has them.
function body(f) {
  const out = { water: f.water, deck: f.deck };
  if (f.hidden) out.hidden = f.hidden;
  if (f.heads) out.heads = f.heads;
  if (typeof f.heads_reviewed === "boolean") out.heads_reviewed = f.heads_reviewed;
  return out;
}

async function save(idx) {
  const f = S.frames[idx];
  try {
    const res = await fetch(api(`/${idx}`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body(f)),
    });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    if (idx === S.idx) setSaveState("saved", "Saved ✓");
    renderList();
  } catch (e) {
    setSaveState("error", "Save failed");
    toast(`Save failed: ${e.message}`, "error");
  }
}

// ---------------------------------------------------------------- clicks + boxes

function toFrame(clientX, clientY) {
  const r = img.getBoundingClientRect();
  return [((clientX - r.left) / r.width) * S.w, ((clientY - r.top) / r.height) * S.h];
}

const r1 = (v) => Math.round(v * 10) / 10;
const inFrame = (x, y) => x >= 0 && y >= 0 && x <= S.w && y <= S.h;
const clampX = (x) => Math.min(S.w, Math.max(0, x));
const clampY = (y) => Math.min(S.h, Math.max(0, y));

function addPoint(e) {
  const [x, y] = toFrame(e.clientX, e.clientY);
  if (!inFrame(x, y)) return;
  // alt = a person whose head isn't visible here (under, blocked, or not sure): ignored in scoring
  const kind = e.altKey ? "hidden" : e.shiftKey ? "deck" : "water";
  edit((f) => (f[kind] ||= []).push([r1(x), r1(y)]));
}

function addNoHead(e) {
  const [x, y] = toFrame(e.clientX, e.clientY);
  if (!inFrame(x, y)) return;
  edit((f) => heads(f).no_head.push([r1(x), r1(y)]));
}

function finishBox(e) {
  const b = S.box;
  S.box = null;
  b.deck ||= e.shiftKey; // shift held at the start or the end of the drag
  const x1 = clampX(Math.min(b.x0, b.x1)), x2 = clampX(Math.max(b.x0, b.x1));
  const y1 = clampY(Math.min(b.y0, b.y1)), y2 = clampY(Math.max(b.y0, b.y1));
  if (x2 - x1 < 2 || y2 - y1 < 2) { drawOverlay(); return; }
  edit((f) => heads(f)[b.deck ? "deck" : "water"].push([r1(x1), r1(y1), r1(x2), r1(y2)]));
}

function removeNearest(e) {
  const [x, y] = toFrame(e.clientX, e.clientY);
  const f = cur();
  const reach = 14 * unit(); // about 14 screen pixels at any zoom
  let best = null;
  if (S.mode === "heads") {
    // the smallest box under the cursor, else the nearest "no head" mark
    for (const kind of ["water", "deck"]) {
      heads(f)[kind].forEach(([x1, y1, x2, y2], j) => {
        const area = (x2 - x1) * (y2 - y1);
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && (!best || area < best.area)) {
          best = { kind, j, area };
        }
      });
    }
    if (!best) {
      heads(f).no_head.forEach(([px, py], j) => {
        const d = Math.hypot(px - x, py - y);
        if (d <= reach && (!best || d < best.d)) best = { kind: "no_head", j, d };
      });
    }
    if (!best) { toast("No box or mark here to remove", "warn"); return; }
    edit((g) => heads(g)[best.kind].splice(best.j, 1));
    return;
  }
  for (const kind of ["water", "deck", "hidden"]) {
    (f[kind] || []).forEach(([px, py], j) => {
      const d = Math.hypot(px - x, py - y);
      if (d <= reach && (!best || d < best.d)) best = { kind, j, d };
    });
  }
  if (!best) { toast("No dot here to remove", "warn"); return; }
  edit((g) => g[best.kind].splice(best.j, 1));
}

function toggleReviewed() {
  edit((f) => { f.heads_reviewed = !f.heads_reviewed; heads(f); });
  toast(cur().heads_reviewed ? "Marked reviewed" : "Marked not reviewed");
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

// Dots mode: a click adds a dot and a drag pans. Head-box mode: a drag draws a box,
// alt-click marks "person, no head visible", and space-drag pans.
let drag = null;
stage.addEventListener("mousedown", (e) => {
  if (e.button !== 0 || S.idx < 0) return;
  if (S.mode === "heads") e.preventDefault(); // no text selection while drawing
  drag = { x: e.clientX, y: e.clientY, tx: S.zoom.tx, ty: S.zoom.ty, moved: false };
  if (S.mode === "heads" && !S.space && !e.altKey) {
    const [x, y] = toFrame(e.clientX, e.clientY);
    if (inFrame(x, y)) S.box = { x0: x, y0: y, x1: x, y1: y, deck: e.shiftKey };
  }
});
window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 4) return;
  drag.moved = true;
  if (S.box) {
    [S.box.x1, S.box.y1] = toFrame(e.clientX, e.clientY);
    drawOverlay();
  } else if (S.zoom.k > 1 && (S.mode === "dots" || S.space)) {
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
  if (S.box) {
    if (wasClick) { S.box = null; toast("Drag across a head to draw its box", "warn"); }
    else finishBox(e);
    return;
  }
  if (!wasClick || !stage.contains(e.target) || S.idx < 0) return;
  if (S.mode === "dots") addPoint(e);
  else if (e.altKey) addNoHead(e);
});
stage.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (S.idx >= 0) removeNearest(e);
});
window.addEventListener("resize", () => fitStage(false));

// ---------------------------------------------------------------- rendering

const COLORS = {
  water: "#ef4444", deck: "#3b82f6", hidden: "#f59e0b", headWater: "#22c55e", noHead: "#f59e0b",
};

function drawOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const f = cur();
  if (!f) return;
  const u = unit();
  const heading = S.mode === "heads";
  // dots: full size in dots mode, small and faint as a guide in head-box mode
  ctx.globalAlpha = heading ? 0.45 : 1;
  for (const kind of ["water", "deck"]) {
    ctx.fillStyle = COLORS[kind];
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5 * u;
    for (const [x, y] of f[kind]) {
      ctx.beginPath();
      ctx.arc(x, y, (heading ? 2.5 : 5) * u, 0, Math.PI * 2);
      ctx.fill();
      if (!heading) ctx.stroke();
    }
  }
  ctx.strokeStyle = COLORS.hidden; // hidden: hollow amber ring
  ctx.lineWidth = 2 * u;
  for (const [x, y] of f.hidden || []) {
    ctx.beginPath();
    ctx.arc(x, y, (heading ? 3 : 6) * u, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  if (!heading) return;
  const h = heads(f);
  const suggested = f.heads_source && !f.heads_reviewed; // a model's boxes, not yet checked
  ctx.lineWidth = 2 * u;
  ctx.setLineDash(suggested ? [5 * u, 3 * u] : []);
  for (const [kind, color] of [["water", COLORS.headWater], ["deck", COLORS.deck]]) {
    ctx.strokeStyle = color;
    for (const [x1, y1, x2, y2] of h[kind]) ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  ctx.setLineDash([]);
  ctx.strokeStyle = COLORS.noHead;
  for (const [x, y] of h.no_head) {
    const r = 6 * u;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.moveTo(x - r * 0.7, y - r * 0.7);
    ctx.lineTo(x + r * 0.7, y + r * 0.7);
    ctx.stroke();
  }
  if (S.box) {
    const b = S.box;
    ctx.strokeStyle = b.deck ? COLORS.deck : COLORS.headWater;
    ctx.setLineDash([4 * u, 3 * u]);
    ctx.strokeRect(Math.min(b.x0, b.x1), Math.min(b.y0, b.y1),
      Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
    ctx.setLineDash([]);
  }
}

function renderList() {
  $("#frameList").innerHTML = S.frames.map((f, i) => {
    let n, text;
    if (S.mode === "heads") {
      const h = f.heads || { water: [], deck: [], no_head: [] };
      n = h.water.length + h.deck.length + h.no_head.length;
      text = n ? `${h.water.length} + ${h.deck.length} heads${f.heads_reviewed ? " ✓" : ""}`
        : "no boxes";
    } else {
      const nh = (f.hidden || []).length;
      n = f.water.length + f.deck.length + nh;
      text = n ? `${f.water.length} + ${f.deck.length}${nh ? ` + ${nh} hidden` : ""}` : "not started";
    }
    return `<li data-i="${i}" class="${i === S.idx ? "current" : ""} ${n ? "" : "blank"}">
      <span class="mono">${f.clip_id}</span>
      <span class="muted">${text}</span>
    </li>`;
  }).join("");
}

function render() {
  const heading = S.mode === "heads";
  for (const el of document.querySelectorAll("[data-mode]")) el.hidden = el.dataset.mode !== S.mode;
  $("#modeDots").classList.toggle("active", !heading);
  $("#modeHeads").classList.toggle("active", heading);
  $("#frameSet").value = S.set;
  stage.classList.toggle("boxing", heading);
  const f = cur();
  if (!f) return;
  $("#nWater").textContent = f.water.length;
  $("#nDeck").textContent = f.deck.length;
  $("#nHidden").textContent = (f.hidden || []).length;
  const h = f.heads || { water: [], deck: [], no_head: [] };
  $("#nHeadWater").textContent = h.water.length;
  $("#nHeadDeck").textContent = h.deck.length;
  $("#nNoHead").textContent = h.no_head.length;
  $("#reviewState").textContent = f.heads_reviewed ? "Reviewed ✓"
    : f.heads_source ? `Suggested by ${f.heads_source}: check every box` : "Not reviewed";
  $("#reviewBtn").textContent = f.heads_reviewed ? "Mark not reviewed" : "Mark reviewed";
  $("#counter").textContent = `Frame ${S.idx + 1} of ${S.frames.length}`;
  $("#prev").disabled = S.idx <= 0;
  $("#next").disabled = S.idx >= S.frames.length - 1;
  renderList();
  drawOverlay();
}

function setMode(mode) {
  S.mode = mode;
  S.box = null;
  localStorage.setItem("hk.countMode", mode);
  render();
}

async function setFrameSet(set) {
  if (set === S.set) return;
  await flushSave();
  S.set = set;
  localStorage.setItem("hk.countSet", set);
  await loadFrames();
  render();
}

// ---------------------------------------------------------------- wiring

$("#prev").onclick = () => openFrame(S.idx - 1);
$("#next").onclick = () => openFrame(S.idx + 1);
$("#zoomReset").onclick = () => fitStage(true);
$("#undoBtn").onclick = undo;
$("#modeDots").onclick = () => setMode("dots");
$("#modeHeads").onclick = () => setMode("heads");
$("#reviewBtn").onclick = () => { if (S.idx >= 0) toggleReviewed(); };
$("#frameSet").onchange = (e) => setFrameSet(e.target.value);
$("#frameList").addEventListener("click", (e) => {
  const li = e.target.closest("[data-i]");
  if (li) openFrame(Number(li.dataset.i));
});

document.addEventListener("keydown", (e) => {
  if (e.target.closest("select, input, textarea")) return;
  const plain = !e.metaKey && !e.ctrlKey;
  if (e.key === " ") { e.preventDefault(); S.space = true; stage.classList.add("pan-ready"); return; }
  if (e.key.toLowerCase() === "h" && plain) { setMode(S.mode === "heads" ? "dots" : "heads"); return; }
  if (S.idx < 0) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); openFrame(S.idx - 1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); openFrame(S.idx + 1); }
  else if (e.key === "0") fitStage(true);
  else if (e.key.toLowerCase() === "r" && S.mode === "heads" && plain) toggleReviewed();
  else if (e.key.toLowerCase() === "z" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); undo(); }
});
document.addEventListener("keyup", (e) => {
  if (e.key === " ") { S.space = false; stage.classList.remove("pan-ready"); }
});

// Closing the tab with a save still waiting on its timer: send it anyway.
window.addEventListener("beforeunload", () => {
  if (!S.saveTimer) return;
  navigator.sendBeacon(api(`/${S.idx}`),
    new Blob([JSON.stringify(body(cur()))], { type: "application/json" }));
});

render();
loadFrames();
