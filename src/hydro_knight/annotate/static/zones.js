// Hydro-Knight labeler, Zones tab: one coverage polygon per clip, in original-frame pixels.
// Zoom/pan works like the Swimmer count tab. A zone saves itself once it has 3 corners
// (or the clip is marked "camera moves"); removing every corner deletes it.

const $ = (s) => document.querySelector(s);

const img = $("#frame");
const overlay = $("#overlay");
const ctx = overlay.getContext("2d");
const stage = $("#stage");
const zoomer = $("#zoomer");
const slider = $("#frameSlider");

const S = {
  clips: [],
  idx: -1,
  frame: 0,
  w: 1, h: 1,
  points: [],
  moves: false,
  history: [],       // [{idx, points, moves}] snapshots for undo
  zoom: { k: 1, tx: 0, ty: 0, bx: 0, by: 0, bw: 0, bh: 0 },
  saveTimer: null,
  imgTimer: null,
};

const cur = () => S.clips[S.idx];

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

// ---------------------------------------------------------------- clips + frames

async function loadClips() {
  const res = await fetch("/api/zones");
  S.clips = await res.json();
  if (!S.clips.length) { setSaveState("", "No clips"); return; }
  const saved = localStorage.getItem("hk.zoneClip");
  const i = S.clips.findIndex((c) => c.clip_id === saved);
  await openClip(i >= 0 ? i : 0);
}

async function openClip(i) {
  if (i < 0 || i >= S.clips.length) return;
  await flushSave();
  S.idx = i;
  const c = cur();
  localStorage.setItem("hk.zoneClip", c.clip_id);
  S.points = c.zone ? c.zone.points.map((p) => [...p]) : [];
  S.moves = c.zone ? c.zone.camera_moves : false;
  $("#clipId").textContent = c.clip_id;
  $("#clipNotes").textContent = c.notes || "";
  slider.max = Math.max(0, c.frames - 1);
  await showFrame(c.zone ? c.zone.frame : c.default_frame, true);
  setSaveState(c.zone ? "saved" : "", c.zone ? "Saved ✓" : "Not started");
  render();
}

async function showFrame(f, refit) {
  const c = cur();
  S.frame = Math.max(0, Math.min(c.frames - 1, Math.round(f)));
  slider.value = S.frame;
  const t = c.fps ? (S.frame / c.fps).toFixed(1) : "?";
  $("#frameLabel").textContent = `frame ${S.frame} · ${t} s`;
  img.src = `/api/zones/${c.clip_id}/image?frame=${S.frame}`;
  await img.decode().catch(() => toast("Couldn't load this frame", "error"));
  const sizeChanged = S.w !== img.naturalWidth || S.h !== img.naturalHeight;
  S.w = img.naturalWidth || 1;
  S.h = img.naturalHeight || 1;
  overlay.width = S.w;
  overlay.height = S.h;
  fitStage(refit || sizeChanged);
}

// Slider drags fire constantly: load the image once the hand stops.
function scrubTo(f) {
  clearTimeout(S.imgTimer);
  S.imgTimer = setTimeout(() => showFrame(f, false).then(() => scheduleSave()), 120);
}

// ---------------------------------------------------------------- edits + save

function edit(fn) {
  S.history.push({ idx: S.idx, points: S.points.map((p) => [...p]), moves: S.moves });
  if (S.history.length > 500) S.history.shift();
  fn();
  render();
  scheduleSave();
}

async function undo() {
  const prev = S.history.pop();
  if (!prev) { toast("Nothing to undo", "warn"); return; }
  if (prev.idx !== S.idx) await openClip(prev.idx);
  S.points = prev.points;
  S.moves = prev.moves;
  render();
  scheduleSave();
  toast("Undone");
}

function scheduleSave() {
  clearTimeout(S.saveTimer);
  const ready = S.moves || S.points.length >= 3 || (S.points.length === 0 && cur().zone);
  if (!ready) { setSaveState("", S.points.length ? "Need 3+ corners" : "Not started"); return; }
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

function payload() {
  if (!S.moves && S.points.length === 0) return { clear: true };
  return { points: S.points, frame: S.frame, camera_moves: S.moves };
}

async function save(idx) {
  const c = S.clips[idx];
  try {
    const res = await fetch(`/api/zones/${c.clip_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
    });
    const out = await res.json();
    if (!res.ok) throw new Error(out.error || res.statusText);
    c.zone = out.zone;
    if (idx === S.idx) setSaveState(out.zone ? "saved" : "", out.zone ? "Saved ✓" : "Cleared");
    renderList();
  } catch (e) {
    setSaveState("error", "Save failed");
    toast(`Save failed: ${e.message}`, "error");
  }
}

// ---------------------------------------------------------------- mouse

function toFrame(clientX, clientY) {
  const r = img.getBoundingClientRect();
  return [((clientX - r.left) / r.width) * S.w, ((clientY - r.top) / r.height) * S.h];
}

// Frame pixels per screen pixel at the current fit and zoom.
const unit = () => S.w / (S.zoom.bw * S.zoom.k || 1);

function nearestCorner(x, y) {
  const reach = 12 * unit();
  let best = -1, bd = Infinity;
  S.points.forEach(([px, py], j) => {
    const d = Math.hypot(px - x, py - y);
    if (d <= reach && d < bd) { best = j; bd = d; }
  });
  return best;
}

let drag = null;
stage.addEventListener("mousedown", (e) => {
  if (e.button !== 0 || S.idx < 0) return;
  const [x, y] = toFrame(e.clientX, e.clientY);
  const corner = nearestCorner(x, y);
  drag = { x: e.clientX, y: e.clientY, tx: S.zoom.tx, ty: S.zoom.ty, moved: false, corner, snap: null };
});
window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 4) return;
  if (!drag.moved && drag.corner >= 0) {  // snapshot once, so one drag = one undo step
    S.history.push({ idx: S.idx, points: S.points.map((p) => [...p]), moves: S.moves });
  }
  drag.moved = true;
  if (drag.corner >= 0) {
    const [x, y] = toFrame(e.clientX, e.clientY);
    S.points[drag.corner] = [clamp(x, S.w), clamp(y, S.h)];
    drawOverlay();
  } else if (S.zoom.k > 1) {
    stage.classList.add("panning");
    S.zoom.tx = drag.tx + dx; S.zoom.ty = drag.ty + dy;
    applyZoom();
  }
});
window.addEventListener("mouseup", (e) => {
  if (!drag) return;
  const d = drag;
  drag = null;
  stage.classList.remove("panning");
  if (d.moved && d.corner >= 0) { render(); scheduleSave(); return; }
  if (!d.moved && stage.contains(e.target) && d.corner < 0) addCorner(e);
});
stage.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (S.idx < 0) return;
  const [x, y] = toFrame(e.clientX, e.clientY);
  const j = nearestCorner(x, y);
  if (j < 0) { toast("No corner here to remove", "warn"); return; }
  edit(() => S.points.splice(j, 1));
});

const clamp = (v, max) => Math.round(Math.max(0, Math.min(max, v)) * 10) / 10;

function addCorner(e) {
  const [x, y] = toFrame(e.clientX, e.clientY);
  if (x < 0 || y < 0 || x > S.w || y > S.h) return;
  if (S.moves) { toast("Camera marked as moving: turn that off to draw a zone", "warn"); return; }
  edit(() => S.points.push([clamp(x, S.w), clamp(y, S.h)]));
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
window.addEventListener("resize", () => fitStage(false));

// ---------------------------------------------------------------- rendering

function drawOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (S.idx < 0) return;
  const u = unit();
  if (S.moves) {
    ctx.fillStyle = "rgba(15, 18, 22, 0.55)";
    ctx.fillRect(0, 0, S.w, S.h);
    ctx.fillStyle = "#fff";
    ctx.font = `${22 * u}px system-ui, sans-serif`;
    ctx.fillText("Camera moves: no fixed zone (size cutoff only)", 20 * u, 40 * u);
    return;
  }
  const pts = S.points;
  if (pts.length >= 3) {  // dim everything outside the zone
    ctx.save();
    ctx.fillStyle = "rgba(15, 18, 22, 0.5)";
    ctx.beginPath();
    ctx.rect(0, 0, S.w, S.h);
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fill("evenodd");
    ctx.restore();
  }
  if (pts.length >= 2) {
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2.5 * u;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    if (pts.length >= 3) ctx.closePath();
    ctx.stroke();
  }
  pts.forEach(([x, y], j) => {
    ctx.fillStyle = j === pts.length - 1 ? "#facc15" : "#22d3ee";
    ctx.strokeStyle = "#0f1216";
    ctx.lineWidth = 1.5 * u;
    ctx.beginPath();
    ctx.arc(x, y, 6 * u, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

function status(c) {
  if (!c.zone) return { cls: "blank", text: "not started" };
  if (c.zone.camera_moves) return { cls: "", text: "camera moves" };
  return { cls: "good", text: `${c.zone.points.length} corners` };
}

function renderList() {
  const done = S.clips.filter((c) => c.zone).length;
  $("#doneCount").textContent = `${done} / ${S.clips.length}`;
  $("#clipList").innerHTML = S.clips.map((c, i) => {
    const s = status(c);
    return `<li data-i="${i}" class="${i === S.idx ? "current" : ""} ${s.cls}">
      <span class="mono">${c.clip_id}</span><span class="muted v">${s.text}</span>
    </li>`;
  }).join("");
  const li = $("#clipList .current");
  if (li) li.scrollIntoView({ block: "nearest" });
}

function render() {
  $("#counter").textContent = `Clip ${S.idx + 1} of ${S.clips.length}`;
  $("#prev").disabled = S.idx <= 0;
  $("#next").disabled = S.idx >= S.clips.length - 1;
  $("#movesBtn").classList.toggle("active", S.moves);
  renderList();
  drawOverlay();
}

// ---------------------------------------------------------------- wiring

const step = (sec) => scrubTo(S.frame + sec * (cur().fps || 30));

$("#prev").onclick = () => openClip(S.idx - 1);
$("#next").onclick = () => openClip(S.idx + 1);
$("#back1s").onclick = () => step(-1);
$("#fwd1s").onclick = () => step(1);
slider.addEventListener("input", () => scrubTo(Number(slider.value)));
$("#movesBtn").onclick = () => edit(() => { S.moves = !S.moves; });
$("#clearBtn").onclick = () => edit(() => { S.points = []; S.moves = false; });
$("#undoBtn").onclick = undo;
$("#zoomReset").onclick = () => fitStage(true);
$("#clipList").addEventListener("click", (e) => {
  const li = e.target.closest("[data-i]");
  if (li) openClip(Number(li.dataset.i));
});

document.addEventListener("keydown", (e) => {
  if (S.idx < 0 || e.target === slider) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); openClip(S.idx - 1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); openClip(S.idx + 1); }
  else if (e.key === ",") step(-1);
  else if (e.key === ".") step(1);
  else if (e.key === "0") fitStage(true);
  else if (e.key === "Backspace" && S.points.length) { e.preventDefault(); edit(() => S.points.pop()); }
  else if (e.key.toLowerCase() === "z" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); undo(); }
});

// Closing the tab with a save still waiting on its timer: send it anyway.
window.addEventListener("beforeunload", () => {
  if (!S.saveTimer) return;
  navigator.sendBeacon(`/api/zones/${cur().clip_id}`,
    new Blob([JSON.stringify(payload())], { type: "application/json" }));
});

loadClips();
