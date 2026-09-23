// Hydro-Knight labeler — single-page UI talking to annotate/server.py.
// Times inside the UI are LOCAL FILE seconds; the server stores SOURCE seconds
// (file + offset). Victim marks are pixel positions in the original frame.

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const video = $("#video");
const overlay = $("#overlay");
const ctx = overlay.getContext("2d");
const stage = $("#stage");
const zoomer = $("#zoomer");

const S = {
  filter: localStorage.getItem("hk.filter") || "needs_victim",
  clips: [],
  idx: -1,
  clip: null,        // server payload for the open clip
  fps: 30, offset: 0, vw: 1, vh: 1, dur: 0,
  label: null,
  trimStart: null,   // file seconds, null = start of file
  trimEnd: null,     // file seconds, null = end of file
  ev: null,          // {start, end, contact, victim:[{t,x,y,kind}]} in file seconds, or null
  extraEvents: [],   // any further events, passed through untouched
  history: [],
  armed: null,       // "onset" | "last_seen" | "extra" | null
  zoom: { k: 1, tx: 0, ty: 0, bx: 0, by: 0, bw: 0, bh: 0 },
  showBoxes: false,
  boxCache: new Map(),
  boxFetching: false,
  boxesAvailable: false,
  saveTimer: null,
  lastFrame: -1,
};

// ---------------------------------------------------------------- feedback

function toast(msg, kind = "ok") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 2600);
}

function flash(btn) {
  if (!btn) return;
  btn.classList.add("flash");
  setTimeout(() => btn.classList.remove("flash"), 350);
}

function setSaveState(state, text) {
  const pill = $("#saveState");
  pill.className = `pill ${state}`;
  pill.textContent = text;
}

const fmt = (t) => {
  if (t == null || Number.isNaN(t)) return "—";
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(2).padStart(5, "0");
  return `${m}:${s}`;
};

// ---------------------------------------------------------------- queue

async function loadQueue(keepClip) {
  $("#filter").value = S.filter;
  const res = await fetch(`/api/clips?filter=${S.filter}`);
  S.clips = await res.json();
  $("#emptyState").hidden = S.clips.length > 0;
  if (!S.clips.length) {
    S.idx = -1;
    $("#counter").textContent = "0 clips";
    setSaveState("", "—");
    return;
  }
  const want = keepClip || localStorage.getItem("hk.clip");
  const found = S.clips.findIndex((c) => c.clip_id === want);
  await openClip(found >= 0 ? found : 0);
}

function updateCounter() {
  const done = S.clips.filter((c) => c.done).length;
  $("#counter").textContent = `Clip ${S.idx + 1} of ${S.clips.length} · ${done} done`;
  $("#prev").disabled = S.idx <= 0;
  $("#next").disabled = S.idx >= S.clips.length - 1;
}

async function openClip(i) {
  if (i < 0 || i >= S.clips.length) return;
  if (S.saveTimer) { clearTimeout(S.saveTimer); S.saveTimer = null; await save(); } // flush pending edit
  S.idx = i;
  const id = S.clips[i].clip_id;
  localStorage.setItem("hk.clip", id);
  const res = await fetch(`/api/clip/${id}`);
  const c = await res.json();
  S.clip = c;
  S.fps = c.video.fps || 30;
  S.offset = c.offset || 0;
  S.vw = c.video.width || 1;
  S.vh = c.video.height || 1;
  S.dur = c.video.duration || 0;
  S.label = c.label;
  S.trimStart = c.start_sec > 0 ? c.start_sec - S.offset : null;
  S.trimEnd = c.end_sec > 0 ? c.end_sec - S.offset : null;
  const evs = (c.events || []).map((e) => ({
    start: e.start - S.offset,
    end: e.end - S.offset,
    contact: e.contact == null ? null : e.contact - S.offset,
    label: e.label || "distress",
    victim: (e.victim || []).map((m) => ({ ...m, t: m.t - S.offset })),
  }));
  S.ev = evs[0] || null;
  S.extraEvents = evs.slice(1);
  S.history = [];
  S.boxCache.clear();
  S.boxesAvailable = c.has_detections;
  S.lastFrame = -1;
  disarm();

  $("#clipId").textContent = c.clip_id;
  $("#clipNotes").textContent = c.notes || "";
  $("#boxesBtn").disabled = !c.has_detections;
  $("#boxesBtn").title = c.has_detections
    ? "Show what the detector found on this frame"
    : "No detections extracted for this clip yet";
  overlay.width = S.vw;
  overlay.height = S.vh;

  video.src = `/video/${id}`;
  await new Promise((r) => video.addEventListener("loadedmetadata", r, { once: true }));
  video.pause();
  fitStage(true);
  const startAt = S.ev ? S.ev.start : S.trimStart || 0;
  seek(startAt);
  setPlaying(false);
  setSaveState("saved", "Saved ✓");
  render();
  updateCounter();
}

// ---------------------------------------------------------------- playback

const frameNow = () => Math.floor(video.currentTime * S.fps + 1e-4);

function seek(t) {
  video.currentTime = Math.max(0, Math.min(S.dur - 0.5 / S.fps, t));
}

function seekFrame(f) {
  seek((f + 0.5) / S.fps); // land in the middle of the frame so the browser shows it
}

function setPlaying(on) {
  if (on) video.play(); else video.pause();
  $("#playBtn").innerHTML = on ? "⏸ Pause <kbd>Space</kbd>" : "▶ Play <kbd>Space</kbd>";
}

function setSpeed(r) {
  video.playbackRate = r;
  $$("[data-speed]").forEach((b) => b.classList.toggle("active", Number(b.dataset.speed) === r));
}

const transport = {
  play: () => setPlaying(video.paused),
  backFrame: () => { setPlaying(false); seekFrame(frameNow() - 1); },
  fwdFrame: () => { setPlaying(false); seekFrame(frameNow() + 1); },
  back1s: () => seek(video.currentTime - 1),
  fwd1s: () => seek(video.currentTime + 1),
};

// ---------------------------------------------------------------- edits + save

function snapshot() {
  return JSON.stringify({ label: S.label, trimStart: S.trimStart, trimEnd: S.trimEnd, ev: S.ev });
}

function edit(fn, msg) {
  S.history.push(snapshot());
  if (S.history.length > 200) S.history.shift();
  fn();
  render();
  scheduleSave();
  if (msg) toast(msg);
}

function undo() {
  const prev = S.history.pop();
  if (!prev) { toast("Nothing to undo", "warn"); return; }
  Object.assign(S, JSON.parse(prev));
  render();
  scheduleSave();
  toast("Undone");
}

function ensureEvent() {
  if (!S.ev) S.ev = { start: null, end: null, contact: null, label: "distress", victim: [] };
  return S.ev;
}

function eventPayload() {
  const out = [];
  const conv = (e) => ({
    start: e.start + S.offset,
    end: e.end + S.offset,
    label: e.label || "distress",
    contact: e.contact == null ? null : e.contact + S.offset,
    victim: (e.victim || []).map((m) => ({ ...m, t: m.t + S.offset })),
  });
  if (S.ev && S.ev.start != null && S.ev.end != null) out.push(conv(S.ev));
  S.extraEvents.forEach((e) => out.push(conv(e)));
  return out;
}

function scheduleSave() {
  clearTimeout(S.saveTimer);
  setSaveState("saving", "Saving…");
  S.saveTimer = setTimeout(() => { S.saveTimer = null; save(); }, 400);
}

async function save() {
  const c = S.clip;
  if (!c) return;
  // A sectioned download can't be widened past its section, so "cleared" trim keeps its bounds.
  const body = {
    label: S.label,
    start_sec: S.trimStart == null ? S.offset : S.trimStart + S.offset,
    end_sec: S.trimEnd != null ? S.trimEnd + S.offset : S.offset > 0 ? c.end_sec : -1,
    events: eventPayload(),
  };
  try {
    const res = await fetch(`/api/clip/${c.clip_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const pending = S.ev && (S.ev.start == null) !== (S.ev.end == null);
    setSaveState(pending ? "saving" : "saved", pending ? "Needs onset + saved" : "Saved ✓");
    S.clips[S.idx].done = isDone();
    S.clips[S.idx].label = S.label;
    updateCounter();
  } catch (e) {
    setSaveState("error", "Save failed");
    toast(`Save failed: ${e.message}`, "error");
  }
}

function isDone() {
  if (S.label === "normal") return true;
  if (S.label !== "distress") return false;
  const e = S.ev;
  if (!e || e.start == null || e.end == null || e.contact == null) return false;
  const kinds = new Set((e.victim || []).map((m) => m.kind));
  return kinds.has("onset") && kinds.has("last_seen");
}

// ---------------------------------------------------------------- field rows

const FIELD_NAMES = {
  start: "Drowning onset", contact: "Guard contact", end: "Victim saved",
  trimStart: "Clip start", trimEnd: "Clip end",
};

function getField(f) {
  if (f === "trimStart") return S.trimStart;
  if (f === "trimEnd") return S.trimEnd;
  return S.ev ? S.ev[f] : null;
}

function setField(f, v) {
  if (f === "trimStart") S.trimStart = v;
  else if (f === "trimEnd") S.trimEnd = v;
  else ensureEvent()[f] = v;
}

function onRowAction(row, action) {
  const f = row.dataset.field;
  const t = video.currentTime;
  if (action === "set") {
    edit(() => setField(f, t), `${FIELD_NAMES[f]} set to ${fmt(t)}`);
    if (["start", "contact", "end"].includes(f) && S.label !== "distress") {
      edit(() => { S.label = "distress"; }, "Clip labeled Distress");
    }
    checkOrder();
  } else if (action === "go") {
    const v = getField(f);
    if (v == null) { toast(`${FIELD_NAMES[f]} isn't set yet`, "warn"); return; }
    setPlaying(false);
    seek(v);
  } else if (action === "clear") {
    edit(() => setField(f, null), `${FIELD_NAMES[f]} cleared`);
  }
}

function checkOrder() {
  const e = S.ev;
  const warn = $("#eventWarn");
  const msgs = [];
  if (e) {
    if (e.start != null && e.end != null && e.end < e.start) msgs.push("Saved is before onset.");
    if (e.contact != null && e.start != null && e.contact < e.start) msgs.push("Guard contact is before onset.");
    if (e.contact != null && e.end != null && e.contact > e.end) msgs.push("Guard contact is after the save.");
    if ((e.start == null) !== (e.end == null)) msgs.push("Set both onset and saved; the event is saved once both exist.");
  }
  warn.hidden = msgs.length === 0;
  warn.textContent = msgs.join(" ");
}

// ---------------------------------------------------------------- victim marks

function arm(kind) {
  if (S.label !== "distress") { toast("Label the clip Distress first", "warn"); return; }
  S.armed = kind;
  setPlaying(false);
  if (kind === "onset" && S.ev && S.ev.start != null) seek(S.ev.start);
  stage.classList.add("armed");
  $("#armBanner").hidden = false;
  $("#armText").textContent = {
    onset: "Click on the victim (at onset)",
    last_seen: "Click where the victim was last visible",
    extra: "Click on the victim",
  }[kind];
  $$("[data-arm]").forEach((b) => b.classList.toggle("armed", b.dataset.arm === kind));
}

function disarm() {
  S.armed = null;
  stage.classList.remove("armed");
  $("#armBanner").hidden = true;
  $$("[data-arm]").forEach((b) => b.classList.remove("armed"));
}

function placeMark(clientX, clientY) {
  const r = video.getBoundingClientRect();
  const x = ((clientX - r.left) / r.width) * S.vw;
  const y = ((clientY - r.top) / r.height) * S.vh;
  if (x < 0 || y < 0 || x > S.vw || y > S.vh) return;
  const kind = S.armed;
  const t = video.currentTime;
  const names = { onset: "Victim at onset", last_seen: "Last visible", extra: "Extra point" };
  edit(() => {
    const e = ensureEvent();
    if (kind !== "extra") e.victim = e.victim.filter((m) => m.kind !== kind); // one of each
    e.victim.push({ t, x: Math.round(x * 10) / 10, y: Math.round(y * 10) / 10, kind });
    e.victim.sort((a, b) => a.t - b.t);
  }, `${names[kind]} marked at ${fmt(t)}`);
  if (kind === "onset" && S.ev.start != null && Math.abs(t - S.ev.start) > 0.5) {
    toast("Heads up: this onset click isn't at the onset time", "warn");
  }
  disarm();
}

function removeMark(i) {
  edit(() => S.ev.victim.splice(i, 1), "Mark removed");
}

// ---------------------------------------------------------------- zoom + pan

function fitStage(reset) {
  const sw = stage.clientWidth, sh = stage.clientHeight;
  const scale = Math.min(sw / S.vw, sh / S.vh);
  const z = S.zoom;
  z.bw = S.vw * scale; z.bh = S.vh * scale;
  z.bx = (sw - z.bw) / 2; z.by = (sh - z.bh) / 2;
  if (reset) { z.k = 1; z.tx = 0; z.ty = 0; }
  applyZoom();
}

function clampPan() {
  const z = S.zoom;
  const minX = z.bw - z.bw * z.k, minY = z.bh - z.bh * z.k;
  z.tx = Math.min(0, Math.max(minX, z.tx));
  z.ty = Math.min(0, Math.max(minY, z.ty));
}

function applyZoom() {
  const z = S.zoom;
  clampPan();
  Object.assign(zoomer.style, {
    left: `${z.bx}px`, top: `${z.by}px`, width: `${z.bw}px`, height: `${z.bh}px`,
    transform: `translate(${z.tx}px, ${z.ty}px) scale(${z.k})`,
  });
  $("#zoomBadge").hidden = z.k <= 1.001;
  $("#zoomBadge").textContent = `${z.k.toFixed(1)}×`;
}

stage.addEventListener("wheel", (e) => {
  e.preventDefault();
  const z = S.zoom;
  const rect = stage.getBoundingClientRect();
  const px = e.clientX - rect.left - z.bx, py = e.clientY - rect.top - z.by;
  const lx = (px - z.tx) / z.k, ly = (py - z.ty) / z.k; // point under cursor, unzoomed
  const k = Math.min(10, Math.max(1, z.k * Math.exp(-e.deltaY * 0.0015)));
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
  if (!wasClick || !stage.contains(e.target) || e.target.closest("button")) return;
  if (S.armed) placeMark(e.clientX, e.clientY);
  else setPlaying(video.paused); // plain click on the video toggles play, like any player
});
window.addEventListener("resize", () => fitStage(false));

// ---------------------------------------------------------------- detector boxes

async function fetchBoxes(frame) {
  if (S.boxCache.has(frame) || S.boxFetching) return;
  S.boxFetching = true;
  const first = S.boxCache.size === 0;
  if (first) toast("Loading detector boxes for this clip…", "warn");
  try {
    const res = await fetch(`/api/boxes/${S.clip.clip_id}?frame=${frame}`);
    const data = await res.json();
    S.boxCache.set(frame, data.boxes || []);
    if (frame === frameNow()) drawOverlay(); // paused video: nothing else would trigger a redraw
  } finally {
    S.boxFetching = false;
  }
}

// ---------------------------------------------------------------- rendering

function drawOverlay() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const k = S.zoom.k;
  const unit = Math.max(S.vh / 540, 1) / k; // ~1 display px at this zoom
  const now = video.currentTime;

  if (S.showBoxes) {
    const boxes = S.boxCache.get(frameNow()) || [];
    ctx.lineWidth = 1.5 * unit;
    ctx.font = `${11 * unit}px ui-monospace, Menlo, monospace`;
    for (const b of boxes) {
      const [x1, y1, x2, y2] = b.box;
      ctx.strokeStyle = "rgba(96,165,250,0.95)";
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = "rgba(96,165,250,0.95)";
      ctx.fillText(b.conf.toFixed(2), x1, y1 - 3 * unit);
    }
  }

  const colors = { onset: "#ffffff", last_seen: "#facc15", extra: "#a5b4fc" };
  for (const m of (S.ev && S.ev.victim) || []) {
    const near = Math.abs(m.t - now) < 0.35;
    ctx.globalAlpha = near ? 1 : 0.35;
    ctx.strokeStyle = colors[m.kind];
    ctx.lineWidth = 2.5 * unit;
    const r = 14 * unit;
    ctx.beginPath();
    ctx.arc(m.x, m.y, r, 0, Math.PI * 2);
    ctx.moveTo(m.x - r * 1.6, m.y); ctx.lineTo(m.x - r * 0.5, m.y);
    ctx.moveTo(m.x + r * 0.5, m.y); ctx.lineTo(m.x + r * 1.6, m.y);
    ctx.moveTo(m.x, m.y - r * 1.6); ctx.lineTo(m.x, m.y - r * 0.5);
    ctx.moveTo(m.x, m.y + r * 0.5); ctx.lineTo(m.x, m.y + r * 1.6);
    ctx.stroke();
    if (near) {
      ctx.fillStyle = colors[m.kind];
      ctx.font = `600 ${12 * unit}px -apple-system, sans-serif`;
      ctx.fillText({ onset: "onset", last_seen: "last visible", extra: "victim" }[m.kind], m.x + r * 1.8, m.y - r);
    }
  }
  ctx.globalAlpha = 1;
}

function pct(t) { return `${(100 * t) / (S.dur || 1)}%`; }

function drawTimeline() {
  const segs = [];
  const add = (a, b, cls) => { if (a != null && b != null && b > a) segs.push([a, b, cls]); };
  const t0 = S.trimStart ?? 0, t1 = S.trimEnd ?? S.dur;
  add(0, t0, "trimmed");
  add(t1, S.dur, "trimmed");
  const e = S.ev;
  if (S.label === "distress" && e && e.start != null) {
    const end = e.end ?? t1;
    const contact = e.contact;
    add(t0, e.start, "normal");
    if (contact != null && contact > e.start && contact < end) {
      add(e.start, contact, "distress");
      add(contact, end, "rescue");
    } else add(e.start, end, "distress");
    add(end, t1, "after");
  } else if (S.label === "normal") add(t0, t1, "normal");

  $("#tlSegments").innerHTML = segs
    .map(([a, b, c]) => `<div class="tl-seg ${c}" style="left:${pct(a)};width:${pct(b - a)}"></div>`)
    .join("");
  $("#tlMarks").innerHTML = ((e && e.victim) || [])
    .map((m, i) => `<div class="tl-dot ${m.kind}" data-mark="${i}" style="left:${pct(m.t)}" title="${m.kind} ${fmt(m.t)}"></div>`)
    .join("");
}

function renderRows() {
  $$(".row[data-field]").forEach((row) => {
    const v = getField(row.dataset.field);
    const el = row.querySelector("[data-val]");
    const isTrim = row.dataset.field.startsWith("trim");
    el.textContent = v == null ? (isTrim ? (row.dataset.field === "trimStart" ? "start" : "end") : "—") : fmt(v);
    el.classList.toggle("set", v != null);
  });
}

function renderMarks() {
  const list = $("#markList");
  const marks = (S.ev && S.ev.victim) || [];
  if (!marks.length) {
    list.innerHTML = `<li class="empty">No victim marks yet.</li>`;
    return;
  }
  const names = { onset: "At onset", last_seen: "Last visible", extra: "Extra point" };
  list.innerHTML = marks.map((m, i) => `
    <li>
      <span class="dot ${m.kind}"></span>
      <span>${names[m.kind]} <span class="muted mono">${fmt(m.t)}</span></span>
      <button class="btn small ghost" data-jump="${i}">Go</button>
      <button class="btn small ghost danger" data-remove="${i}" title="Remove">✕</button>
    </li>`).join("");
}

function renderChecklist() {
  const set = (id, ok, todo) => {
    const el = $(id);
    el.className = `chk ${ok ? "done" : "todo"}`;
    el.textContent = ok ? "✓ done" : todo;
  };
  const distress = S.label === "distress";
  set("#chkLabel", S.label === "distress" || S.label === "normal", "choose one");
  const e = S.ev;
  if (!distress) {
    $("#chkTimes").className = "chk"; $("#chkTimes").textContent = "not needed";
    $("#chkVictim").className = "chk"; $("#chkVictim").textContent = "not needed";
  } else {
    const missing = ["start", "contact", "end"].filter((f) => !e || e[f] == null).map((f) => FIELD_NAMES[f].toLowerCase());
    set("#chkTimes", missing.length === 0, `missing ${missing.join(", ")}`);
    const kinds = new Set(((e && e.victim) || []).map((m) => m.kind));
    const need = [["onset", "onset"], ["last_seen", "last visible"]].filter(([k]) => !kinds.has(k)).map(([, n]) => n);
    set("#chkVictim", need.length === 0, `missing ${need.join(", ")}`);
  }
  $("#eventCard").style.opacity = distress ? 1 : 0.55;
  $("#victimCard").style.opacity = distress ? 1 : 0.55;
  $$("[data-label]").forEach((b) => b.classList.toggle("active", b.dataset.label === S.label));
}

function render() {
  renderRows();
  renderMarks();
  renderChecklist();
  drawTimeline();
  drawOverlay();
  checkOrder();
}

function tick() {
  if (S.clip) {
    const f = frameNow();
    $("#clock").textContent = `${fmt(video.currentTime)} · frame ${f}`;
    $("#tlHead").style.left = pct(video.currentTime);
    if (f !== S.lastFrame) {
      S.lastFrame = f;
      if (S.showBoxes) fetchBoxes(f);
      drawOverlay();
    } else if (S.showBoxes && !S.boxCache.has(f)) {
      fetchBoxes(f);
      drawOverlay();
    }
    if (!video.paused !== $("#playBtn").textContent.includes("Pause")) setPlaying(!video.paused);
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- wiring

document.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn) flash(btn);
  if (!btn) return;
  if (btn.dataset.act) transport[btn.dataset.act]();
  else if (btn.dataset.speed) setSpeed(Number(btn.dataset.speed));
  else if (btn.dataset.label) {
    const lbl = btn.dataset.label;
    edit(() => { S.label = lbl; }, `Labeled ${lbl === "distress" ? "Distress" : "Normal"}`);
  } else if (btn.dataset.arm) {
    if (S.armed === btn.dataset.arm) disarm(); else arm(btn.dataset.arm);
  } else if (btn.dataset.jump != null) { setPlaying(false); seek(S.ev.victim[Number(btn.dataset.jump)].t); }
  else if (btn.dataset.remove != null) removeMark(Number(btn.dataset.remove));
  else if ("set" in btn.dataset) onRowAction(btn.closest(".row"), "set");
  else if ("go" in btn.dataset) onRowAction(btn.closest(".row"), "go");
  else if ("clear" in btn.dataset) onRowAction(btn.closest(".row"), "clear");
});

$("#timeline").addEventListener("click", (e) => {
  const dot = e.target.closest("[data-mark]");
  if (dot) { setPlaying(false); seek(S.ev.victim[Number(dot.dataset.mark)].t); return; }
  const r = $("#timeline").getBoundingClientRect();
  seek(((e.clientX - r.left) / r.width) * S.dur);
});

$("#prev").onclick = () => openClip(S.idx - 1);
$("#next").onclick = () => openClip(S.idx + 1);
$("#nextTodo").onclick = () => {
  for (let j = 1; j <= S.clips.length; j++) {
    const i = (S.idx + j) % S.clips.length;
    if (!S.clips[i].done) { openClip(i); return; }
  }
  toast("Every clip in this list is done 🎉");
};
$("#filter").onchange = (e) => {
  S.filter = e.target.value;
  localStorage.setItem("hk.filter", S.filter);
  loadQueue(S.clip && S.clip.clip_id);
};
$("#armCancel").onclick = disarm;
$("#zoomReset").onclick = () => { fitStage(true); toast("Zoom reset"); };
$("#undoBtn").onclick = undo;
$("#boxesBtn").onclick = () => {
  S.showBoxes = !S.showBoxes;
  $("#boxesBtn").classList.toggle("active", S.showBoxes);
  toast(S.showBoxes ? "Showing detector boxes" : "Detector boxes hidden");
  drawOverlay();
};
$("#deleteBtn").onclick = () => { $("#modal").hidden = false; };
$("#modalCancel").onclick = () => { $("#modal").hidden = true; };
$("#modalConfirm").onclick = async () => {
  $("#modal").hidden = true;
  const id = S.clip.clip_id;
  const res = await fetch(`/api/clip/${id}/delete`, { method: "POST" });
  if (!res.ok) { toast("Delete failed", "error"); return; }
  toast(`Deleted ${id} and blocked its URL`, "warn");
  S.clips.splice(S.idx, 1);
  if (S.clips.length) openClip(Math.min(S.idx, S.clips.length - 1)); else loadQueue();
};

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || !$("#modal").hidden) return;
  const pressBtn = (sel) => flash(document.querySelector(sel));
  if (e.key === " ") { e.preventDefault(); transport.play(); pressBtn("#playBtn"); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); e.shiftKey ? transport.back1s() : transport.backFrame(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); e.shiftKey ? transport.fwd1s() : transport.fwdFrame(); }
  else if (e.key === "Escape") disarm();
  else if (e.key.toLowerCase() === "z" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); undo(); pressBtn("#undoBtn"); }
  else if (e.key.toLowerCase() === "b" && !$("#boxesBtn").disabled) $("#boxesBtn").click();
});

// Closing the tab with an edit still waiting on the 400 ms save timer: send it anyway.
window.addEventListener("beforeunload", () => {
  if (!S.saveTimer || !S.clip) return;
  const c = S.clip;
  const body = {
    label: S.label,
    start_sec: S.trimStart == null ? S.offset : S.trimStart + S.offset,
    end_sec: S.trimEnd != null ? S.trimEnd + S.offset : S.offset > 0 ? c.end_sec : -1,
    events: eventPayload(),
  };
  navigator.sendBeacon(`/api/clip/${c.clip_id}`, new Blob([JSON.stringify(body)], { type: "application/json" }));
});

loadQueue().then(() => requestAnimationFrame(tick));
