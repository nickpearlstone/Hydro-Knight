// Hydro-Knight labeler, Review tab: play each Claude-labeled clip around its event with
// the victim marked, and record a good / not good verdict (annotate/review.py).
// Times inside the UI are LOCAL FILE seconds; the server stores SOURCE seconds
// (file + offset). Victim clicks are pixel positions in the original frame.

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const video = $("#video");
const overlay = $("#overlay");
const ctx = overlay.getContext("2d");
const stage = $("#stage");
const zoomer = $("#zoomer");

const LEAD = 2.0; // seconds shown before onset
const TAIL = 1.5; // seconds shown after the save
const ZOOM = 3;
const MARKS = [
  ["start", "Onset"],
  ["lastSeen", "Last seen"],
  ["contact", "Contact"],
  ["end", "Saved"],
];

const R = {
  rows: [],
  idx: -1,
  fps: 30, offset: 0, vw: 1, vh: 1, dur: 0,
  ev: null,        // {start, lastSeen, contact, end, offCamera} in file seconds
  clicks: [],      // [{t, x, y, kind}] in file seconds, sorted
  a: 0, b: 0,      // loop window
  zoom: false,
  prevT: 0,
  box: { x: 0, y: 0, w: 1, h: 1 }, // unzoomed video rectangle inside the stage
  flash: null,     // {text, until}
};

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

const fmt = (t) => {
  if (t == null || Number.isNaN(t)) return "—";
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(2).padStart(5, "0");
  return `${m}:${s}`;
};

// ---------------------------------------------------------------- queue

async function loadQueue() {
  const res = await fetch("/api/review");
  R.rows = await res.json();
  $("#emptyState").hidden = R.rows.length > 0;
  renderList();
  if (!R.rows.length) {
    $("#counter").textContent = "0 clips";
    setSaveState("", "—");
    return;
  }
  const want = localStorage.getItem("hk.review.clip");
  let i = R.rows.findIndex((r) => r.clip_id === want);
  if (i < 0) i = R.rows.findIndex((r) => !r.verdict);
  await openClip(i >= 0 ? i : 0);
}

function updateCounter() {
  const done = R.rows.filter((r) => r.verdict).length;
  $("#counter").textContent = `Clip ${R.idx + 1} of ${R.rows.length} · ${done} reviewed`;
  $("#prev").disabled = R.idx <= 0;
  $("#next").disabled = R.idx >= R.rows.length - 1;
  $("#nextTodo").disabled = !R.rows.some((r, i) => !r.verdict && i !== R.idx);
}

function renderList() {
  const ul = $("#clipList");
  ul.innerHTML = "";
  R.rows.forEach((r, i) => {
    const li = document.createElement("li");
    li.className = `${i === R.idx ? "current" : ""} ${r.verdict || "todo"}`;
    const mark = { good: "✓ good", not_good: "✗ not good" }[r.verdict] || "• to review";
    li.innerHTML = `<span class="mono">${r.clip_id}</span><span class="v">${mark}</span>`;
    li.onclick = () => openClip(i);
    ul.appendChild(li);
  });
}

function renderVerdict() {
  const r = R.rows[R.idx];
  $("#goodBtn").classList.toggle("active", r.verdict === "good");
  $("#badBtn").classList.toggle("active", r.verdict === "not_good");
  $("#clearBtn").disabled = !r.verdict;
  if (document.activeElement !== $("#note")) $("#note").value = r.note || "";
  setSaveState(
    r.verdict ? "saved" : "",
    { good: "Reviewed: good", not_good: "Reviewed: not good" }[r.verdict] || "Not reviewed",
  );
}

async function openClip(i) {
  if (i < 0 || i >= R.rows.length) return;
  R.idx = i;
  const row = R.rows[i];
  localStorage.setItem("hk.review.clip", row.clip_id);
  const c = await (await fetch(`/api/clip/${row.clip_id}`)).json();
  R.fps = c.video.fps || 30;
  R.offset = c.offset || 0;
  R.vw = c.video.width || 1;
  R.vh = c.video.height || 1;
  R.dur = c.video.duration || 0;
  const f = (t) => (t == null ? null : t - R.offset);
  const e = (c.events || [])[0] || {};
  R.clicks = (e.victim || []).map((m) => ({ ...m, t: m.t - R.offset })).sort((p, q) => p.t - q.t);
  const lastSeen = R.clicks.find((m) => m.kind === "last_seen");
  R.ev = {
    start: f(e.start),
    lastSeen: lastSeen ? lastSeen.t : null,
    contact: f(e.contact),
    end: f(e.end),
    offCamera: !!e.saved_off_camera,
  };
  const first = R.ev.start ?? (R.clicks[0] || {}).t ?? 0;
  const last = R.ev.offCamera ? R.dur : (R.ev.end ?? R.ev.contact ?? first + 10) + TAIL;
  R.a = Math.max(0, first - LEAD);
  R.b = Math.min(R.dur || last, last);
  R.prevT = R.a;
  R.flash = null;

  $("#clipId").textContent = row.clip_id;
  $("#claudeNotes").textContent = row.claude_notes || "(no notes)";
  $("#openRescue").href = `/?clip=${row.clip_id}`;
  $("#openRescue").target = "_blank";
  $("#note").value = row.note || "";
  overlay.width = R.vw;
  overlay.height = R.vh;
  renderTimes();
  renderTimeline();
  renderVerdict();
  renderList();
  updateCounter();
  fitStage();

  video.src = `/video/${row.clip_id}`;
  video.addEventListener(
    "loadedmetadata",
    () => {
      video.currentTime = R.a;
      video.play().catch(() => {});
    },
    { once: true },
  );
}

// ---------------------------------------------------------------- verdicts

async function sendVerdict(verdict) {
  const row = R.rows[R.idx];
  const note = $("#note").value;
  setSaveState("saving", "Saving…");
  const res = await fetch(`/api/review/${row.clip_id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ verdict, note }),
  });
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))).error || res.statusText;
    setSaveState("error", "Not saved");
    toast(`Not saved: ${err}`, "error");
    return false;
  }
  row.verdict = verdict;
  row.note = verdict ? note.trim() : "";
  if (!verdict) $("#note").value = "";
  renderVerdict();
  renderList();
  updateCounter();
  return true;
}

async function markGood() {
  if (!(await sendVerdict("good"))) return;
  toast("Marked good");
  nextUnreviewed();
}

async function markBad() {
  if (!(await sendVerdict("not_good"))) return;
  if (!$("#note").value.trim()) {
    toast("Marked not good. Add a note on what's wrong if you can; it saves when you leave the box.", "warn");
    $("#note").focus();
  } else {
    toast("Marked not good");
  }
}

function nextUnreviewed() {
  const n = R.rows.length;
  for (let k = 1; k <= n; k++) {
    const i = (R.idx + k) % n;
    if (!R.rows[i].verdict) return openClip(i);
  }
  toast("All clips reviewed");
}

// ---------------------------------------------------------------- panel

function renderTimes() {
  const box = $("#times");
  box.innerHTML = "";
  for (const [key, name] of MARKS) {
    const t = R.ev[key];
    const row = document.createElement("button");
    row.className = "time-row";
    const val = t == null ? (key === "end" && R.ev.offCamera ? "off camera" : "—") : fmt(t);
    row.innerHTML = `<i class="dot ${key}"></i><span>${name}</span><span class="mono">${val}</span>`;
    row.disabled = t == null;
    row.onclick = () => {
      video.pause();
      video.currentTime = t;
    };
    box.appendChild(row);
  }
}

function renderTimeline() {
  const span = Math.max(R.b - R.a, 0.001);
  const pct = (t) => `${(100 * (t - R.a)) / span}%`;
  const seg = $("#tlSegments");
  seg.innerHTML = "";
  const { start, contact, end } = R.ev;
  const cuts = [
    ["normal", R.a, start ?? R.a],
    ["distress", start ?? R.a, contact ?? R.b],
    ["rescue", contact ?? R.b, end ?? R.b],
    ["after", end ?? R.b, R.b],
  ];
  for (const [cls, t0, t1] of cuts) {
    if (t1 <= t0) continue;
    const d = document.createElement("div");
    d.className = `tl-seg ${cls}`;
    d.style.left = pct(t0);
    d.style.width = `${(100 * (t1 - t0)) / span}%`;
    seg.appendChild(d);
  }
  const marks = $("#tlMarks");
  marks.innerHTML = "";
  for (const m of R.clicks) {
    const d = document.createElement("div");
    d.className = `tl-dot ${m.kind}`;
    d.style.left = pct(m.t);
    d.title = `${m.kind} ${fmt(m.t)}`;
    marks.appendChild(d);
  }
}

// ---------------------------------------------------------------- stage

function fitStage() {
  const w = stage.clientWidth, h = stage.clientHeight;
  const k = Math.min(w / R.vw, h / R.vh);
  const bw = R.vw * k, bh = R.vh * k;
  R.box = { x: (w - bw) / 2, y: (h - bh) / 2, w: bw, h: bh };
  Object.assign(zoomer.style, {
    width: `${bw}px`, height: `${bh}px`, left: `${R.box.x}px`, top: `${R.box.y}px`,
  });
}

// Where the victim marker sits at time t: on a click, between two clicks (linear), or
// held at the first / last click outside them. style: "exact" | "between" | "held".
function victimAt(t) {
  const c = R.clicks;
  if (!c.length) return null;
  const near = 0.6 / R.fps + 0.1;
  if (t <= c[0].t) return { x: c[0].x, y: c[0].y, style: Math.abs(t - c[0].t) < near ? "exact" : "held", kind: c[0].kind };
  const z = c[c.length - 1];
  if (t >= z.t) return { x: z.x, y: z.y, style: t - z.t < near ? "exact" : "held", kind: z.kind };
  for (let i = 0; i < c.length - 1; i++) {
    const p = c[i], q = c[i + 1];
    if (t < q.t) {
      const u = (t - p.t) / Math.max(q.t - p.t, 1e-6);
      const hit = t - p.t < near ? p : q.t - t < near ? q : null;
      return {
        x: p.x + u * (q.x - p.x),
        y: p.y + u * (q.y - p.y),
        style: hit ? "exact" : "between",
        kind: hit ? hit.kind : null,
      };
    }
  }
  return null;
}

function phaseAt(t) {
  const { start, contact, end } = R.ev;
  if (start != null && t < start) return ["normal", "Before onset"];
  if (contact == null || t < contact) return ["distress", "Drowning: onset → contact"];
  if (end == null || t < end) return ["rescue", "Rescue: contact → saved"];
  return ["after", "After save"];
}

function draw() {
  const t = video.currentTime;
  // loop the window while playing
  if (!video.paused && (t >= R.b || t < R.a - 0.25)) {
    video.currentTime = R.a;
    R.prevT = R.a;
  }
  // flash a mark's name as the playhead crosses it
  for (const [key, name] of MARKS) {
    const m = R.ev[key];
    if (m != null && R.prevT < m && t >= m && t - R.prevT < 0.5) {
      R.flash = { text: name.toUpperCase(), until: performance.now() + 900 };
    }
  }
  R.prevT = t;

  const [cls, label] = phaseAt(t);
  const ph = $("#phase");
  ph.className = `phase ${cls}`;
  ph.textContent = label;
  $("#clock").textContent = `${fmt(t)} · frame ${Math.floor(t * R.fps)}`;
  const span = Math.max(R.b - R.a, 0.001);
  $("#tlHead").style.left = `${Math.min(100, Math.max(0, (100 * (t - R.a)) / span))}%`;

  ctx.clearRect(0, 0, R.vw, R.vh);
  const v = victimAt(t);
  const lw = Math.max(2, R.vw / 480);
  if (v) {
    const r = Math.max(18, R.vw * 0.022);
    ctx.save();
    ctx.lineWidth = lw * (v.style === "exact" ? 1.6 : 1);
    ctx.strokeStyle = v.style === "held" ? "rgba(250,204,21,0.45)" : "#facc15";
    ctx.setLineDash(v.style === "exact" ? [] : [lw * 4, lw * 3]);
    ctx.beginPath();
    ctx.arc(v.x, v.y, r, 0, Math.PI * 2);
    ctx.stroke();
    if (v.style === "exact" && v.kind) {
      ctx.font = `600 ${Math.round(r * 0.7)}px -apple-system, sans-serif`;
      ctx.fillStyle = "#facc15";
      ctx.fillText(v.kind.replace("_", " "), v.x + r + lw * 2, v.y - r * 0.4);
    }
    ctx.restore();
  }
  if (R.flash && performance.now() < R.flash.until) {
    const size = Math.round(R.vh * 0.06);
    ctx.font = `700 ${size}px -apple-system, sans-serif`;
    const w = ctx.measureText(R.flash.text).width;
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect((R.vw - w) / 2 - size * 0.4, size * 0.4, w + size * 0.8, size * 1.4);
    ctx.fillStyle = "#fff";
    ctx.fillText(R.flash.text, (R.vw - w) / 2, size * 1.45);
  }

  // follow the victim when zoomed
  // keeps the zoomed frame covering the stage (the stage clips the rest), like the Rescue tab
  const { w: bw, h: bh } = R.box;
  if (R.zoom && v) {
    const mx = (v.x / R.vw) * bw, my = (v.y / R.vh) * bh;
    const tx = Math.min(0, Math.max(bw - ZOOM * bw, bw / 2 - ZOOM * mx));
    const ty = Math.min(0, Math.max(bh - ZOOM * bh, bh / 2 - ZOOM * my));
    setTransform(`translate(${tx}px, ${ty}px) scale(${ZOOM})`);
  } else {
    setTransform("");
  }
  const playLabel = video.paused ? "▶ Play <kbd>Space</kbd>" : "❚❚ Pause <kbd>Space</kbd>";
  if ($("#playBtn").innerHTML !== playLabel) $("#playBtn").innerHTML = playLabel;
  requestAnimationFrame(draw);
}

// Write the zoom transform only when it changes: rewriting it every frame can blank the video.
function setTransform(value) {
  if (zoomer.style.transform !== value) zoomer.style.transform = value;
}

function togglePlay() {
  if (video.paused) {
    if (video.currentTime >= R.b || video.currentTime < R.a) video.currentTime = R.a;
    video.play();
  } else {
    video.pause();
  }
}

function toggleZoom() {
  R.zoom = !R.zoom;
  $("#zoomBtn").classList.toggle("active", R.zoom);
  $("#zoomBadge").hidden = !R.zoom;
}

// ---------------------------------------------------------------- wiring

$("#prev").onclick = () => openClip(R.idx - 1);
$("#next").onclick = () => openClip(R.idx + 1);
$("#nextTodo").onclick = nextUnreviewed;
$("#goodBtn").onclick = markGood;
$("#badBtn").onclick = markBad;
$("#clearBtn").onclick = () => sendVerdict(null);
$("#playBtn").onclick = togglePlay;
$("#zoomBtn").onclick = toggleZoom;
$$("[data-speed]").forEach((b) => {
  b.onclick = () => {
    video.playbackRate = Number(b.dataset.speed);
    $$("[data-speed]").forEach((x) => x.classList.toggle("active", x === b));
  };
});
$("#timeline").onclick = (e) => {
  const box = $("#timeline").getBoundingClientRect();
  video.currentTime = R.a + ((e.clientX - box.left) / box.width) * (R.b - R.a);
};
// a note typed after the verdict is saved when you leave the box
$("#note").addEventListener("change", () => {
  const row = R.rows[R.idx];
  if (row && row.verdict) sendVerdict(row.verdict);
});
window.addEventListener("resize", fitStage);

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA") {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "g") markGood();
  else if (k === "n") markBad();
  else if (k === "z") toggleZoom();
  else if (k === "arrowleft") openClip(R.idx - 1);
  else if (k === "arrowright") openClip(R.idx + 1);
});

loadQueue();
requestAnimationFrame(draw);
