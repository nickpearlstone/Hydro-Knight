"""
Interactive clip annotator — tkinter UI with OpenCV video backend.

Layout
------
  Left panel  : video canvas + progress bar + playback controls
  Right panel : metadata dropdowns (camera, setting, time, weather)
                label buttons, trim controls, delete button

Keyboard shortcuts
------------------
  SPACE         pause / resume
  LEFT / RIGHT  seek -5s / +5s
  I / O         set trim in-point / out-point at current position
  [ / ]         mark event start / end (uses the Event-type dropdown)
  \\            clear all events
  N/D/S/F/R     label normal / distress / submerged / face_down / review
  X             delete clip (blocklist + remove files)
  Q             quit session
"""

from __future__ import annotations

import dataclasses
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from ..ingest.blocklist import Blocklist
from ..ingest.download import done_marker, is_downloaded, local_path
from ..ingest.manifest import (
    CameraView,
    ClipRecord,
    Label,
    Manifest,
    Setting,
    TimeOfDay,
    Weather,
)

FRAME_DELAY_MS = 40  # ~25fps refresh rate for the UI


def _fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Main annotator window for a single clip
# ---------------------------------------------------------------------------


class ClipAnnotator(tk.Toplevel):
    """
    A tkinter window that plays one clip and collects annotations.

    tkinter is Python's built-in GUI library. A Toplevel is a child window
    separate from the hidden root window. We use Toplevel so we can create
    and destroy one window per clip without restarting the whole application.
    """

    def __init__(
        self, master, record: ClipRecord, manifest: Manifest, blocklist: Blocklist
    ):
        super().__init__(master)
        self.record = record
        self.manifest = manifest
        self.blocklist = blocklist
        self.action = "quit"  # updated when the user makes a decision

        self.title(record.notes[:80] or record.clip_id)
        self.configure(bg="#1e1e1e")
        self.resizable(True, True)

        # --- Video capture ---
        self.cap = cv2.VideoCapture(str(local_path(record)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_sec = self.total_frames / self.fps
        self.paused = False
        self.current_sec = 0.0

        # Trim points — seeded from existing manifest values if already set
        self.in_point = record.start_sec if record.start_sec > 0 else None
        self.out_point = record.end_sec if record.end_sec > 0 else None

        # Event windows — list of typed dicts {"start","end","label"} marking
        # where a specific anomaly is visible. Seeded from the record.
        # `_event_pending` holds a start that hasn't been closed with an end yet.
        self.events: list[dict] = [dict(e) for e in record.events]
        self._event_pending: float | None = None

        self._build_ui()
        self._bind_keys()

        # Force the window to the front. On macOS, tkinter windows launched from
        # a terminal often open *behind* other apps. lift() + a momentary
        # topmost flag + focus_force pulls it forward; we drop topmost right
        # after so it doesn't stay pinned above everything.
        self.lift()
        self.attributes("-topmost", True)
        self.after(300, lambda: self.attributes("-topmost", False))
        self.focus_force()

        # Start the frame loop after a short delay so the window can render first
        self.after(100, self._next_frame)

        # Block until this window is closed (wait_window makes the call synchronous)
        self.wait_window(self)

    # --- UI construction ---

    def _build_ui(self):
        """Build the two-panel layout."""

        # ── Left panel: video ──────────────────────────────────────────────
        left = tk.Frame(self, bg="#1e1e1e")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas where video frames are drawn.
        # ImageTk.PhotoImage is the bridge between Pillow images and tkinter.
        self.canvas = tk.Canvas(
            left, width=960, height=540, bg="black", highlightthickness=0
        )
        self.canvas.pack()

        # Progress bar — a thin canvas we draw a rectangle on
        self.progress_canvas = tk.Canvas(
            left, height=22, bg="#333", highlightthickness=0, cursor="hand2"
        )
        self.progress_canvas.pack(fill=tk.X, pady=(4, 0))
        # Click anywhere on the bar to seek; drag (B1-Motion) to scrub.
        self.progress_canvas.bind("<Button-1>", self._seek_click)
        self.progress_canvas.bind("<B1-Motion>", self._seek_click)

        # Timestamp label
        self.time_label = tk.Label(
            left, text="00:00 / 00:00", bg="#1e1e1e", fg="#aaa", font=("Helvetica", 11)
        )
        self.time_label.pack(pady=(2, 6))

        # Playback control buttons
        ctrl = tk.Frame(left, bg="#1e1e1e")
        ctrl.pack()

        def btn(text, cmd):
            return self._mk_button(ctrl, text, cmd)

        btn("◀◀ -5s", lambda: self._seek_relative(-5)).pack(side=tk.LEFT, padx=4)
        self.play_btn = self._mk_button(ctrl, "⏸ Pause", self._toggle_pause)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        btn("+5s ▶▶", lambda: self._seek_relative(5)).pack(side=tk.LEFT, padx=4)

        # ── Right panel: metadata + actions ───────────────────────────────
        right = tk.Frame(self, bg="#2a2a2a", width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        right.pack_propagate(False)  # keep fixed width

        def section(text):
            tk.Label(
                right, text=text, bg="#2a2a2a", fg="#888", font=("Helvetica", 9, "bold")
            ).pack(anchor=tk.W, padx=12, pady=(14, 2))

        def dropdown(parent, label_text, options, initial):
            """
            Build a labelled dropdown (ttk.Combobox) and return its StringVar.
            StringVar is a tkinter object that holds a string value and can
            notify other widgets when it changes — used to read the selection.
            """
            tk.Label(
                parent, text=label_text, bg="#2a2a2a", fg="#ccc", font=("Helvetica", 10)
            ).pack(anchor=tk.W, padx=12, pady=(4, 0))
            var = tk.StringVar(value=initial)
            cb = ttk.Combobox(
                parent,
                textvariable=var,
                values=options,
                state="readonly",
                width=22,
                takefocus=False,
            )
            cb.pack(padx=12, pady=(2, 0))
            # Return keyboard focus to the window after a selection so the
            # shortcut keys ([ ] i o n d s f r) keep working afterward.
            cb.bind("<<ComboboxSelected>>", lambda e: self.focus_set())
            return var

        # Metadata dropdowns
        section("METADATA")
        self.var_camera = dropdown(
            right,
            "Camera angle",
            [e.value for e in CameraView],
            self.record.camera_view.value,
        )
        self.var_setting = dropdown(
            right, "Setting", [e.value for e in Setting], self.record.setting.value
        )
        self.var_time = dropdown(
            right,
            "Time of day",
            [e.value for e in TimeOfDay],
            self.record.time_of_day.value,
        )
        self.var_weather = dropdown(
            right, "Weather", [e.value for e in Weather], self.record.weather.value
        )

        # Trim controls
        section("TRIM")
        trim_frame = tk.Frame(right, bg="#2a2a2a")
        trim_frame.pack(fill=tk.X, padx=12, pady=4)

        self.in_label = tk.Label(
            trim_frame,
            text=f"In:  {_fmt(self.in_point or 0)}",
            bg="#2a2a2a",
            fg="#FFD700",
            font=("Helvetica", 10),
        )
        self.in_label.pack(anchor=tk.W)
        self.out_label = tk.Label(
            trim_frame,
            text=f"Out: {_fmt(self.out_point or self.total_sec)}",
            bg="#2a2a2a",
            fg="#FF8C00",
            font=("Helvetica", 10),
        )
        self.out_label.pack(anchor=tk.W)

        self._mk_button(
            trim_frame, "Set In-point  [I]", self._set_in, bg="#444", padx=8, pady=3
        ).pack(fill=tk.X, pady=(6, 2))
        self._mk_button(
            trim_frame, "Set Out-point [O]", self._set_out, bg="#444", padx=8, pady=3
        ).pack(fill=tk.X)

        # Event-window controls — mark WHERE the anomaly is visible.
        # Distinct from trim: trim cuts junk; events mark the positive region
        # while keeping the normal lead-in reusable.
        section("EVENTS (anomaly visible)")
        ev_frame = tk.Frame(right, bg="#2a2a2a")
        ev_frame.pack(fill=tk.X, padx=12, pady=4)

        # Event-type dropdown: the type assigned to the next event you close
        # with ']'. Only the anomaly types are valid choices for an event.
        EVENT_TYPES = [
            Label.DISTRESS.value,
            Label.SUBMERGED.value,
            Label.FACE_DOWN.value,
        ]
        tk.Label(
            ev_frame, text="Event type", bg="#2a2a2a", fg="#ccc", font=("Helvetica", 10)
        ).pack(anchor=tk.W)
        self.var_event_type = tk.StringVar(value=EVENT_TYPES[0])
        ev_cb = ttk.Combobox(
            ev_frame,
            textvariable=self.var_event_type,
            values=EVENT_TYPES,
            state="readonly",
            width=22,
            takefocus=False,
        )
        ev_cb.pack(anchor=tk.W, pady=(2, 4))
        ev_cb.bind("<<ComboboxSelected>>", lambda e: self.focus_set())

        self.event_label = tk.Label(
            ev_frame,
            text="",
            justify=tk.LEFT,
            bg="#2a2a2a",
            fg="#ff6b6b",
            font=("Helvetica", 10),
        )
        self.event_label.pack(anchor=tk.W)

        self._mk_button(
            ev_frame,
            "Mark event start [ [ ]",
            self._event_start,
            bg="#444",
            padx=8,
            pady=3,
        ).pack(fill=tk.X, pady=(6, 2))
        self._mk_button(
            ev_frame,
            "Mark event end  [ ] ]",
            self._event_end,
            bg="#444",
            padx=8,
            pady=3,
        ).pack(fill=tk.X)
        self._mk_button(
            ev_frame,
            "Clear events [ \\ ]",
            self._event_clear,
            bg="#5a3a3a",
            padx=8,
            pady=3,
        ).pack(fill=tk.X, pady=(2, 0))

        self._refresh_event_label()

        # Label buttons
        section("LABEL")
        label_colors = {
            Label.NORMAL: ("#2d6a2d", "N — Normal"),
            Label.DISTRESS: ("#6a2d2d", "D — Distress"),
            Label.SUBMERGED: ("#6a5a2d", "S — Submerged"),
            Label.FACE_DOWN: ("#2d5a6a", "F — Face down"),
            Label.REVIEW: ("#4a4a4a", "R — Review later"),
        }
        for lbl, (color, text) in label_colors.items():
            self._mk_button(
                right,
                text,
                lambda lab=lbl: self._save_and_next(lab),
                bg=color,
                padx=8,
                pady=5,
                anchor=tk.W,
            ).pack(fill=tk.X, padx=12, pady=2)

        # Delete + skip buttons
        section("ACTIONS")
        self._mk_button(
            right, "X — Delete + blocklist", self._delete, bg="#8b0000", padx=8, pady=5
        ).pack(fill=tk.X, padx=12, pady=2)
        tk.Button(
            right,
            text="Q — Quit session",
            command=self._quit,
            bg="#333",
            fg="white",
            relief=tk.FLAT,
            padx=8,
            pady=5,
        ).pack(fill=tk.X, padx=12, pady=2)

    def _bind_keys(self):
        # bind_all binds at the application level, so a shortcut fires no matter
        # which child widget (canvas, button, dropdown) currently has focus.
        # Plain self.bind only fired when the Toplevel itself had focus, which
        # is why keys silently stopped working after clicking a dropdown.
        keys = {
            "<space>": lambda e: self._toggle_pause(),
            "<Left>": lambda e: self._seek_relative(-5),
            "<Right>": lambda e: self._seek_relative(5),
            "i": lambda e: self._set_in(),
            "o": lambda e: self._set_out(),
            "bracketleft": lambda e: self._event_start(),
            "bracketright": lambda e: self._event_end(),
            "backslash": lambda e: self._event_clear(),
            "n": lambda e: self._save_and_next(Label.NORMAL),
            "d": lambda e: self._save_and_next(Label.DISTRESS),
            "s": lambda e: self._save_and_next(Label.SUBMERGED),
            "f": lambda e: self._save_and_next(Label.FACE_DOWN),
            "r": lambda e: self._save_and_next(Label.REVIEW),
            "x": lambda e: self._delete(),
            "q": lambda e: self._quit(),
        }
        for seq, fn in keys.items():
            self.bind_all(seq, fn)
        # Also bind the literal bracket/backslash characters in case the
        # keysym names differ on this keyboard layout — belt and suspenders.
        self.bind_all("[", lambda e: self._event_start())
        self.bind_all("]", lambda e: self._event_end())
        self.bind_all("\\", lambda e: self._event_clear())
        self.focus_set()

    def _mk_button(self, parent, text, cmd, **kw):
        """
        Build a button that never keeps keyboard focus.

        takefocus=False stops Tab-focus, and after the command runs we hand
        focus back to the window (guarded with winfo_exists since some commands
        — label/delete/quit — destroy the window). Without this, clicking a
        button left focus off the window and the shortcut keys stopped firing.
        """

        def run():
            cmd()
            try:
                if self.winfo_exists():
                    self.focus_set()
            except tk.TclError:
                pass

        opts = dict(
            bg="#333", fg="white", relief=tk.FLAT, padx=10, pady=4, takefocus=False
        )
        opts.update(kw)
        return tk.Button(parent, text=text, command=run, **opts)

    # --- Frame loop ---

    def _next_frame(self):
        """
        Read the next frame from OpenCV and display it on the canvas.

        self.after(delay, callback) is tkinter's way of scheduling a function
        to run after `delay` milliseconds without blocking the UI. By scheduling
        itself at the end of each call, it creates a continuous loop that stops
        when the window is destroyed.
        """
        if self.paused:
            # Keep the progress bar live while paused so trim/event markers
            # appear the moment they're set.
            self._draw_progress()
            self.after(FRAME_DELAY_MS, self._next_frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames - 1)
            self.paused = True
            self.play_btn.config(text="▶ Play")
            self.after(FRAME_DELAY_MS, self._next_frame)
            return

        self.current_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self._update_display(frame)
        self.after(FRAME_DELAY_MS, self._next_frame)

    def _update_display(self, frame):
        """Convert an OpenCV frame (BGR numpy array) to a tkinter image and draw it."""

        # OpenCV stores colors as BGR (Blue, Green, Red). Pillow expects RGB.
        # cvtColor converts between color formats.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fit the canvas while preserving aspect ratio
        h, w = rgb.shape[:2]
        canvas_w, canvas_h = 960, 540
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)

        # We must keep a reference to the PhotoImage or Python's garbage
        # collector will delete it before tkinter can display it.
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=self._photo
        )

        # Update timestamp and progress bar
        self.time_label.config(
            text=f"{_fmt(self.current_sec)} / {_fmt(self.total_sec)}"
        )
        self._draw_progress()

    def _draw_progress(self):
        """Redraw the progress bar with current position, trim and event markers."""
        w = self.progress_canvas.winfo_width()
        h = int(self.progress_canvas.winfo_height()) or 18
        self.progress_canvas.delete("all")

        if self.total_sec <= 0 or w <= 1:
            return

        # Background
        self.progress_canvas.create_rectangle(0, 0, w, h, fill="#333", outline="")

        # Played portion (thin strip along the bottom so it doesn't hide bands)
        played_w = int(w * self.current_sec / self.total_sec)
        self.progress_canvas.create_rectangle(
            0, h - 4, played_w, h, fill="#4a9a4a", outline=""
        )

        # Event windows — solid bands across the upper area, colored by type.
        # Enforce a minimum width in PIXELS (not seconds) so a short event on
        # a long video is still visible rather than collapsing sub-pixel.
        type_color = {
            "distress": "#e23b3b",
            "submerged": "#e0922e",
            "face_down": "#33aacc",
        }
        for e in self.events:
            xs = int(w * e["start"] / self.total_sec)
            xe = int(w * e["end"] / self.total_sec)
            if xe - xs < 4:
                xe = xs + 4
            color = type_color.get(e.get("label"), "#e23b3b")
            self.progress_canvas.create_rectangle(
                xs, 0, xe, h - 4, fill=color, outline=""
            )

        # Trim markers — bright full-height bars (drawn last, on top)
        if self.in_point is not None:
            x = int(w * self.in_point / self.total_sec)
            self.progress_canvas.create_rectangle(
                x - 2, 0, x + 2, h, fill="#FFD700", outline=""
            )
        if self.out_point is not None:
            x = int(w * self.out_point / self.total_sec)
            self.progress_canvas.create_rectangle(
                x - 2, 0, x + 2, h, fill="#FF8C00", outline=""
            )

    # --- Controls ---

    def _toggle_pause(self):
        self.paused = not self.paused
        self.play_btn.config(text="▶ Play" if self.paused else "⏸ Pause")

    def _seek_relative(self, delta_sec: float):
        target = max(0.0, min(self.total_sec, self.current_sec + delta_sec))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000)
        self._refresh_if_paused()

    def _seek_click(self, event):
        """Seek to position clicked on the progress bar."""
        w = self.progress_canvas.winfo_width()
        if w <= 0:
            return
        ratio = event.x / w
        self.cap.set(cv2.CAP_PROP_POS_MSEC, ratio * self.total_sec * 1000)
        self._refresh_if_paused()

    def _refresh_if_paused(self):
        """When paused, read and show the frame at the new seek position."""
        if not self.paused:
            return
        ret, frame = self.cap.read()
        if ret:
            self.current_sec = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            self._update_display(frame)

    def _set_in(self):
        self.in_point = self.current_sec
        self.in_label.config(text=f"In:  {_fmt(self.in_point)}")

    def _set_out(self):
        self.out_point = self.current_sec
        self.out_label.config(text=f"Out: {_fmt(self.out_point)}")

    # --- Event windows ---

    def _event_start(self):
        """Mark the start of an event at the current position."""
        self._event_pending = self.current_sec
        self._refresh_event_label()
        self._draw_progress()
        print(f"  Event start pending at {_fmt(self.current_sec)} — press ] to close")

    def _event_end(self):
        """Close the pending event and store it with the selected type."""
        if self._event_pending is None:
            print("  (press '[' to mark event start first)")
            return
        start = self._event_pending
        end = self.current_sec
        # Guard against marking end before start (e.g. after seeking back).
        if end < start:
            start, end = end, start
        label = self.var_event_type.get()
        self.events.append({"start": start, "end": end, "label": label})
        self._event_pending = None
        self._refresh_event_label()
        self._draw_progress()
        print(f"  Event marked: {label} {_fmt(start)}–{_fmt(end)}")

    def _event_clear(self):
        """Remove all events and any pending start."""
        self.events = []
        self._event_pending = None
        self._refresh_event_label()
        self._draw_progress()
        print("  Events cleared.")

    def _refresh_event_label(self):
        """Update the right-panel readout of marked events."""
        lines = [
            f"{e['label']}: {_fmt(e['start'])}–{_fmt(e['end'])}" for e in self.events
        ]
        if self._event_pending is not None:
            lines.append(
                f"{self.var_event_type.get()}: {_fmt(self._event_pending)}– (pending…)"
            )
        self.event_label.config(text="\n".join(lines) if lines else "none")

    def _current_metadata(self) -> dict:
        """Read the current dropdown values and return them as a dict."""
        return {
            "camera_view": CameraView(self.var_camera.get()),
            "setting": Setting(self.var_setting.get()),
            "time_of_day": TimeOfDay(self.var_time.get()),
            "weather": Weather(self.var_weather.get()),
        }

    def _save_and_next(self, label: Label):
        meta = self._current_metadata()
        updated = dataclasses.replace(
            self.record,
            label=label,
            start_sec=self.in_point
            if self.in_point is not None
            else self.record.start_sec,
            end_sec=self.out_point
            if self.out_point is not None
            else self.record.end_sec,
            camera_view=meta["camera_view"],
            setting=meta["setting"],
            time_of_day=meta["time_of_day"],
            weather=meta["weather"],
            events=self.events,
        )
        self.manifest.update(updated)
        print(
            f"  Saved: {label.value} | {meta['camera_view'].value} | {meta['setting'].value} | {meta['time_of_day'].value} | {meta['weather'].value} | events={len(self.events)}"
        )
        self.action = "next"
        self._close()

    def _delete(self):
        self.blocklist.add(self.record.source_url)
        self.manifest.delete(self.record.clip_id)
        video = local_path(self.record)
        marker = done_marker(self.record)
        if video.exists():
            video.unlink()
        if marker.exists():
            marker.unlink()
        print(f"  Deleted {self.record.clip_id} — video removed, URL blocklisted.")
        self.action = "deleted"
        self._close()

    def _quit(self):
        self.action = "quit"
        self._close()

    def _close(self):
        self.cap.release()
        self.destroy()


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------


def annotate(
    manifest_path: Path,
    include_review: bool = False,
    review_only: bool = False,
) -> None:
    """Run an annotation session over downloaded clips (one tkinter window per clip).

    By default only UNLABELED clips are queued; REVIEW clips are skipped so you don't re-see them.
    Args:
        manifest_path: JSONL manifest of clips to annotate.
        include_review: also queue REVIEW clips alongside UNLABELED.
        review_only: queue ONLY REVIEW clips (a dedicated review pass).
    Returns:
        None (writes label/metadata/event updates back to the manifest as you go).
    """
    manifest = Manifest(manifest_path)
    blocklist = Blocklist()
    records = manifest.load()

    if review_only:
        wanted = {Label.REVIEW}
    elif include_review:
        wanted = {Label.UNLABELED, Label.REVIEW}
    else:
        wanted = {Label.UNLABELED}

    queue = [r for r in records if is_downloaded(r) and r.label in wanted]

    if not queue:
        print(
            "No clips to annotate. Run download.py first, "
            "or pass include_review=True / review_only=True to revisit REVIEW clips."
        )
        return

    print(f"{len(queue)} clips in queue.")

    # Hidden root window — tkinter requires one root to exist even if we only
    # ever show Toplevel child windows. We withdraw it so it never appears.
    root = tk.Tk()
    root.withdraw()

    for i, record in enumerate(queue, start=1):
        print(f"\n[{i}/{len(queue)}] {record.notes[:70] or record.clip_id}")
        win = ClipAnnotator(root, record, manifest, blocklist)
        if win.action == "quit":
            print("Session ended.")
            break

    root.destroy()
    print("\nAnnotation session complete.")


if __name__ == "__main__":
    annotate(
        manifest_path=Path("data/manifests/pool_footage.jsonl"),
    )
