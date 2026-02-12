#!/usr/bin/env python3
"""Tkinter tool for painting spawn zones (green) and kill boundaries (red) on map PGMs.

Saves a single `{map_name}_zones.png` companion file (RGB: R=kill, G=spawn).

Controls:
    Scroll              Zoom in/out (centered on cursor)
    Middle mouse drag   Pan the map
    Ctrl+LMB drag       Pan the map (alternative)
    LMB drag            Paint active layer
    RMB drag            Erase from active layer
    Ctrl+Z              Undo
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.map_loader import read_pgm  # noqa: E402

BG_COLOR = "#cdcdcd"
ZOOM_MIN = 0.2
ZOOM_MAX = 10.0
ZOOM_FACTOR = 1.15
WINDOW_W = 1200
WINDOW_H = 800


class ZonePainter:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Map Zone Painter")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")

        self.map_image: np.ndarray | None = None
        self.map_path: Path | None = None
        self.spawn_layer: np.ndarray | None = None
        self.kill_layer: np.ndarray | None = None
        self.lookat_layer: np.ndarray | None = None

        # Viewport
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Pan drag state
        self._pan_start_x = 0
        self._pan_start_y = 0
        self._pan_start_px = 0.0
        self._pan_start_py = 0.0

        self.brush_size = 4  # radius in map pixels
        self.mode = tk.StringVar(value="kill")
        self.show_kill = tk.BooleanVar(value=True)
        self.show_spawn = tk.BooleanVar(value=True)
        self.show_lookat = tk.BooleanVar(value=True)

        # Undo
        self._undo_stack: list[tuple[str, np.ndarray]] = []
        self._undo_max = 200
        self._stroke_active = False
        self._stroke_layer: str = ""
        self._stroke_snapshot: np.ndarray | None = None

        # Brush cursor tracking
        self._cursor_x = 0
        self._cursor_y = 0

        self._build_ui()
        self._composite: np.ndarray | None = None

    # ── UI ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self.root, bg=BG_COLOR)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        tk.Button(toolbar, text="Open PGM", command=self._open_pgm).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Save Zones", command=self._save_zones).pack(side=tk.LEFT, padx=4)

        tk.Frame(toolbar, width=2, bg="gray").pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        tk.Radiobutton(toolbar, text="Kill (Red)", variable=self.mode, value="kill",
                        bg=BG_COLOR, fg="#cc0000", selectcolor=BG_COLOR).pack(side=tk.LEFT)
        tk.Radiobutton(toolbar, text="Spawn (Green)", variable=self.mode, value="spawn",
                        bg=BG_COLOR, fg="#008800", selectcolor=BG_COLOR).pack(side=tk.LEFT)
        tk.Radiobutton(toolbar, text="LookAt (Yellow)", variable=self.mode, value="lookat",
                        bg=BG_COLOR, fg="#aa8800", selectcolor=BG_COLOR).pack(side=tk.LEFT)

        tk.Frame(toolbar, width=2, bg="gray").pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        tk.Label(toolbar, text="Brush:", bg=BG_COLOR).pack(side=tk.LEFT)
        self.brush_slider = tk.Scale(toolbar, from_=1, to=60, orient=tk.HORIZONTAL,
                                      length=140, command=self._on_brush_change, bg=BG_COLOR)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(side=tk.LEFT, padx=4)

        tk.Frame(toolbar, width=2, bg="gray").pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        tk.Checkbutton(toolbar, text="Show Kill", variable=self.show_kill,
                        command=self._on_layer_toggle, bg=BG_COLOR, selectcolor=BG_COLOR).pack(side=tk.LEFT)
        tk.Checkbutton(toolbar, text="Show Spawn", variable=self.show_spawn,
                        command=self._on_layer_toggle, bg=BG_COLOR, selectcolor=BG_COLOR).pack(side=tk.LEFT)
        tk.Checkbutton(toolbar, text="Show LookAt", variable=self.show_lookat,
                        command=self._on_layer_toggle, bg=BG_COLOR, selectcolor=BG_COLOR).pack(side=tk.LEFT)

        tk.Frame(toolbar, width=2, bg="gray").pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        tk.Button(toolbar, text="Clear Kill", command=self._clear_kill).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Clear Spawn", command=self._clear_spawn).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Clear LookAt", command=self._clear_lookat).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Fit", command=self._fit_view).pack(side=tk.LEFT, padx=4)

        self.canvas = tk.Canvas(self.root, bg="#222222", cursor="none", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Paint / erase
        self.canvas.bind("<Button-1>", self._on_lmb_down)
        self.canvas.bind("<B1-Motion>", self._on_lmb_motion)
        self.canvas.bind("<Button-3>", self._on_rmb_down)
        self.canvas.bind("<B3-Motion>", self._erase)

        # End of stroke
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self.canvas.bind("<ButtonRelease-3>", self._on_rmb_release)

        # Middle mouse pan
        self.canvas.bind("<Button-2>", self._on_mmb_down)
        self.canvas.bind("<B2-Motion>", self._on_mmb_motion)

        # Zoom
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Button-4>", self._on_scroll)
        self.canvas.bind("<Button-5>", self._on_scroll)

        # Brush preview cursor
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

        # Undo
        self.root.bind("<Control-z>", self._undo)

        # Status bar
        self.status = tk.Label(self.root, text="Open a PGM file to begin.", bg=BG_COLOR, anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=2)

        self._photo: ImageTk.PhotoImage | None = None

    # ── File I/O ────────────────────────────────────────────────────

    def _open_pgm(self) -> None:
        path = filedialog.askopenfilename(
            title="Select map PGM",
            filetypes=[("PGM files", "*.pgm"), ("All files", "*.*")],
            initialdir=str(ROOT / "assets" / "maps"),
        )
        if not path:
            return
        self.map_path = Path(path)
        try:
            self.map_image = read_pgm(self.map_path)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to read PGM: {exc}")
            return

        h, w = self.map_image.shape
        self.spawn_layer = np.zeros((h, w), dtype=bool)
        self.kill_layer = np.zeros((h, w), dtype=bool)
        self.lookat_layer = np.zeros((h, w), dtype=bool)
        self._undo_stack.clear()

        zones_path = self.map_path.with_name(self.map_path.stem + "_zones.png")
        if zones_path.exists():
            try:
                zones_img = Image.open(zones_path).convert("RGB")
                zones_arr = np.asarray(zones_img)
                if zones_arr.shape[:2] == (h, w):
                    self.kill_layer = zones_arr[:, :, 0] >= 128
                    self.spawn_layer = zones_arr[:, :, 1] >= 128
                    self.lookat_layer = zones_arr[:, :, 2] >= 128
                    self.status.config(text=f"Loaded existing zones from {zones_path.name}")
                else:
                    self.status.config(text="Zone file shape mismatch, starting fresh")
            except Exception:
                self.status.config(text=f"Could not read {zones_path.name}, starting fresh")
        else:
            self.status.config(text=f"Opened {self.map_path.name} ({w}x{h})")

        self._rebuild_composite()
        self._fit_view()

    def _save_zones(self) -> None:
        if self.map_image is None or self.map_path is None:
            messagebox.showwarning("No map", "Open a PGM file first.")
            return
        if self.kill_layer is None or self.spawn_layer is None:
            return

        has_kill = bool(np.any(self.kill_layer))
        has_spawn = bool(np.any(self.spawn_layer))
        has_lookat = self.lookat_layer is not None and bool(np.any(self.lookat_layer))
        if not has_kill and not has_spawn and not has_lookat:
            messagebox.showinfo("Empty", "All layers are empty. Nothing to save.")
            return

        h, w = self.map_image.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[self.kill_layer, 0] = 255
        out[self.spawn_layer, 1] = 255
        if has_lookat:
            out[self.lookat_layer, 2] = 255

        zones_path = self.map_path.with_name(self.map_path.stem + "_zones.png")
        try:
            Image.fromarray(out, "RGB").save(zones_path)
        except Exception as exc:
            messagebox.showerror("Error", f"Save failed: {exc}")
            return

        kill_count = int(np.count_nonzero(self.kill_layer))
        spawn_count = int(np.count_nonzero(self.spawn_layer))
        lookat_count = int(np.count_nonzero(self.lookat_layer)) if has_lookat else 0
        self.status.config(text=f"Saved {zones_path.name} (kill: {kill_count}px, spawn: {spawn_count}px, lookat: {lookat_count}px)")

    # ── Layer operations ────────────────────────────────────────────

    def _clear_kill(self) -> None:
        if self.kill_layer is not None:
            self.kill_layer[:] = False
            self._rebuild_composite()
            self._refresh_viewport()

    def _clear_spawn(self) -> None:
        if self.spawn_layer is not None:
            self.spawn_layer[:] = False
            self._rebuild_composite()
            self._refresh_viewport()

    def _clear_lookat(self) -> None:
        if self.lookat_layer is not None:
            self.lookat_layer[:] = False
            self._rebuild_composite()
            self._refresh_viewport()

    def _on_layer_toggle(self) -> None:
        self._rebuild_composite()
        self._refresh_viewport()

    def _on_brush_change(self, val: str) -> None:
        self.brush_size = int(val)
        self._update_brush_cursor(self._cursor_x, self._cursor_y)

    # ── Coordinate helpers ──────────────────────────────────────────

    def _canvas_to_map(self, cx: int, cy: int) -> tuple[int, int] | None:
        if self.map_image is None:
            return None
        mx = int((cx / self.zoom) + self.pan_x)
        my = int((cy / self.zoom) + self.pan_y)
        h, w = self.map_image.shape
        if 0 <= mx < w and 0 <= my < h:
            return mx, my
        return None

    def _map_to_canvas(self, mx: float, my: float) -> tuple[float, float]:
        cx = (mx - self.pan_x) * self.zoom
        cy = (my - self.pan_y) * self.zoom
        return cx, cy

    # ── Brush preview cursor ────────────────────────────────────────

    def _on_mouse_move(self, event: tk.Event) -> None:
        self._cursor_x = event.x
        self._cursor_y = event.y
        self._update_brush_cursor(event.x, event.y)

    def _on_mouse_leave(self, event: tk.Event) -> None:
        self.canvas.delete("brush_cursor")

    def _update_brush_cursor(self, cx: int, cy: int) -> None:
        self.canvas.delete("brush_cursor")
        if self.map_image is None:
            return
        pos = self._canvas_to_map(cx, cy)
        if pos is None:
            return
        mx, my = pos
        r = self.brush_size
        # Map pixel corners of the brush rectangle
        x0, y0 = mx - r, my - r
        x1, y1 = mx + r + 1, my + r + 1
        # Convert to canvas coords
        cx0, cy0 = self._map_to_canvas(x0, y0)
        cx1, cy1 = self._map_to_canvas(x1, y1)
        mode = self.mode.get()
        color = "#ff4444" if mode == "kill" else "#44ff44" if mode == "spawn" else "#ffdd44"
        self.canvas.create_rectangle(cx0, cy0, cx1, cy1,
                                      outline=color, width=2, tags="brush_cursor")

    # ── Zoom & Pan ──────────────────────────────────────────────────

    def _on_scroll(self, event: tk.Event) -> None:
        if self.map_image is None:
            return
        if event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            factor = ZOOM_FACTOR
        elif event.num == 5 or (hasattr(event, "delta") and event.delta < 0):
            factor = 1.0 / ZOOM_FACTOR
        else:
            return

        new_zoom = self.zoom * factor
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, new_zoom))
        if new_zoom == self.zoom:
            return

        map_x = (event.x / self.zoom) + self.pan_x
        map_y = (event.y / self.zoom) + self.pan_y
        self.zoom = new_zoom
        self.pan_x = map_x - (event.x / self.zoom)
        self.pan_y = map_y - (event.y / self.zoom)

        self._clamp_pan()
        self._refresh_viewport()
        self._update_brush_cursor(event.x, event.y)

    def _start_pan(self, event: tk.Event) -> None:
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self._pan_start_px = self.pan_x
        self._pan_start_py = self.pan_y

    def _do_pan(self, event: tk.Event) -> None:
        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y
        self.pan_x = self._pan_start_px - dx / self.zoom
        self.pan_y = self._pan_start_py - dy / self.zoom
        self._clamp_pan()
        self._refresh_viewport()
        self._update_brush_cursor(event.x, event.y)

    def _on_lmb_down(self, event: tk.Event) -> None:
        if event.state & 0x4:  # Ctrl held
            self._start_pan(event)
        else:
            self._apply_brush(event.x, event.y, True)

    def _on_lmb_motion(self, event: tk.Event) -> None:
        self._cursor_x = event.x
        self._cursor_y = event.y
        if event.state & 0x4:  # Ctrl held
            self._do_pan(event)
        else:
            self._apply_brush(event.x, event.y, True)
            self._update_brush_cursor(event.x, event.y)

    def _on_mmb_down(self, event: tk.Event) -> None:
        self._start_pan(event)

    def _on_mmb_motion(self, event: tk.Event) -> None:
        self._do_pan(event)

    def _fit_view(self) -> None:
        if self.map_image is None:
            return
        self.root.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            cw, ch = WINDOW_W, WINDOW_H - 80
        h, w = self.map_image.shape
        self.zoom = min(cw / w, ch / h)
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom))
        self.pan_x = (w - cw / self.zoom) / 2.0
        self.pan_y = (h - ch / self.zoom) / 2.0
        self._refresh_viewport()

    def _clamp_pan(self) -> None:
        if self.map_image is None:
            return
        h, w = self.map_image.shape
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        view_w = cw / self.zoom
        view_h = ch / self.zoom
        margin = 50
        self.pan_x = max(-margin, min(w - view_w + margin, self.pan_x))
        self.pan_y = max(-margin, min(h - view_h + margin, self.pan_y))

    # ── Painting ────────────────────────────────────────────────────

    def _begin_stroke(self, layer_name: str) -> None:
        layer = self._get_layer(layer_name)
        if layer is None:
            return
        self._stroke_active = True
        self._stroke_layer = layer_name
        self._stroke_snapshot = layer.copy()

    def _end_stroke(self) -> None:
        if not self._stroke_active or self._stroke_snapshot is None:
            self._stroke_active = False
            return
        layer = self._get_layer(self._stroke_layer)
        if layer is not None and not np.array_equal(layer, self._stroke_snapshot):
            if len(self._undo_stack) >= self._undo_max:
                self._undo_stack.pop(0)
            self._undo_stack.append((self._stroke_layer, self._stroke_snapshot))
        self._stroke_active = False
        self._stroke_snapshot = None

    def _get_layer(self, layer_name: str) -> np.ndarray | None:
        if layer_name == "kill":
            return self.kill_layer
        elif layer_name == "spawn":
            return self.spawn_layer
        elif layer_name == "lookat":
            return self.lookat_layer
        return None

    def _apply_brush(self, cx: int, cy: int, value: bool) -> None:
        pos = self._canvas_to_map(cx, cy)
        if pos is None:
            return
        mx, my = pos
        layer_name = self.mode.get()
        layer = self._get_layer(layer_name)
        if layer is None:
            return
        if not self._stroke_active:
            self._begin_stroke(layer_name)
        h, w = layer.shape
        r = self.brush_size
        y0 = max(0, my - r)
        y1 = min(h, my + r + 1)
        x0 = max(0, mx - r)
        x1 = min(w, mx + r + 1)
        layer[y0:y1, x0:x1] = value
        self._rebuild_composite_region(x0, y0, x1, y1)
        self._refresh_viewport()
        self._update_brush_cursor(cx, cy)

    def _on_rmb_down(self, event: tk.Event) -> None:
        self._apply_brush(event.x, event.y, False)

    def _erase(self, event: tk.Event) -> None:
        self._cursor_x = event.x
        self._cursor_y = event.y
        self._apply_brush(event.x, event.y, False)

    def _on_lmb_release(self, event: tk.Event) -> None:
        self._end_stroke()

    def _on_rmb_release(self, event: tk.Event) -> None:
        self._end_stroke()

    def _undo(self, event: tk.Event = None) -> None:
        if not self._undo_stack:
            return
        layer_name, snapshot = self._undo_stack.pop()
        layer = self._get_layer(layer_name)
        if layer is not None:
            layer[:] = snapshot
        self._rebuild_composite()
        self._refresh_viewport()

    # ── Compositing & rendering ─────────────────────────────────────

    def _rebuild_composite(self) -> None:
        if self.map_image is None:
            self._composite = None
            return
        rgb = np.repeat(self.map_image[:, :, None], 3, axis=2).copy()
        self._apply_overlays(rgb)
        self._composite = rgb

    def _rebuild_composite_region(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._composite is None or self.map_image is None:
            self._rebuild_composite()
            return
        patch = np.repeat(self.map_image[y0:y1, x0:x1, None], 3, axis=2).copy()
        self._apply_overlays_region(patch, x0, y0, x1, y1)
        self._composite[y0:y1, x0:x1] = patch

    def _apply_overlays(self, rgb: np.ndarray) -> None:
        if self.show_kill.get() and self.kill_layer is not None and np.any(self.kill_layer):
            m = self.kill_layer
            rgb[m, 0] = np.minimum(rgb[m, 0].astype(np.int16) + 160, 255).astype(np.uint8)
            rgb[m, 1] = (rgb[m, 1] * 0.3).astype(np.uint8)
            rgb[m, 2] = (rgb[m, 2] * 0.3).astype(np.uint8)
        if self.show_spawn.get() and self.spawn_layer is not None and np.any(self.spawn_layer):
            m = self.spawn_layer
            rgb[m, 0] = (rgb[m, 0] * 0.5).astype(np.uint8)
            rgb[m, 1] = np.minimum(rgb[m, 1].astype(np.int16) + 120, 255).astype(np.uint8)
            rgb[m, 2] = (rgb[m, 2] * 0.5).astype(np.uint8)
        if self.show_lookat.get() and self.lookat_layer is not None and np.any(self.lookat_layer):
            m = self.lookat_layer
            rgb[m, 0] = np.minimum(rgb[m, 0].astype(np.int16) + 140, 255).astype(np.uint8)
            rgb[m, 1] = np.minimum(rgb[m, 1].astype(np.int16) + 140, 255).astype(np.uint8)
            rgb[m, 2] = (rgb[m, 2] * 0.3).astype(np.uint8)

    def _apply_overlays_region(self, patch: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
        if self.show_kill.get() and self.kill_layer is not None:
            m = self.kill_layer[y0:y1, x0:x1]
            if np.any(m):
                patch[m, 0] = np.minimum(patch[m, 0].astype(np.int16) + 160, 255).astype(np.uint8)
                patch[m, 1] = (patch[m, 1] * 0.3).astype(np.uint8)
                patch[m, 2] = (patch[m, 2] * 0.3).astype(np.uint8)
        if self.show_spawn.get() and self.spawn_layer is not None:
            m = self.spawn_layer[y0:y1, x0:x1]
            if np.any(m):
                patch[m, 0] = (patch[m, 0] * 0.5).astype(np.uint8)
                patch[m, 1] = np.minimum(patch[m, 1].astype(np.int16) + 120, 255).astype(np.uint8)
                patch[m, 2] = (patch[m, 2] * 0.5).astype(np.uint8)
        if self.show_lookat.get() and self.lookat_layer is not None:
            m = self.lookat_layer[y0:y1, x0:x1]
            if np.any(m):
                patch[m, 0] = np.minimum(patch[m, 0].astype(np.int16) + 140, 255).astype(np.uint8)
                patch[m, 1] = np.minimum(patch[m, 1].astype(np.int16) + 140, 255).astype(np.uint8)
                patch[m, 2] = (patch[m, 2] * 0.3).astype(np.uint8)

    def _refresh_viewport(self) -> None:
        if self._composite is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        h, w = self._composite.shape[:2]

        vx0 = int(max(0, self.pan_x))
        vy0 = int(max(0, self.pan_y))
        vx1 = int(min(w, self.pan_x + cw / self.zoom + 1))
        vy1 = int(min(h, self.pan_y + ch / self.zoom + 1))

        if vx1 <= vx0 or vy1 <= vy0:
            return

        crop = self._composite[vy0:vy1, vx0:vx1]
        pil_crop = Image.fromarray(crop, "RGB")

        out_w = max(1, int((vx1 - vx0) * self.zoom))
        out_h = max(1, int((vy1 - vy0) * self.zoom))
        resample = Image.NEAREST if self.zoom >= 2.0 else Image.BILINEAR
        pil_scaled = pil_crop.resize((out_w, out_h), resample)

        offset_x = int((vx0 - self.pan_x) * self.zoom)
        offset_y = int((vy0 - self.pan_y) * self.zoom)

        self._photo = ImageTk.PhotoImage(pil_scaled)
        self.canvas.delete("map")
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self._photo, tags="map")
        # Keep brush cursor on top
        self.canvas.tag_raise("brush_cursor")


def main() -> None:
    root = tk.Tk()
    ZonePainter(root)
    root.mainloop()


if __name__ == "__main__":
    main()
