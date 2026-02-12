#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.map_loader import read_pgm  # noqa: E402


BG_RGB = (205, 205, 205)
WHITE_THRESH = 250
OUTLINE_RADIUS = 2
BG_TOL = 10


def write_pgm(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape
    with path.open("wb") as handle:
        handle.write(b"P5\n")
        handle.write(f"{width} {height}\n255\n".encode("ascii"))
        handle.write(image.tobytes())


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    y0 = max(0, dy)
    y1 = mask.shape[0] + min(0, dy)
    x0 = max(0, dx)
    x1 = mask.shape[1] + min(0, dx)
    out[y0:y1, x0:x1] = mask[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= _shift_mask(mask, dy, dx)
    return out


def add_outline(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    white = img >= WHITE_THRESH
    if not np.any(white):
        return img

    bg_candidates = img[~white]
    bg_value = int(np.median(bg_candidates)) if bg_candidates.size else BG_RGB[0]
    bg_min = max(0, bg_value - BG_TOL)
    bg_max = min(255, bg_value + BG_TOL)
    bg_mask = (img >= bg_min) & (img <= bg_max)

    dilated = _dilate(white, OUTLINE_RADIUS)
    border = dilated & ~white & bg_mask
    img[border] = 0
    return img


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PGM Outline Tool")
        self.root.configure(bg=_rgb_to_hex(BG_RGB))

        self._build_ui()

    def _build_ui(self) -> None:
        title = tk.Label(
            self.root,
            text="PGM Outline Tool",
            font=("Arial", 16, "bold"),
            bg=_rgb_to_hex(BG_RGB),
        )
        title.grid(row=0, column=0, columnspan=4, pady=(12, 6))

        canvas = tk.Canvas(self.root, width=420, height=60, bg=_rgb_to_hex(BG_RGB), highlightthickness=0)
        canvas.grid(row=1, column=0, columnspan=4, pady=(0, 12))
        canvas.create_rectangle(20, 24, 400, 36, fill="white", outline="white")

        tk.Label(self.root, text="Prefix:", bg=_rgb_to_hex(BG_RGB)).grid(row=2, column=0, sticky="e", padx=6)
        self.prefix_var = tk.StringVar(value="K_")
        tk.Entry(self.root, textvariable=self.prefix_var, width=8).grid(row=2, column=1, sticky="w")

        tk.Label(self.root, text="Start:", bg=_rgb_to_hex(BG_RGB)).grid(row=2, column=2, sticky="e", padx=6)
        self.start_var = tk.StringVar(value="01")
        tk.Entry(self.root, textvariable=self.start_var, width=6).grid(row=2, column=3, sticky="w")

        tk.Label(self.root, text="->", bg=_rgb_to_hex(BG_RGB)).grid(row=3, column=1, sticky="e")
        tk.Label(self.root, text="End:", bg=_rgb_to_hex(BG_RGB)).grid(row=3, column=2, sticky="e", padx=6)
        self.end_var = tk.StringVar(value="09")
        tk.Entry(self.root, textvariable=self.end_var, width=6).grid(row=3, column=3, sticky="w")

        tk.Label(self.root, text="Folder:", bg=_rgb_to_hex(BG_RGB)).grid(row=4, column=0, sticky="e", padx=6)
        self.folder_var = tk.StringVar(value=str(Path.home() / "maps_pgm"))
        tk.Entry(self.root, textvariable=self.folder_var, width=45).grid(row=4, column=1, columnspan=2, sticky="we")
        tk.Button(self.root, text="Browse", command=self._browse).grid(row=4, column=3, padx=6)

        tk.Button(self.root, text="Process", command=self._process, width=20).grid(
            row=5, column=0, columnspan=4, pady=(10, 6)
        )

        self.log = tk.Text(self.root, width=70, height=12)
        self.log.grid(row=6, column=0, columnspan=4, padx=12, pady=(0, 12))
        self.log.configure(state="disabled")

        for col in range(4):
            self.root.grid_columnconfigure(col, weight=1)

    def _browse(self) -> None:
        folder = filedialog.askdirectory(initialdir=str(Path.home()))
        if folder:
            self.folder_var.set(folder)

    def _log(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _process(self) -> None:
        prefix = self.prefix_var.get().strip()
        start_raw = self.start_var.get().strip()
        end_raw = self.end_var.get().strip()
        folder = Path(self.folder_var.get().strip()).expanduser()

        if not prefix:
            messagebox.showerror("Error", "Prefix is required.")
            return
        if not start_raw.isdigit() or not end_raw.isdigit():
            messagebox.showerror("Error", "Start/End must be numbers (e.g., 01).")
            return

        start = int(start_raw)
        end = int(end_raw)
        if end < start:
            messagebox.showerror("Error", "End must be >= Start.")
            return

        width = max(len(start_raw), len(end_raw))
        folder.mkdir(parents=True, exist_ok=True)

        self._log(f"Using folder: {folder}")
        processed = 0
        for idx in range(start, end + 1):
            name = f"{prefix}{idx:0{width}d}.pgm"
            path = folder / name
            if not path.exists():
                self._log(f"SKIP: {name} not found")
                continue

            try:
                image = read_pgm(path)
            except Exception as exc:
                self._log(f"ERROR: {name} read failed: {exc}")
                continue

            out = add_outline(image)

            backup_path = path.with_suffix(".bak")
            if not backup_path.exists():
                try:
                    shutil.copy2(path, backup_path)
                except Exception as exc:
                    self._log(f"ERROR: {name} backup failed: {exc}")
                    continue

            try:
                write_pgm(path, out)
            except Exception as exc:
                self._log(f"ERROR: {name} write failed: {exc}")
                continue

            processed += 1
            self._log(f"OK: {name} updated (backup: {backup_path.name})")

        self._log(f"Done. Processed: {processed}")


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def main() -> None:
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
