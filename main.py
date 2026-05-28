"""Tkinter application for flame height analysis and heat value calculation."""

from __future__ import annotations

import threading
import time
import tkinter as tk
import os
import sys
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

VERSION = "0.1.0"

PROJECT_DIR = Path(__file__).resolve().parent

# Matplotlib cache — use a folder next to the executable when frozen by PyInstaller
if getattr(sys, "frozen", False):
    MPL_CACHE_DIR = Path(sys.executable).parent / ".mpl-cache"
else:
    MPL_CACHE_DIR = PROJECT_DIR / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")  # suppress deprecation warning on macOS

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from analyzer import FlameAnalyzer, HSVThresholds, analyze_video
from calibration import Calibration
from exporter import export_annotated_frame, export_annotated_video, export_csv
from formula_engine import FormulaEngine, FormulaError
from video_processor import VideoProcessor


class FlameAnalysisApp(tk.Tk):
    """Main application window (ttk/grid version — use --legacy-ui flag)."""

    def __init__(self) -> None:
        super().__init__()
        self.withdraw()  # hide before first show (Tk Aqua fix)
        self.title("Flame Height Analysis")
        self.geometry("1360x860")
        self.minsize(1100, 740)

        self.video = VideoProcessor()
        self.calibration = Calibration()
        self.analyzer = FlameAnalyzer()
        self.formula_engine = FormulaEngine()
        self.results = pd.DataFrame()

        self.current_frame: np.ndarray | None = None
        self.current_measurement = None
        self.photo_original: ImageTk.PhotoImage | None = None
        self.photo_mask: ImageTk.PhotoImage | None = None
        self.display_scale = 1.0
        self.original_display_size = (0, 0)
        self.is_playing = False
        self.analysis_running = False
        self.status_var = tk.StringVar(value="Starting...")
        self.plot_panel = None

        self._build_ui()
        self.status_var.set("Ready. Open a video using the Open Video button.")
        self.after(100, self._show_window)

    def _build_ui(self) -> None:
        """Builds the GUI layout."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=8)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(9, weight=1)

        ttk.Button(top, text="Open Video", command=self.open_video).grid(row=0, column=0, padx=3)
        ttk.Button(top, text="Play", command=self.toggle_playback).grid(row=0, column=1, padx=3)
        ttk.Button(top, text="◀ Frame", command=lambda: self.step_frame(-1)).grid(row=0, column=2, padx=3)
        ttk.Button(top, text="Frame ▶", command=lambda: self.step_frame(1)).grid(row=0, column=3, padx=3)
        ttk.Label(top, text="Frame step").grid(row=0, column=4, padx=(14, 3))
        self.frame_step_var = tk.IntVar(value=5)
        ttk.Spinbox(top, from_=1, to=500, textvariable=self.frame_step_var, width=6).grid(row=0, column=5)
        ttk.Button(top, text="Analyse Video", command=self.start_analysis).grid(row=0, column=6, padx=8)
        ttk.Button(top, text="Export CSV", command=self.export_results).grid(row=0, column=7, padx=3)
        ttk.Button(top, text="Export Frame", command=self.export_frame).grid(row=0, column=8, padx=3)
        ttk.Button(top, text="Export Video", command=self.export_video).grid(row=0, column=9, padx=3, sticky="w")

        main = ttk.Frame(self, padding=(8, 4))
        main.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")

        viewer = ttk.Frame(left)
        viewer.pack(fill=tk.BOTH, expand=True)
        viewer.columnconfigure(0, weight=1)
        viewer.columnconfigure(1, weight=1)
        viewer.rowconfigure(1, weight=1)

        ttk.Label(viewer, text="Original Frame").grid(row=0, column=0, sticky="w")
        ttk.Label(viewer, text="Main Flame Mask").grid(row=0, column=1, sticky="w")
        self.original_label = ttk.Label(viewer, anchor=tk.CENTER)
        self.original_label.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.original_label.bind("<Button-1>", self.on_image_click)
        self.mask_label = ttk.Label(viewer, anchor=tk.CENTER)
        self.mask_label.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        slider_frame = ttk.Frame(left)
        slider_frame.pack(fill=tk.X, pady=(8, 0))
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider)
        self.frame_slider.pack(fill=tk.X, side=tk.LEFT, expand=True)
        self.frame_label = ttk.Label(slider_frame, text="Frame: - | Time: -")
        self.frame_label.pack(side=tk.LEFT, padx=10)

        controls = ttk.Notebook(right)
        controls.pack(fill=tk.BOTH, expand=True)

        detection_tab = ttk.Frame(controls, padding=8)
        formula_tab = ttk.Frame(controls, padding=8)
        results_tab = ttk.Frame(controls, padding=8)
        controls.add(detection_tab, text="Detection")
        controls.add(formula_tab, text="Formula")
        controls.add(results_tab, text="Results")

        self._build_detection_controls(detection_tab)
        self._build_formula_controls(formula_tab)
        self._build_results_panel(results_tab)

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(8, 4))
        status.grid(row=2, column=0, sticky="ew")

    def _build_detection_controls(self, parent: ttk.Frame) -> None:
        """Creates segmentation and calibration controls."""
        self.h_min = tk.IntVar(value=0)
        self.h_max = tk.IntVar(value=35)
        self.s_min = tk.IntVar(value=100)
        self.s_max = tk.IntVar(value=255)
        self.v_min = tk.IntVar(value=150)
        self.v_max = tk.IntVar(value=255)
        self.min_area_var = tk.IntVar(value=500)
        self.percentile_var = tk.DoubleVar(value=5.0)
        self.distance_cm_var = tk.DoubleVar(value=10.0)
        self.cm_per_px_var = tk.StringVar(value="Not calibrated")

        for row, (label, var, minimum, maximum) in enumerate(
            [
                ("Hue min", self.h_min, 0, 179),
                ("Hue max", self.h_max, 0, 179),
                ("Saturation min", self.s_min, 0, 255),
                ("Saturation max", self.s_max, 0, 255),
                ("Value min", self.v_min, 0, 255),
                ("Value max", self.v_max, 0, 255),
            ]
        ):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
            scale = ttk.Scale(parent, from_=minimum, to=maximum, variable=var, command=lambda _v: self.refresh_frame())
            scale.grid(row=row, column=1, sticky="ew", padx=6)
            ttk.Label(parent, textvariable=var, width=4).grid(row=row, column=2)

        parent.columnconfigure(1, weight=1)
        row = 6
        ttk.Label(parent, text="Min. area [px]").grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(parent, from_=1, to=100000, textvariable=self.min_area_var, width=10, command=self.refresh_frame).grid(
            row=row, column=1, sticky="w", pady=(10, 0)
        )
        row += 1
        ttk.Label(parent, text="Top percentile [%]").grid(row=row, column=0, sticky="w")
        ttk.Spinbox(parent, from_=0, to=50, increment=0.5, textvariable=self.percentile_var, width=10, command=self.refresh_frame).grid(
            row=row, column=1, sticky="w"
        )
        row += 1
        ttk.Separator(parent).grid(row=row, column=0, columnspan=3, sticky="ew", pady=10)
        row += 1
        ttk.Label(parent, text="Calibration: click two points in the frame.").grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1
        ttk.Label(parent, text="Distance [cm]").grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self.distance_cm_var, width=10).grid(row=row, column=1, sticky="w")
        row += 1
        ttk.Button(parent, text="Compute Calibration", command=self.compute_calibration).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        ttk.Label(parent, textvariable=self.cm_per_px_var).grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _build_formula_controls(self, parent: ttk.Frame) -> None:
        """Creates the formula and constants editor."""
        ttk.Label(parent, text="Formula where H is flame height [cm]").pack(anchor="w")
        self.formula_var = tk.StringVar(value="2.5 * H**0.67")
        ttk.Entry(parent, textvariable=self.formula_var).pack(fill=tk.X, pady=(2, 10))
        ttk.Label(parent, text="Constants, e.g. a=2.5, b=1.0").pack(anchor="w")
        self.constants_text = tk.Text(parent, height=5, wrap=tk.WORD)
        self.constants_text.pack(fill=tk.X)
        ttk.Button(parent, text="Test Formula", command=self.test_formula).pack(anchor="w", pady=10)
        self.formula_status = tk.StringVar(value="Formula ready.")
        ttk.Label(parent, textvariable=self.formula_status, wraplength=420).pack(anchor="w")

    def _build_results_panel(self, parent: ttk.Frame) -> None:
        """Creates charts, progress bar, and statistics panel."""
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill=tk.X)
        self.plot_container = ttk.Frame(parent)
        self.plot_container.pack(fill=tk.BOTH, expand=True, pady=8)
        ttk.Label(
            self.plot_container,
            text="Charts will appear after analysis completes.",
            anchor=tk.CENTER,
        ).pack(fill=tk.BOTH, expand=True)
        self.stats_text = tk.StringVar(value="No results available yet.")
        ttk.Label(parent, textvariable=self.stats_text, justify=tk.LEFT, wraplength=460).pack(fill=tk.X, anchor="w")

    def _ensure_plot_panel(self) -> None:
        """Initialises Matplotlib only when charts are needed."""
        if self.plot_panel is not None:
            return
        for child in self.plot_container.winfo_children():
            child.destroy()
        from visualizer import PlotPanel

        self.plot_panel = PlotPanel(self.plot_container)
        self.plot_panel.widget.pack(fill=tk.BOTH, expand=True)

    def _load_sample_if_available(self) -> None:
        """Automatically loads the first video found in the videos/ folder."""
        for pattern in ("*.mp4", "*.avi", "*.mov", "*.mts", "*.MTS"):
            matches = sorted(Path("videos").glob(pattern))
            if matches:
                try:
                    self.status_var.set(f"Loading sample video: {matches[0]}")
                    self.update_idletasks()
                    self.load_video(matches[0])
                    self.status_var.set(f"Video loaded: {matches[0]}")
                except Exception as exc:  # noqa: BLE001
                    self.status_var.set("Failed to load sample video.")
                    messagebox.showwarning("Sample Video", str(exc))
                return
        self.status_var.set("Ready. Select a video using the Open Video button.")

    def open_video(self) -> None:
        """Opens a file dialog to select a video."""
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mts *.MTS"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.load_video(Path(path))

    def load_video(self, path: Path) -> None:
        """Loads a video and displays the first frame."""
        self.status_var.set(f"Loading video: {path}")
        self.update_idletasks()
        info = self.video.open(path)
        self.frame_slider.configure(to=max(0, info.frame_count - 1))
        self.frame_slider.set(0)
        self.calibration.reset_points()
        self.cm_per_px_var.set("Not calibrated")
        self.show_frame(0)
        self.status_var.set(f"Loaded: {path.name} | {info.frame_count} frames | {info.fps:.2f} FPS")

    def _sync_analyzer_from_controls(self) -> None:
        """Transfers control values to the analyser."""
        self.analyzer.thresholds = HSVThresholds(
            int(self.h_min.get()),
            int(self.h_max.get()),
            int(self.s_min.get()),
            int(self.s_max.get()),
            int(self.v_min.get()),
            int(self.v_max.get()),
        )
        self.analyzer.min_area_px = int(self.min_area_var.get())
        self.analyzer.top_percentile = float(self.percentile_var.get())

    def show_frame(self, frame_index: int) -> None:
        """Loads, analyses, and renders a frame."""
        try:
            self._sync_analyzer_from_controls()
            frame = self.video.read_frame(frame_index)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Video", str(exc))
            return

        self.current_frame = frame
        self.current_measurement = self.analyzer.measure_frame(frame, self.calibration.cm_per_px)
        annotated = self.calibration.draw(self.analyzer.draw_annotation(frame, self.current_measurement))
        mask_preview = self.current_measurement.mask

        self._set_image(self.original_label, annotated, "photo_original")
        self._set_image(self.mask_label, cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR), "photo_mask")
        self.frame_slider.set(frame_index)
        self.frame_label.configure(
            text=f"Frame: {frame_index + 1}/{self.video.info.frame_count} | Time: {self.video.frame_time_s(frame_index):.3f} s"
        )

    def _set_image(self, label: ttk.Label, frame_bgr: np.ndarray, attr_name: str) -> None:
        """Scales the frame for the GUI and stores the PhotoImage to prevent GC."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_width = 520
        max_height = 520
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        if attr_name == "photo_original":
            self.display_scale = scale
            self.original_display_size = (int(image.width * scale), int(image.height * scale)) if scale >= 1.0 else image.size
        if scale < 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
            if attr_name == "photo_original":
                self.original_display_size = image.size
        photo = ImageTk.PhotoImage(image)
        setattr(self, attr_name, photo)
        label.configure(image=photo)

    def refresh_frame(self) -> None:
        """Re-renders the current frame after a parameter change."""
        if self.video.info is not None:
            self.show_frame(self.video.current_frame_index)

    def on_slider(self, value: str) -> None:
        """Responds to frame slider movement."""
        if self.video.info is None or self.analysis_running:
            return
        frame_index = int(float(value))
        if frame_index != self.video.current_frame_index:
            self.show_frame(frame_index)

    def step_frame(self, direction: int) -> None:
        """Steps the video one frame forward or backward."""
        if self.video.info is None:
            return
        self.show_frame(self.video.current_frame_index + direction)

    def toggle_playback(self) -> None:
        """Starts or stops video playback."""
        if self.video.info is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.after(1, self._play_next_frame)

    def _play_next_frame(self) -> None:
        """Advances playback by one frame, respecting FPS."""
        if not self.is_playing or self.video.info is None:
            return
        next_frame = self.video.current_frame_index + 1
        if next_frame >= self.video.info.frame_count:
            self.is_playing = False
            return
        self.show_frame(next_frame)
        delay_ms = max(1, int(1000 / self.video.info.fps))
        self.after(delay_ms, self._play_next_frame)

    def on_image_click(self, event) -> None:
        """Adds a calibration point from a click in the frame preview."""
        if self.current_frame is None:
            return
        display_w, display_h = self.original_display_size
        offset_x = max(0, (event.widget.winfo_width() - display_w) // 2)
        offset_y = max(0, (event.widget.winfo_height() - display_h) // 2)
        x = int((event.x - offset_x) / self.display_scale)
        y = int((event.y - offset_y) / self.display_scale)
        if x < 0 or y < 0 or x >= self.current_frame.shape[1] or y >= self.current_frame.shape[0]:
            return
        h, w = self.current_frame.shape[:2]
        self.calibration.add_point((max(0, min(x, w - 1)), max(0, min(y, h - 1))))
        self.refresh_frame()

    def compute_calibration(self) -> None:
        """Computes and displays the calibration factor."""
        try:
            factor = self.calibration.compute(float(self.distance_cm_var.get()))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Calibration", str(exc))
            return
        self.cm_per_px_var.set(f"cm_per_px = {factor:.6g}")
        self.refresh_frame()

    def _update_formula_engine(self) -> None:
        """Reads formula and constants from the GUI."""
        self.formula_engine.expression = self.formula_var.get().strip()
        self.formula_engine.set_constants_from_text(self.constants_text.get("1.0", tk.END))

    def test_formula(self) -> None:
        """Tests the formula against the current or a model height."""
        try:
            self._update_formula_engine()
            height = 10.0
            if self.current_measurement is not None and not np.isnan(self.current_measurement.height_cm):
                height = self.current_measurement.height_cm
            value = self.formula_engine.evaluate(height)
            self.formula_status.set(f"OK: H={height:.4g} cm → {value:.4g}")
        except FormulaError as exc:
            self.formula_status.set(str(exc))

    def start_analysis(self) -> None:
        """Starts batch analysis in a background thread."""
        if self.video.info is None:
            messagebox.showwarning("Analysis", "Please load a video first.")
            return
        if self.calibration.cm_per_px is None:
            messagebox.showwarning("Analysis", "Please calibrate the scale first.")
            return
        try:
            self._update_formula_engine()
            self.formula_engine.evaluate(10.0)
        except FormulaError as exc:
            messagebox.showerror("Formula", str(exc))
            return

        self.analysis_running = True
        self.progress.configure(value=0, maximum=100)
        thread = threading.Thread(target=self._analysis_worker, daemon=True)
        thread.start()

    def _analysis_worker(self) -> None:
        """Runs batch analysis outside the main GUI thread."""
        try:
            frame_step = int(self.frame_step_var.get())
            thresholds = HSVThresholds(
                self.analyzer.thresholds.hue_min,
                self.analyzer.thresholds.hue_max,
                self.analyzer.thresholds.sat_min,
                self.analyzer.thresholds.sat_max,
                self.analyzer.thresholds.val_min,
                self.analyzer.thresholds.val_max,
            )
            analyzer = FlameAnalyzer(thresholds, self.analyzer.min_area_px, self.analyzer.top_percentile)
            formula_engine = FormulaEngine(self.formula_engine.expression, self.formula_engine.constants.copy())

            def progress(done: int, total: int) -> None:
                self.after(0, lambda: self.progress.configure(value=100 * done / max(1, total)))

            results = analyze_video(
                self.video,
                analyzer,
                self.calibration.cm_per_px or 1.0,
                formula_engine,
                frame_step,
                progress,
            )
            self.after(0, lambda: self._analysis_finished(results))
        except Exception as exc:  # noqa: BLE001
            self.after(0, lambda error=exc: messagebox.showerror("Analysis", str(error)))
            self.after(0, self._analysis_cleanup)

    def _analysis_finished(self, results: pd.DataFrame) -> None:
        """Updates the GUI after analysis completes."""
        from visualizer import build_stats_text

        self.results = results
        self._ensure_plot_panel()
        self.plot_panel.draw_results(results)
        self.stats_text.set(build_stats_text(results))
        self._analysis_cleanup()
        messagebox.showinfo("Analysis", "Video analysis completed.")

    def _analysis_cleanup(self) -> None:
        """Clears the running-analysis state."""
        self.analysis_running = False

    def _show_window(self) -> None:
        """Shows the window after layout is built (Tk Aqua workaround)."""
        try:
            self.attributes("-alpha", 0.0)
        except Exception:
            pass
        self.deiconify()
        self.lift()
        self.update_idletasks()
        try:
            self.attributes("-alpha", 1.0)
        except Exception:
            pass
        if not self.winfo_exists():
            return
        try:
            w, h = self.winfo_width(), self.winfo_height()
            if w > 100:
                self.geometry(f"{w + 2}x{h}")
                self.update_idletasks()
                if self.winfo_exists():
                    self.geometry(f"{w}x{h}")
        except Exception:
            pass

    def export_results(self) -> None:
        """Exports batch analysis results to CSV."""
        if self.results.empty:
            messagebox.showwarning("Export", "Please run video analysis first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_csv(self.results, path)
            messagebox.showinfo("Export", f"CSV saved:\n{path}")

    def export_frame(self) -> None:
        """Exports the current annotated frame."""
        if self.current_frame is None or self.current_measurement is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            export_annotated_frame(self.current_frame, self.current_measurement, self.analyzer, path)
            messagebox.showinfo("Export", f"Frame saved:\n{path}")

    def export_video(self) -> None:
        """Exports the annotated video."""
        if self.video.info is None or self.calibration.cm_per_px is None:
            messagebox.showwarning("Export", "Please load a video and complete calibration first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if not path:
            return
        start = time.time()
        try:
            export_annotated_video(self.video, self.analyzer, self.calibration.cm_per_px, path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export", str(exc))
            return
        messagebox.showinfo("Export", f"Video saved in {time.time() - start:.1f} s:\n{path}")


if __name__ == "__main__":
    if "--legacy-ui" in sys.argv:
        # ttk version — works better with newer Tk (8.6+)
        app = FlameAnalysisApp()
    else:
        from simple_app import SimpleFlameAnalysisApp
        app = SimpleFlameAnalysisApp()
    app.mainloop()
