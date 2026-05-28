"""Plain Tkinter GUI without ttk, compatible with macOS system Tk and Windows.

Layout uses ``pack`` exclusively. On first show the window is briefly hidden and
re-displayed to force a clean Expose pass (Tk 8.5 Aqua workaround).
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from analyzer import FlameAnalyzer, HSVThresholds, analyze_video
from calibration import Calibration
from exporter import export_annotated_frame, export_annotated_video, export_csv
from formula_engine import FormulaEngine, FormulaError
from video_processor import VideoProcessor

VERSION = "0.1.0"

BG = "#f4f4f4"
BG_BAR = "#ececec"
BG_SECTION = "#dcdcdc"
BG_STATUS = "#e8e8e8"
RIGHT_PANEL_WIDTH = 400

PRESETS: dict[str, dict] = {
    "Orange flame (default)": dict(
        h_min=0, h_max=35, s_min=100, s_max=255, v_min=150, v_max=255,
        min_area=500, percentile=5.0, kernel=5, dual=False,
    ),
    "Bright flame (shadow suppressed)": dict(
        h_min=0, h_max=25, s_min=160, s_max=255, v_min=190, v_max=255,
        min_area=500, percentile=5.0, kernel=5, dual=False,
    ),
    "Yellow / white flame": dict(
        h_min=15, h_max=60, s_min=60, s_max=255, v_min=200, v_max=255,
        min_area=300, percentile=5.0, kernel=5, dual=False,
    ),
    "Blue flame (gas)": dict(
        h_min=90, h_max=135, s_min=80, s_max=255, v_min=100, v_max=255,
        min_area=200, percentile=5.0, kernel=3, dual=False,
    ),
    "Red-orange (dual range)": dict(
        h_min=160, h_max=25, s_min=120, s_max=255, v_min=150, v_max=255,
        min_area=500, percentile=5.0, kernel=5, dual=True,
        white_core=False, white_val_min=220, white_sat_max=80, merge_kernel=0, select_by="bottom",
    ),
    "Large fire (white core + merge)": dict(
        h_min=0, h_max=35, s_min=80, s_max=255, v_min=140, v_max=255,
        min_area=500, percentile=5.0, kernel=5, dual=False,
        white_core=True, white_val_min=210, white_sat_max=90, merge_kernel=30, select_by="area",
    ),
}


class SimpleFlameAnalysisApp(tk.Tk):
    """Main application built with plain Tk widgets only."""

    def __init__(self) -> None:
        super().__init__()
        # Hide the window immediately before the first show. deiconify() below
        # triggers the first clean Expose pass with the fully-built layout
        # (Tk 8.5 Aqua blank-window workaround).
        self.withdraw()

        self.title("Flame Height Analysis")
        self.geometry("1380x860")
        self.minsize(1100, 720)
        self.configure(bg=BG)

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

        self._build_ui()
        self.status_var.set("Ready. Open a video using the Open Video button.")

        self.after(100, self._show_window)
        self.after(700, self._load_sample_if_available)

    # ------------------------------------------------------------------
    # Sestavení rozhraní (pouze pack)
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Builds the plain-Tk interface using pack layout."""
        # Pack toolbar and status bar first so they reserve fixed space;
        # the content area gets the remaining room.
        toolbar = tk.Frame(self, bg=BG_BAR, padx=6, pady=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self._build_toolbar(toolbar)

        status = tk.Label(
            self,
            textvariable=self.status_var,
            bg=BG_STATUS,
            anchor="w",
            padx=8,
            pady=5,
        )
        status.pack(side=tk.BOTTOM, fill=tk.X)

        content = tk.Frame(self, bg=BG)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Right panel has a fixed width; left panel fills the rest.
        right = tk.Frame(content, bg=BG, width=RIGHT_PANEL_WIDTH)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)
        self._build_right_panel(right)

        left = tk.Frame(content, bg=BG, padx=8, pady=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_left_panel(left)

    def _build_toolbar(self, bar: tk.Frame) -> None:
        """Top toolbar with action buttons."""
        self.frame_step_var = tk.IntVar(value=5)

        self._button(bar, "Open Video", self.open_video).pack(side=tk.LEFT, padx=3)
        self._button(bar, "Play", self.toggle_playback).pack(side=tk.LEFT, padx=3)
        self._button(bar, "< Frame", lambda: self.step_frame(-1)).pack(side=tk.LEFT, padx=3)
        self._button(bar, "Frame >", lambda: self.step_frame(1)).pack(side=tk.LEFT, padx=3)
        tk.Label(bar, text="Frame step", bg=BG_BAR).pack(side=tk.LEFT, padx=(14, 3))
        tk.Spinbox(bar, from_=1, to=500, textvariable=self.frame_step_var, width=6).pack(side=tk.LEFT)
        self._button(bar, "Analyse Video", self.start_analysis).pack(side=tk.LEFT, padx=(14, 3))
        self._button(bar, "Export CSV", self.export_results).pack(side=tk.LEFT, padx=3)
        self._button(bar, "Export Frame", self.export_frame).pack(side=tk.LEFT, padx=3)
        self._button(bar, "Export Video", self.export_video).pack(side=tk.LEFT, padx=3)
        # Help / About buttons on the right
        self._button(bar, "About", self._show_about).pack(side=tk.RIGHT, padx=(3, 6))
        self._button(bar, "Help", self._show_help).pack(side=tk.RIGHT, padx=3)

    def _build_left_panel(self, left: tk.Frame) -> None:
        """Left area: two video previews + frame slider."""
        # Pack slider at the bottom first; previews get the remaining space.
        slider_area = tk.Frame(left, bg=BG)
        slider_area.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        self.frame_slider = tk.Scale(
            slider_area,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self.on_slider,
            showvalue=False,
            bg=BG,
            highlightthickness=0,
        )
        self.frame_slider.pack(fill=tk.X)
        self.frame_label_var = tk.StringVar(value="Frame: - | Time: -")
        tk.Label(slider_area, textvariable=self.frame_label_var, bg=BG, anchor="w").pack(fill=tk.X)

        video_area = tk.Frame(left, bg=BG)
        video_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        original_panel = tk.Frame(video_area, bg=BG)
        original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        tk.Label(original_panel, text="Original Frame", bg=BG, anchor="w").pack(side=tk.TOP, fill=tk.X)
        self.original_label = tk.Label(original_panel, bg="white", relief=tk.SOLID, bd=1)
        self.original_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.original_label.bind("<Button-1>", self.on_image_click)

        mask_panel = tk.Frame(video_area, bg=BG)
        mask_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(mask_panel, text="HSV Mask", bg=BG, anchor="w").pack(side=tk.TOP, fill=tk.X)
        self.mask_label = tk.Label(mask_panel, bg="white", relief=tk.SOLID, bd=1)
        self.mask_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_right_panel(self, right: tk.Frame) -> None:
        """Scrollable right control panel (Canvas, Tk 8.6+)."""
        scrollbar = tk.Scrollbar(right, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas = tk.Canvas(right, bg=BG, highlightthickness=0, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)

        inner = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))

        def _scroll(event: tk.Event) -> None:
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        right.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", _scroll))
        right.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

        self._build_controls(inner)

    def _build_controls(self, parent: tk.Frame) -> None:
        """Builds the scrollable right-panel content."""
        self.preset_var = tk.StringVar(value=list(PRESETS.keys())[0])
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
        self.analysis_start_var = tk.IntVar(value=0)
        self.analysis_end_var = tk.IntVar(value=0)
        self.range_info_var = tk.StringVar(value="")
        self.kernel_size_var = tk.IntVar(value=5)
        self.use_dual_range_var = tk.BooleanVar(value=False)
        self.include_white_core_var = tk.BooleanVar(value=False)
        self.white_val_min_var = tk.IntVar(value=220)
        self.white_sat_max_var = tk.IntVar(value=80)
        self.merge_kernel_var = tk.IntVar(value=0)
        self.select_by_var = tk.StringVar(value="bottom")

        # --- Presets ---
        self._section(parent, "Preset")
        preset_row = self._row(parent)
        menu = tk.OptionMenu(
            preset_row, self.preset_var, *PRESETS.keys(),
            command=lambda _: self._apply_preset(),
        )
        menu.config(anchor="w", highlightbackground=BG)
        menu["menu"].config(bg=BG)
        menu.pack(fill=tk.X, expand=True)

        # --- Flame Detection ---
        self._section(parent, "Flame Detection")
        for label, var, minimum, maximum in [
            ("Hue min", self.h_min, 0, 179),
            ("Hue max", self.h_max, 0, 179),
            ("Saturation min", self.s_min, 0, 255),
            ("Saturation max", self.s_max, 0, 255),
            ("Value min", self.v_min, 0, 255),
            ("Value max", self.v_max, 0, 255),
        ]:
            self._slider_row(parent, label, var, minimum, maximum)

        spin_row = self._row(parent)
        tk.Label(spin_row, text="Min. area [px]", bg=BG, width=16, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(spin_row, from_=1, to=100000, textvariable=self.min_area_var, width=10, command=self.refresh_frame).pack(side=tk.LEFT)

        pct_row = self._row(parent)
        tk.Label(pct_row, text="Top percentile [%]", bg=BG, width=16, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(pct_row, from_=0, to=50, increment=0.5, textvariable=self.percentile_var, width=10, command=self.refresh_frame).pack(side=tk.LEFT)

        # --- Analysis Range ---
        self._section(parent, "Analysis Range")
        start_row = self._row(parent)
        tk.Label(start_row, text="From frame", bg=BG, width=12, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(start_row, from_=0, to=9999999, textvariable=self.analysis_start_var,
                   width=9, command=self._update_range_info).pack(side=tk.LEFT)
        self._button(start_row, "←", self._set_range_start).pack(side=tk.LEFT, padx=2)
        end_row = self._row(parent)
        tk.Label(end_row, text="To frame", bg=BG, width=12, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(end_row, from_=0, to=9999999, textvariable=self.analysis_end_var,
                   width=9, command=self._update_range_info).pack(side=tk.LEFT)
        self._button(end_row, "←", self._set_range_end).pack(side=tk.LEFT, padx=2)
        tk.Label(parent, textvariable=self.range_info_var, bg=BG, anchor="w",
                 wraplength=RIGHT_PANEL_WIDTH - 40).pack(fill=tk.X, padx=6)

        # --- Calibration ---
        self._section(parent, "Calibration")
        tk.Label(parent, text="Click two points in the image.", bg=BG, anchor="w").pack(fill=tk.X, padx=6)
        cal_row = self._row(parent)
        tk.Label(cal_row, text="Distance [cm]", bg=BG, width=16, anchor="w").pack(side=tk.LEFT)
        tk.Entry(cal_row, textvariable=self.distance_cm_var, width=12).pack(side=tk.LEFT)
        self._button(parent, "Compute Calibration", self.compute_calibration).pack(anchor="w", padx=6, pady=3)
        tk.Label(parent, textvariable=self.cm_per_px_var, bg=BG, anchor="w").pack(fill=tk.X, padx=6)

        # --- Formula ---
        self._section(parent, "Formula")
        tk.Label(parent, text="Formula where H is flame height [cm]", bg=BG, anchor="w").pack(fill=tk.X, padx=6)
        self.formula_var = tk.StringVar(value="2.5 * H**0.67")
        tk.Entry(parent, textvariable=self.formula_var).pack(fill=tk.X, padx=6, pady=(2, 6))
        tk.Label(parent, text="Constants, e.g. a=2.5, b=1.0", bg=BG, anchor="w").pack(fill=tk.X, padx=6)
        self.constants_text = tk.Text(parent, height=3)
        self.constants_text.pack(fill=tk.X, padx=6, pady=(2, 4))
        self._button(parent, "Test Formula", self.test_formula).pack(anchor="w", padx=6, pady=3)
        self.formula_status = tk.StringVar(value="Formula ready.")
        tk.Label(parent, textvariable=self.formula_status, bg=BG, wraplength=RIGHT_PANEL_WIDTH - 40, justify=tk.LEFT, anchor="w").pack(fill=tk.X, padx=6)

        # --- Advanced Detection (add-on) ---
        self._section(parent, "Advanced Detection")

        # White-hot core
        wc_row = self._row(parent)
        tk.Checkbutton(wc_row, text="White-hot flame core", variable=self.include_white_core_var,
                       bg=BG, command=self.refresh_frame).pack(side=tk.LEFT)
        wc_params = self._row(parent)
        tk.Label(wc_params, text="  Val min", bg=BG, width=10, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(wc_params, from_=150, to=255, textvariable=self.white_val_min_var,
                   width=5, command=self.refresh_frame).pack(side=tk.LEFT)
        tk.Label(wc_params, text="Sat max", bg=BG, anchor="w", padx=6).pack(side=tk.LEFT)
        tk.Spinbox(wc_params, from_=0, to=150, textvariable=self.white_sat_max_var,
                   width=5, command=self.refresh_frame).pack(side=tk.LEFT)

        # Merge fragments
        merge_row = self._row(parent)
        tk.Label(merge_row, text="Merge fragments [px]", bg=BG, width=16, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(merge_row, from_=0, to=150, increment=5, textvariable=self.merge_kernel_var,
                   width=6, command=self.refresh_frame).pack(side=tk.LEFT)
        tk.Label(merge_row, text="0=off", bg=BG, anchor="w", padx=4).pack(side=tk.LEFT)

        # Kernel
        kernel_row = self._row(parent)
        tk.Label(kernel_row, text="Kernel [px]", bg=BG, width=16, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(kernel_row, from_=1, to=21, increment=2, textvariable=self.kernel_size_var,
                   width=6, command=self.refresh_frame).pack(side=tk.LEFT)
        tk.Label(kernel_row, text="(odd)", bg=BG, anchor="w", padx=4).pack(side=tk.LEFT)

        # Dual range
        dual_row = self._row(parent)
        tk.Checkbutton(dual_row, text="Dual hue range (H=0/179)", variable=self.use_dual_range_var,
                       bg=BG, command=self.refresh_frame).pack(side=tk.LEFT)
        tk.Label(parent, text="Hue min = upper part, Hue max = lower part", bg=BG,
                 anchor="w", wraplength=RIGHT_PANEL_WIDTH - 40,
                 font=("TkDefaultFont", 9)).pack(fill=tk.X, padx=6)

        # Select-by strategy
        sel_row = self._row(parent)
        tk.Label(sel_row, text="Component select", bg=BG, width=16, anchor="w").pack(side=tk.LEFT)
        sel_menu = tk.OptionMenu(sel_row, self.select_by_var, "bottom", "area",
                                 command=lambda _: self.refresh_frame())
        sel_menu.config(highlightbackground=BG)
        sel_menu.pack(side=tk.LEFT)
        tk.Label(parent, text="bottom = lowest base  |  area = largest area",
                 bg=BG, anchor="w", wraplength=RIGHT_PANEL_WIDTH - 40,
                 font=("TkDefaultFont", 9)).pack(fill=tk.X, padx=6, pady=(0, 4))

        # --- Results ---
        self._section(parent, "Results")
        self.progress_var = tk.StringVar(value="Progress: 0 %")
        tk.Label(parent, textvariable=self.progress_var, bg=BG, anchor="w").pack(fill=tk.X, padx=6)
        self.stats_text = tk.StringVar(value="No results available yet.")
        tk.Label(parent, textvariable=self.stats_text, bg=BG, justify=tk.LEFT, wraplength=RIGHT_PANEL_WIDTH - 40, anchor="w").pack(fill=tk.X, padx=6, pady=(0, 8))

    # ------------------------------------------------------------------
    # Pomocné prvky layoutu
    # ------------------------------------------------------------------
    def _button(self, parent, text: str, command) -> tk.Button:
        """Vrátí jednotně stylované tlačítko."""
        return tk.Button(parent, text=text, command=command, padx=8, pady=3, highlightbackground=BG)

    def _row(self, parent: tk.Frame) -> tk.Frame:
        """Vodorovný řádek roztažený na šířku panelu."""
        row = tk.Frame(parent, bg=BG)
        row.pack(fill=tk.X, padx=6, pady=1)
        return row

    def _slider_row(self, parent: tk.Frame, label: str, var, minimum: int, maximum: int) -> None:
        """Řádek s popiskem, posuvníkem a hodnotou."""
        row = self._row(parent)
        tk.Label(row, text=label, bg=BG, width=14, anchor="w").pack(side=tk.LEFT)
        tk.Scale(
            row,
            from_=minimum,
            to=maximum,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda _v: self.refresh_frame(),
            bg=BG,
            highlightthickness=0,
            showvalue=False,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(row, textvariable=var, bg=BG, width=4, anchor="e").pack(side=tk.LEFT)

    def _section(self, parent: tk.Frame, title: str) -> None:
        """Vloží nadpis sekce roztažený na šířku."""
        tk.Label(
            parent,
            text=title,
            bg=BG_SECTION,
            anchor="w",
            padx=6,
            pady=3,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(fill=tk.X, padx=6, pady=(6, 2))

    def _show_about(self) -> None:
        """Shows the About dialog with author and version information."""
        import webbrowser

        win = tk.Toplevel(self)
        win.title("About")
        win.geometry("380x240")
        win.resizable(False, False)
        win.configure(bg=BG)
        win.grab_set()

        tk.Label(
            win, text="Flame Height Analysis",
            bg=BG, font=("TkDefaultFont", 16, "bold"),
        ).pack(pady=(24, 4))
        tk.Label(win, text=f"Version {VERSION}", bg=BG,
                 font=("TkDefaultFont", 11)).pack()

        tk.Label(win, text="", bg=BG).pack()

        tk.Label(win, text="Author:", bg=BG,
                 font=("TkDefaultFont", 10, "bold")).pack()

        site = tk.Label(win, text="hvck.cz", bg=BG, fg="#0969da",
                        cursor="hand2", font=("TkDefaultFont", 10, "underline"))
        site.pack()
        site.bind("<Button-1>", lambda _e: webbrowser.open("https://hvck.cz"))

        orcid = tk.Label(win, text="ORCID: 0000-0002-1276-2702", bg=BG, fg="#0969da",
                         cursor="hand2", font=("TkDefaultFont", 10, "underline"))
        orcid.pack()
        orcid.bind("<Button-1>", lambda _e: webbrowser.open("https://orcid.org/0000-0002-1276-2702"))

        tk.Label(win, text="", bg=BG).pack()
        tk.Button(win, text="Close", command=win.destroy, padx=14, pady=3,
                  highlightbackground=BG).pack(pady=(0, 18))

    _HELP_TEXT = """\
FLAME HEIGHT ANALYSIS — User Guide
===================================

1. OPEN A VIDEO
   • Click "Open Video" and select an .mp4, .avi, .mov or .mts file.
   • The first frame appears automatically in both preview panels.
   • Use the slider at the bottom to browse frames.
     "< Frame" / "Frame >" step one frame at a time.
     "Play" plays back at the video's native FPS.

2. CALIBRATE THE SCALE
   • Click two points on a known distance in the left preview
     (e.g. top and bottom of a ruler or a reference object).
   • Two red dots appear, connected by a line.
   • Enter the real-world distance in the "Distance [cm]" field.
   • Click "Compute Calibration". The result is shown as cm_per_px.

3. CHOOSE A PRESET (optional)
   • The "Preset" dropdown provides ready-made HSV settings for
     common flame types:
       – Orange flame (default)
       – Bright flame (shadow suppressed)
       – Yellow / white flame
       – Blue flame (gas)
       – Red-orange (dual range)
       – Large fire (white core + merge)
   • Selecting a preset overwrites all detection sliders.

4. TUNE FLAME DETECTION
   • Use the Hue / Saturation / Value sliders to isolate the flame
     colour in the HSV mask preview on the right side of the screen.
   • The green line in the left panel shows the detected flame height.
   • "Min. area [px]" — ignore detections smaller than this.
   • "Top percentile [%]" — how many top pixels define the flame tip
     (lower = higher, more precise tip; higher = more conservative).

5. ADVANCED DETECTION (optional add-on)
   • White-hot flame core — captures overexposed white centres that
     lose colour saturation.  Tune "Val min" and "Sat max".
   • Merge fragments [px] — dilates the mask before region selection
     to bridge small gaps. 0 = off.
   • Kernel [px] — morphological open/close kernel size (odd numbers).
   • Dual hue range — splits the hue range around H=0/179 for red
     or deep-red flames.
   • Component select:
       bottom — picks the region whose base is lowest in the frame.
       area   — picks the largest connected region.

6. SET THE ANALYSIS RANGE
   • Navigate to the desired start frame, then click "←" next to
     "From frame" to set the start.
   • Do the same for the end frame.
   • The info label shows the time range and estimated frame count.

7. ENTER A FORMULA (optional)
   • The formula is evaluated for every analysed frame.
     H is the measured flame height in cm.
   • Default: 2.5 * H**0.67
   • You can define constants in the "Constants" box, e.g.:
       a=2.5
       n=0.67
     Then use them in the formula: a * H**n
   • Click "Test Formula" to verify against the current frame.

8. RUN THE ANALYSIS
   • Set "Frame step" — every Nth frame is analysed.
     Step 5 = every 5th frame; step 1 = every frame.
   • Click "Analyse Video". A progress indicator shows completion.
   • After analysis a chart window opens with height and heat value
     plotted over time.
   • Statistics (mean, median, min, max, std dev) appear in the
     "Results" section of the right panel.

9. EXPORT
   • Export CSV   — saves all frame-by-frame measurements.
   • Export Frame — saves the current annotated frame as PNG/JPEG.
   • Export Video — renders the full annotated video as MP4.

TIPS
----
• If the mask includes shadows, increase "Saturation min" or use the
  "Bright flame (shadow suppressed)" preset.
• For fragmented masks on large fires, enable "White-hot flame core"
  and set "Merge fragments" to 20–40 px, then switch "Component
  select" to "area".
• For red flames near H=0/179, enable "Dual hue range".
"""

    def _show_help(self) -> None:
        """Shows the user guide in a scrollable window."""
        win = tk.Toplevel(self)
        win.title("User Guide")
        win.geometry("580x560")
        win.configure(bg=BG)

        frame = tk.Frame(win, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 4))

        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text = tk.Text(
            frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            bg="white",
            relief=tk.FLAT,
            padx=10,
            pady=8,
            font=("TkDefaultFont", 11),
            state=tk.NORMAL,
        )
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)

        text.insert(tk.END, self._HELP_TEXT)
        text.config(state=tk.DISABLED)

        # Mousewheel scrolling inside the help window
        def _scroll_help(event: tk.Event) -> None:
            text.yview_scroll(-1 if event.delta > 0 else 1, "units")
        text.bind("<MouseWheel>", _scroll_help)

        tk.Button(win, text="Close", command=win.destroy, padx=14, pady=3,
                  highlightbackground=BG).pack(pady=(0, 10))

    def _show_window(self) -> None:
        """Shows the window after layout is complete.
        Contains workarounds for macOS Tk 8.5 Aqua where non-native widgets
        do not receive an Expose event on the initial window map.
        On Tk 8.6+ (Python 3.11 via python-tk) everything works without tricks.
        """
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

    def _load_sample_if_available(self) -> None:
        """Automatically loads the first video found in the videos/ folder."""
        for pattern in ("*.mp4", "*.avi", "*.mov", "*.mts", "*.MTS"):
            matches = sorted(Path("videos").glob(pattern))
            if matches:
                try:
                    self.load_video(matches[0])
                except Exception:
                    pass
                return

    # ------------------------------------------------------------------
    # Logika aplikace
    # ------------------------------------------------------------------
    def open_video(self) -> None:
        """Opens a video file via file dialog."""
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mts *.MTS"), ("All files", "*.*")],
        )
        if path:
            self.load_video(Path(path))

    def load_video(self, path: Path) -> None:
        """Loads a video and displays the first frame."""
        try:
            self.status_var.set(f"Loading video: {path}")
            self.update_idletasks()
            info = self.video.open(path)
            self.frame_slider.configure(to=max(0, info.frame_count - 1))
            self.calibration.reset_points()
            self.cm_per_px_var.set("Not calibrated")
            self.analysis_start_var.set(0)
            self.analysis_end_var.set(max(0, info.frame_count - 1))
            self._update_range_info()
            self.show_frame(0)
            self.status_var.set(f"Loaded: {path.name} | {info.frame_count} frames | {info.fps:.2f} FPS")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Video", str(exc))
            self.status_var.set("Failed to load video.")

    def _sync_analyzer_from_controls(self) -> None:
        """Přenese hodnoty z ovládání do analyzátoru."""
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
        self.analyzer.kernel_size = int(self.kernel_size_var.get())
        self.analyzer.use_dual_range = bool(self.use_dual_range_var.get())
        self.analyzer.include_white_core = bool(self.include_white_core_var.get())
        self.analyzer.white_val_min = int(self.white_val_min_var.get())
        self.analyzer.white_sat_max = int(self.white_sat_max_var.get())
        self.analyzer.merge_kernel_size = int(self.merge_kernel_var.get())
        self.analyzer.select_by = self.select_by_var.get()

    def show_frame(self, frame_index: int) -> None:
        """Zobrazí snímek a jeho HSV masku."""
        if self.video.info is None:
            return
        try:
            self._sync_analyzer_from_controls()
            frame = self.video.read_frame(frame_index)
            self.current_frame = frame
            self.current_measurement = self.analyzer.measure_frame(frame, self.calibration.cm_per_px)
            annotated = self.calibration.draw(self.analyzer.draw_annotation(frame, self.current_measurement))
            self._set_image(self.original_label, annotated, "photo_original")
            self._set_image(self.mask_label, cv2.cvtColor(self.current_measurement.mask, cv2.COLOR_GRAY2BGR), "photo_mask")
            self.frame_slider.set(frame_index)
            self.frame_label_var.set(f"Frame: {frame_index + 1}/{self.video.info.frame_count} | Time: {self.video.frame_time_s(frame_index):.3f} s")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Video", str(exc))

    def _set_image(self, label: tk.Label, frame_bgr: np.ndarray, attr_name: str) -> None:
        """Převede OpenCV snímek do Tk náhledu."""
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        max_width, max_height = 500, 480
        scale = min(max_width / image.width, max_height / image.height, 1.0)
        if scale < 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
        if attr_name == "photo_original":
            self.display_scale = scale
            self.original_display_size = image.size
        photo = ImageTk.PhotoImage(image)
        setattr(self, attr_name, photo)
        label.configure(image=photo)

    def refresh_frame(self) -> None:
        """Překreslí aktuální snímek."""
        if self.video.info is not None and not self.analysis_running:
            self.show_frame(self.video.current_frame_index)

    def on_slider(self, value: str) -> None:
        """Reaguje na posun snímkového slideru."""
        if self.video.info is None or self.analysis_running:
            return
        frame_index = int(float(value))
        if frame_index != self.video.current_frame_index:
            self.show_frame(frame_index)

    def step_frame(self, direction: int) -> None:
        """Posune video o jeden snímek."""
        if self.video.info is not None:
            self.show_frame(self.video.current_frame_index + direction)

    def _set_range_start(self) -> None:
        if self.video.info is not None:
            self.analysis_start_var.set(self.video.current_frame_index)
            self._update_range_info()

    def _set_range_end(self) -> None:
        if self.video.info is not None:
            self.analysis_end_var.set(self.video.current_frame_index)
            self._update_range_info()

    def _update_range_info(self) -> None:
        if self.video.info is None:
            self.range_info_var.set("")
            return
        fps = self.video.info.fps or 1.0
        start = self.analysis_start_var.get()
        end = self.analysis_end_var.get()
        step = max(1, int(self.frame_step_var.get()))
        count = max(0, (end - start) // step + 1) if end >= start else 0
        self.range_info_var.set(
            f"{start / fps:.1f} s — {end / fps:.1f} s  (~{count} frames)"
        )

    def _apply_preset(self) -> None:
        """Nastaví hodnoty ovládacích prvků podle vybraného scénáře."""
        p = PRESETS.get(self.preset_var.get())
        if p is None:
            return
        self.h_min.set(p["h_min"])
        self.h_max.set(p["h_max"])
        self.s_min.set(p["s_min"])
        self.s_max.set(p["s_max"])
        self.v_min.set(p["v_min"])
        self.v_max.set(p["v_max"])
        self.min_area_var.set(p["min_area"])
        self.percentile_var.set(p["percentile"])
        self.kernel_size_var.set(p["kernel"])
        self.use_dual_range_var.set(p["dual"])
        self.include_white_core_var.set(p.get("white_core", False))
        self.white_val_min_var.set(p.get("white_val_min", 220))
        self.white_sat_max_var.set(p.get("white_sat_max", 80))
        self.merge_kernel_var.set(p.get("merge_kernel", 0))
        self.select_by_var.set(p.get("select_by", "bottom"))
        self.refresh_frame()

    def toggle_playback(self) -> None:
        """Spustí nebo zastaví přehrávání."""
        if self.video.info is None:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.after(1, self._play_next_frame)

    def _play_next_frame(self) -> None:
        """Přehraje další snímek."""
        if not self.is_playing or self.video.info is None:
            return
        next_frame = self.video.current_frame_index + 1
        if next_frame >= self.video.info.frame_count:
            self.is_playing = False
            return
        self.show_frame(next_frame)
        self.after(max(1, int(1000 / self.video.info.fps)), self._play_next_frame)

    def on_image_click(self, event) -> None:
        """Přidá kalibrační bod kliknutím do originálního snímku."""
        if self.current_frame is None:
            return
        display_w, display_h = self.original_display_size
        offset_x = max(0, (event.widget.winfo_width() - display_w) // 2)
        offset_y = max(0, (event.widget.winfo_height() - display_h) // 2)
        x = int((event.x - offset_x) / self.display_scale)
        y = int((event.y - offset_y) / self.display_scale)
        h, w = self.current_frame.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.calibration.add_point((x, y))
            self.refresh_frame()

    def compute_calibration(self) -> None:
        """Computes the cm/px conversion factor."""
        try:
            factor = self.calibration.compute(float(self.distance_cm_var.get()))
            self.cm_per_px_var.set(f"cm_per_px = {factor:.6g}")
            self.refresh_frame()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Calibration", str(exc))

    def _update_formula_engine(self) -> None:
        """Načte vzorec z GUI."""
        self.formula_engine.expression = self.formula_var.get().strip()
        self.formula_engine.set_constants_from_text(self.constants_text.get("1.0", tk.END))

    def test_formula(self) -> None:
        """Otestuje vzorec."""
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
        self.progress_var.set("Progress: 0 %")
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self) -> None:
        """Analyzuje video mimo hlavní GUI vlákno."""
        worker_video = VideoProcessor()
        try:
            # Own VideoProcessor so the GUI thread's self.video stays independent.
            worker_video.open(self.video.info.path)
            thresholds = HSVThresholds(
                self.analyzer.thresholds.hue_min,
                self.analyzer.thresholds.hue_max,
                self.analyzer.thresholds.sat_min,
                self.analyzer.thresholds.sat_max,
                self.analyzer.thresholds.val_min,
                self.analyzer.thresholds.val_max,
            )
            analyzer = FlameAnalyzer(
                thresholds,
                self.analyzer.min_area_px,
                self.analyzer.top_percentile,
                self.analyzer.kernel_size,
                self.analyzer.use_dual_range,
                self.analyzer.include_white_core,
                self.analyzer.white_val_min,
                self.analyzer.white_sat_max,
                self.analyzer.merge_kernel_size,
                self.analyzer.select_by,
            )
            formula_engine = FormulaEngine(self.formula_engine.expression, self.formula_engine.constants.copy())
            start_frame = int(self.analysis_start_var.get())
            end_frame = int(self.analysis_end_var.get())

            def progress(done: int, total: int) -> None:
                self.after(0, lambda: self.progress_var.set(f"Progress: {100 * done / max(1, total):.1f} %"))

            results = analyze_video(
                worker_video,
                analyzer,
                self.calibration.cm_per_px or 1.0,
                formula_engine,
                int(self.frame_step_var.get()),
                progress,
                start_frame,
                end_frame,
            )
            self.after(0, lambda: self._analysis_finished(results))
        except Exception as exc:  # noqa: BLE001
            self.after(0, lambda error=exc: messagebox.showerror("Analysis", str(error)))
            self.after(0, self._analysis_cleanup)
        finally:
            worker_video.release()

    def _analysis_finished(self, results: pd.DataFrame) -> None:
        """Zobrazí výsledky analýzy."""
        from visualizer import build_stats_text

        self.results = results
        self.stats_text.set(build_stats_text(results))
        self.progress_var.set("Progress: 100 %")
        self._analysis_cleanup()
        self._show_plots()
        messagebox.showinfo("Analysis", "Video analysis completed.")

    def _show_plots(self) -> None:
        """Otevře grafy v samostatném matplotlib okně."""
        if self.results.empty:
            return
        import matplotlib.pyplot as plt

        fig, (ax_height, ax_heat) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax_height.plot(self.results["time_s"], self.results["height_cm"], color="#0b7285")
        ax_height.set_ylabel("Height [cm]")
        ax_height.grid(True, alpha=0.3)
        ax_heat.plot(self.results["time_s"], self.results["vyhrevnost"], color="#c92a2a")
        ax_heat.set_ylabel("Heat value")
        ax_heat.set_xlabel("Time [s]")
        ax_heat.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show(block=False)

    def _analysis_cleanup(self) -> None:
        """Ukončí stav běžící analýzy."""
        self.analysis_running = False

    def export_results(self) -> None:
        """Saves the analysis results to CSV."""
        if self.results.empty:
            messagebox.showwarning("Export", "Please run video analysis first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_csv(self.results, path)
            messagebox.showinfo("Export", f"CSV saved:\n{path}")

    def export_frame(self) -> None:
        """Saves the current annotated frame."""
        if self.current_frame is None or self.current_measurement is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            export_annotated_frame(self.current_frame, self.current_measurement, self.analyzer, path)
            messagebox.showinfo("Export", f"Frame saved:\n{path}")

    def export_video(self) -> None:
        """Saves the annotated video."""
        if self.video.info is None or self.calibration.cm_per_px is None:
            messagebox.showwarning("Export", "Please load a video and complete calibration first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if not path:
            return
        start = time.time()
        try:
            export_annotated_video(self.video, self.analyzer, self.calibration.cm_per_px, path)
            messagebox.showinfo("Export", f"Video saved in {time.time() - start:.1f} s:\n{path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export", str(exc))
