"""Flame segmentation and height measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from formula_engine import FormulaEngine


@dataclass
class HSVThresholds:
    """HSV threshold ranges for flame detection."""

    hue_min: int = 0
    hue_max: int = 35
    sat_min: int = 100
    sat_max: int = 255
    val_min: int = 150
    val_max: int = 255


@dataclass
class FlameMeasurement:
    """Result of analysing a single frame."""

    height_px: float
    height_cm: float
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray
    component_mask: np.ndarray


class FlameAnalyzer:
    """Analyses flame using an HSV mask and connected components."""

    def __init__(
        self,
        thresholds: HSVThresholds | None = None,
        min_area_px: int = 500,
        top_percentile: float = 5.0,
        kernel_size: int = 5,
        use_dual_range: bool = False,
        include_white_core: bool = False,
        white_val_min: int = 220,
        white_sat_max: int = 80,
        merge_kernel_size: int = 0,
        select_by: str = "bottom",
    ) -> None:
        self.thresholds = thresholds or HSVThresholds()
        self.min_area_px = min_area_px
        self.top_percentile = top_percentile
        self.kernel_size = kernel_size
        self.use_dual_range = use_dual_range
        self.include_white_core = include_white_core
        self.white_val_min = white_val_min
        self.white_sat_max = white_sat_max
        self.merge_kernel_size = merge_kernel_size
        self.select_by = select_by

    def create_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Creates a cleaned binary flame mask."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        t = self.thresholds
        if self.use_dual_range:
            # Dual range covers hues wrapping around H=0/179 (red flames).
            # Range 1: [0, hue_max], Range 2: [hue_min, 179]
            lo1 = np.array([0, t.sat_min, t.val_min], dtype=np.uint8)
            hi1 = np.array([t.hue_max, t.sat_max, t.val_max], dtype=np.uint8)
            lo2 = np.array([t.hue_min, t.sat_min, t.val_min], dtype=np.uint8)
            hi2 = np.array([179, t.sat_max, t.val_max], dtype=np.uint8)
            mask = cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1), cv2.inRange(hsv, lo2, hi2))
        else:
            lower = np.array([t.hue_min, t.sat_min, t.val_min], dtype=np.uint8)
            upper = np.array([t.hue_max, t.sat_max, t.val_max], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        # White-hot core: overexposed flame center loses saturation — capture it separately.
        if self.include_white_core:
            lo_w = np.array([0, 0, self.white_val_min], dtype=np.uint8)
            hi_w = np.array([179, self.white_sat_max, 255], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo_w, hi_w))
        k = max(1, self.kernel_size)
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def select_main_component(
        self, mask: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
        """Selects the main flame component from the binary mask."""
        # Optional pre-merge: dilate to connect nearby fragments before labelling.
        if self.merge_kernel_size > 0:
            mk = max(1, self.merge_kernel_size)
            work = cv2.dilate(mask, np.ones((mk, mk), dtype=np.uint8))
        else:
            work = mask

        count, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
        best_label: int | None = None
        best_score = -1.0

        for label in range(1, count):
            x, y, width, height, area = stats[label]
            if area < self.min_area_px:
                continue
            score = float(area) if self.select_by == "area" else float(y + height)
            if score > best_score:
                best_score = score
                best_label = label

        component_mask = np.zeros_like(mask)
        if best_label is None:
            return component_mask, None

        # Keep only original (non-dilated) pixels within the selected region.
        region = (labels == best_label).astype(np.uint8)
        component_mask = cv2.bitwise_and(mask, mask, mask=region)
        if component_mask.sum() == 0:
            component_mask = region * 255

        x, y, width, height, _ = stats[best_label]
        return component_mask, (int(x), int(y), int(width), int(height))

    def measure_frame(self, frame_bgr: np.ndarray, cm_per_px: float | None) -> FlameMeasurement:
        """Measures the height of the main component in a frame."""
        mask = self.create_mask(frame_bgr)
        component_mask, bbox = self.select_main_component(mask)

        ys = np.where(component_mask > 0)[0]
        if ys.size == 0:
            return FlameMeasurement(float("nan"), float("nan"), None, mask, component_mask)

        flame_top = float(np.percentile(ys, self.top_percentile))
        flame_bottom = float(np.max(ys))
        height_px = max(0.0, flame_bottom - flame_top)
        height_cm = height_px * cm_per_px if cm_per_px is not None else float("nan")
        return FlameMeasurement(height_px, height_cm, bbox, mask, component_mask)

    def draw_annotation(self, frame_bgr: np.ndarray, measurement: FlameMeasurement) -> np.ndarray:
        """Draws the bounding box and vertical height line for the main flame."""
        annotated = frame_bgr.copy()
        if measurement.bbox is None or np.isnan(measurement.height_px):
            return annotated

        x, y, width, height = measurement.bbox
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 255, 0), 2)
        ys = np.where(measurement.component_mask > 0)[0]
        if ys.size:
            top = int(np.percentile(ys, self.top_percentile))
            bottom = int(np.max(ys))
            center_x = x + width + 12
            cv2.line(annotated, (center_x, top), (center_x, bottom), (255, 0, 0), 2)
            cv2.putText(
                annotated,
                f"{measurement.height_cm:.2f} cm",
                (min(center_x + 5, frame_bgr.shape[1] - 120), max(20, top)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated


def analyze_video(
    video_processor,
    analyzer: FlameAnalyzer,
    cm_per_px: float,
    formula_engine: FormulaEngine,
    frame_step: int = 5,
    progress_callback: Callable[[int, int], None] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> pd.DataFrame:
    """Processes the video at the given frame step and returns a results table."""
    if video_processor.info is None:
        raise RuntimeError("No video is loaded.")

    rows: list[dict[str, float | int]] = []
    frame_step = max(1, int(frame_step))
    fc = video_processor.info.frame_count
    eff_start = max(0, int(start_frame))
    eff_end = min(fc - 1, int(end_frame)) if end_frame is not None else fc - 1
    frames = list(range(eff_start, eff_end + 1, frame_step))

    for done, frame_index in enumerate(frames, start=1):
        frame = video_processor.read_frame(frame_index)
        measurement = analyzer.measure_frame(frame, cm_per_px)
        if np.isnan(measurement.height_cm):
            heat_value = float("nan")
        else:
            heat_value = formula_engine.evaluate(measurement.height_cm)
        rows.append(
            {
                "frame": frame_index,
                "time_s": video_processor.frame_time_s(frame_index),
                "height_px": measurement.height_px,
                "height_cm": measurement.height_cm,
                "vyhrevnost": heat_value,
            }
        )
        if progress_callback is not None:
            progress_callback(done, len(frames))

    return pd.DataFrame(rows)

