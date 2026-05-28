"""Export of results and annotated outputs."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from analyzer import FlameAnalyzer


def export_csv(results: pd.DataFrame, path: str | Path) -> None:
    """Saves measurements to a CSV file."""
    results.to_csv(path, index=False, encoding="utf-8")


def export_annotated_frame(frame_bgr, measurement, analyzer: FlameAnalyzer, path: str | Path) -> None:
    """Saves the current frame with the main flame component annotated."""
    annotated = analyzer.draw_annotation(frame_bgr, measurement)
    if not cv2.imwrite(str(path), annotated):
        raise RuntimeError(f"Failed to save annotated frame: {path}")


def export_annotated_video(
    video_processor,
    analyzer: FlameAnalyzer,
    cm_per_px: float,
    path: str | Path,
) -> None:
    """Saves the full video with flame detection overlaid on every frame."""
    if video_processor.info is None:
        raise RuntimeError("No video is loaded.")

    info = video_processor.info
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, info.fps, (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create annotated video: {path}")

    try:
        for frame_index in range(info.frame_count):
            frame = video_processor.read_frame(frame_index)
            measurement = analyzer.measure_frame(frame, cm_per_px)
            writer.write(analyzer.draw_annotation(frame, measurement))
    finally:
        writer.release()
