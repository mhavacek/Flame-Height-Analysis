"""Helper classes for loading and reading video files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Metadata for a loaded video."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        """Returns video duration in seconds."""
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


class VideoProcessor:
    """OpenCV VideoCapture wrapper with a safer API for GUI use."""

    def __init__(self) -> None:
        self.capture: cv2.VideoCapture | None = None
        self.info: VideoInfo | None = None
        self.current_frame_index = 0

    def open(self, path: str | Path) -> VideoInfo:
        """Opens a video file and reads basic metadata."""
        self.release()
        video_path = Path(path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.capture = capture
        self.info = VideoInfo(video_path, frame_count, fps, width, height)
        self.current_frame_index = 0
        return self.info

    def release(self) -> None:
        """Releases the currently open video file."""
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.info = None
        self.current_frame_index = 0

    def read_frame(self, frame_index: int | None = None) -> np.ndarray:
        """Returns a BGR frame at the given position."""
        if self.capture is None or self.info is None:
            raise RuntimeError("No video is loaded.")

        if frame_index is None:
            frame_index = self.current_frame_index
        frame_index = max(0, min(frame_index, self.info.frame_count - 1))

        # Backward seek corrupts libavcodec's async H264 decoder
        # (Assertion fctx->async_lock). Reopen to reset decoder state.
        if frame_index < self.current_frame_index:
            new_cap = cv2.VideoCapture(str(self.info.path))
            if not new_cap.isOpened():
                raise RuntimeError(f"Failed to reopen video: {self.info.path}")
            self.capture.release()
            self.capture = new_cap

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index}.")

        self.current_frame_index = frame_index
        return frame

    def frame_time_s(self, frame_index: int | None = None) -> float:
        """Converts a frame number to a time in seconds."""
        if self.info is None:
            return 0.0
        if frame_index is None:
            frame_index = self.current_frame_index
        return frame_index / self.info.fps if self.info.fps > 0 else 0.0

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Converts an OpenCV BGR frame to RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

