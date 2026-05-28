"""Pixel-to-centimetre calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

import cv2
import numpy as np


Point = tuple[int, int]


@dataclass
class Calibration:
    """Stores calibration points and the computed cm/px factor."""

    points: list[Point] = field(default_factory=list)
    known_distance_cm: float | None = None
    cm_per_px: float | None = None

    def reset_points(self) -> None:
        """Clears all marked calibration points."""
        self.points.clear()
        self.cm_per_px = None

    def add_point(self, point: Point) -> None:
        """Adds a point; resets to a new pair after the second click."""
        if len(self.points) >= 2:
            self.points.clear()
            self.cm_per_px = None
        self.points.append(point)

    def compute(self, known_distance_cm: float) -> float:
        """Computes the calibration factor from a known real-world distance."""
        if len(self.points) != 2:
            raise ValueError("Exactly two points must be selected for calibration.")
        if known_distance_cm <= 0:
            raise ValueError("Known distance must be greater than zero.")

        (x1, y1), (x2, y2) = self.points
        distance_px = hypot(x2 - x1, y2 - y1)
        if distance_px <= 0:
            raise ValueError("Calibration points must not be identical.")

        self.known_distance_cm = known_distance_cm
        self.cm_per_px = known_distance_cm / distance_px
        return self.cm_per_px

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draws calibration points and connecting line onto a copy of the frame."""
        annotated = frame.copy()
        for point in self.points:
            cv2.circle(annotated, point, 6, (255, 0, 255), -1)
        if len(self.points) == 2:
            cv2.line(annotated, self.points[0], self.points[1], (255, 0, 255), 2)
        return annotated
