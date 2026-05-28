"""Plot rendering and summary statistics."""

from __future__ import annotations

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def build_stats_text(results: pd.DataFrame) -> str:
    """Returns a statistics summary string for height and heat value columns."""
    if results.empty:
        return "No results available yet."

    lines = []
    for column, label in [("height_cm", "Height [cm]"), ("vyhrevnost", "Heat value")]:
        series = results[column].dropna()
        if series.empty:
            lines.append(f"{label}: no valid values")
            continue
        lines.append(
            f"{label}: mean {series.mean():.4g}, median {series.median():.4g}, "
            f"min {series.min():.4g}, max {series.max():.4g}, "
            f"std dev {series.std(ddof=1):.4g}"
        )
    return "\n".join(lines)


class PlotPanel:
    """Matplotlib panel embedded in a Tkinter widget."""

    def __init__(self, parent) -> None:
        self.figure = Figure(figsize=(7, 4), dpi=100)
        self.axes_height = self.figure.add_subplot(211)
        self.axes_heat = self.figure.add_subplot(212)
        self.figure.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.widget = self.canvas.get_tk_widget()

    def draw_results(self, results: pd.DataFrame) -> None:
        """Draws flame height and heat value over time."""
        self.axes_height.clear()
        self.axes_heat.clear()

        if not results.empty:
            self.axes_height.plot(results["time_s"], results["height_cm"], color="#0b7285")
            self.axes_heat.plot(results["time_s"], results["vyhrevnost"], color="#c92a2a")

        self.axes_height.set_ylabel("Height [cm]")
        self.axes_height.set_xlabel("Time [s]")
        self.axes_height.grid(True, alpha=0.3)
        self.axes_heat.set_ylabel("Heat value")
        self.axes_heat.set_xlabel("Time [s]")
        self.axes_heat.grid(True, alpha=0.3)
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw_idle()
