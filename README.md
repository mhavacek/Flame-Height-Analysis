# Flame Height Analysis

A Python/Tkinter desktop application for scientific analysis of flame height from video footage. Detection uses HSV thresholding, morphological cleanup, and `connectedComponentsWithStats`. Only the main flame is measured — the component with the lowest base in the frame that exceeds the minimum area threshold.

## Project structure

| File | Description |
|---|---|
| `main.py` | Entry point — launches the GUI |
| `simple_app.py` | Main GUI (plain Tk, no `ttk`) |
| `video_processor.py` | Video loading, metadata, frame reading |
| `calibration.py` | Two-point click calibration, `cm_per_px` |
| `analyzer.py` | HSV segmentation, connected components, height measurement, batch analysis |
| `formula_engine.py` | Safe evaluation of empirical formulas |
| `visualizer.py` | Charts and summary statistics |
| `exporter.py` | CSV export, annotated frame and video export |
| `videos/` | Folder for sample videos |

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

If a video is present in the `videos/` folder the application loads it automatically. Supported formats: `mp4`, `avi`, `mov`, `mts`.

The default launch uses the plain-Tk interface (`simple_app.py`). The legacy `ttk` variant is available for comparison:

```bash
python main.py --legacy-ui
```

## Usage

1. Open a video and navigate to a frame where a known reference distance is clearly visible.
2. Click two points in the frame that correspond to a known real-world distance.
3. Enter the distance in centimetres and click **Compute Calibration**.
4. Adjust the HSV sliders until the mask covers the main orange-yellow flame.
5. Set the minimum component area and top percentile. The default measures height from the 5th percentile of the top of the component to its lowest pixel.
6. Enter a heat-value formula where `H` is flame height in cm. Examples:

```python
2.5 * H**0.67
a * H + b
math.log(H) * 3.1
```

Constants are entered separately, for example:

```text
a=2.5, b=1.0
```

7. Optionally set an **Analysis Range** (start / end frame) to limit batch processing to a specific segment.
8. Click **Analyse Video**. Results appear as in-memory statistics and a chart.
9. Save measurements via **Export CSV**.

## Advanced detection

The **Advanced Detection** panel provides additional controls for difficult conditions:

| Control | Purpose |
|---|---|
| **White-hot flame core** | Adds a second HSV range for overexposed, desaturated flame centres |
| **Merge fragments [px]** | Dilates the mask before component labelling to join nearby fragments |
| **Kernel [px]** | Morphological open/close kernel size |
| **Dual hue range** | Splits the hue range into [0, Hue max] ∪ [Hue min, 179] for flames near H = 0/179 |
| **Component selection** | `bottom` selects the lowest-based component; `area` selects the largest |

## Detection notes

- Detached flames and burning combustion products higher in the frame are excluded if they are not contiguous with the main component.
- Small stray clusters are filtered by **Min. area [px]**.
- The stability of the upper edge is controlled by **Top percentile [%]**. Lower values track the flame tip more closely; higher values are more robust against noise.
- Formula evaluation uses a restricted namespace without `__builtins__`. Allowed: `math`, `sqrt`, `log`, `exp`, `pi`, variable `H`, and user-defined constants.

## Building a Windows executable

A GitHub Actions workflow (`.github/workflows/build_windows.yml`) automatically builds a Windows `.exe` using PyInstaller whenever you push a version tag.

```bash
git tag v1.0.0
git push origin v1.0.0
```

The zipped executable is attached to the GitHub Release. You can also trigger the build manually from the **Actions** tab.

## Author

Created as part of scientific research.

- **Web:** [hvck.cz](https://hvck.cz)
- **ORCID:** [0000-0002-1276-2702](https://orcid.org/0000-0002-1276-2702)
