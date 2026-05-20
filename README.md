<p align="center">
  <img src="docs/labelmate_logo.png" alt="LabelMate Logo" width="320">
</p>

<p align="center">
  <strong>Your local AI companion for YOLO labeling.</strong><br>
  Fast. Offline. Yours.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/GUI-PySide6-green" alt="PySide6">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

LabelMate is a local desktop annotation tool that combines YOLO inference with an interactive canvas, helping you build object detection datasets without sending your images to the cloud.

## Demo

![LabelMate demo](docs/record.gif)

## Fast Windows Setup

For the quickest Windows experience, download the repository and double-click:

```text
install_windows.bat
```

The installer will:

- create a local `.venv`
- install Python dependencies
- create a desktop shortcut named `LabelMate`
- use `docs/labelmate_icon.png` as the shortcut icon when possible
- start LabelMate after installation

After that, you can launch the app from the **LabelMate** desktop icon.

If Windows SmartScreen or PowerShell asks for confirmation, allow the script only if you downloaded it from the official repository.

## Manual Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

**Requirements:** Python 3.9+

## Quick Start

1. Select a YOLO model (`.pt` / `.onnx`) or download a preset model.
2. Choose the folder containing your images.
3. Choose an output folder. LabelMate creates a `labels/` subfolder automatically.
4. Review AI-proposed boxes, or draw your own boxes.
5. Press **Space** to save and move to the next image.

## Features

- **AI-assisted labeling** - load a YOLO model and get automatic box proposals.
- **Interactive canvas** - draw, move, resize, and delete bounding boxes.
- **Zoom and pan** - scroll to zoom, right/middle-click drag to pan.
- **Undo / Redo** - full state snapshots, up to 50 steps.
- **Image list status** - green means labeled, gray means unlabeled.
- **Auto Assist** - runs inference automatically when switching images.
- **Copy Images** - optionally mirrors source images into the output folder.
- **YOLO-format output** - normalized `class_id x_center y_center width height`.
- **Multi-class support** - class names are loaded from `classes.txt`.
- **Cross-platform UI** - Windows, Linux, and macOS via PySide6.
- **Non-blocking inference** - model warmup and prediction run in worker threads.

## Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Confidence | 0.05 | Minimum detection confidence |
| IoU | 0.45 | NMS IoU threshold |
| Image size | 1280 | Inference resolution |
| Max detections | 50 | Maximum boxes per image |
| Device | auto | `auto`, `cpu`, `0`, or `1` |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Space` | Save and next image |
| `A` / `Left` | Previous image |
| `D` / `Right` | Next image |
| `V` / `Esc` | Select mode |
| `Ctrl` (hold) | Temporary draw mode |
| `Delete` | Delete selected bbox |
| `C` | Clear all boxes, with confirmation |
| `F` | Fit image to screen |
| `Ctrl+S` | Save |
| `Ctrl+R` | Re-run AI Assist |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |

## Mouse Controls

| Action | Result |
|--------|--------|
| Left-click on bbox | Select |
| Left-drag in draw mode | Draw new bbox |
| Left-drag on selected bbox | Move |
| Drag corner / edge handle | Resize |
| Right-click + drag / Middle-click + drag | Pan |
| Scroll wheel | Zoom |

## Output Structure

```text
output_folder/
+-- labels/
|   +-- image001.txt
|   +-- image002.txt
|   +-- ...
+-- images/          # only if "Copy Images" is checked
|   +-- ...
+-- classes.txt
```

**Label file format:**

```text
<class_id> <x_center> <y_center> <width> <height>
0 0.512300 0.438200 0.034100 0.028900
```

All coordinates are normalized to `[0, 1]`. Confidence scores are not written to label files.

## Project Structure

```text
labelmate/
+-- main.py
+-- requirements.txt
+-- install_windows.bat
+-- scripts/
|   +-- setup_windows.ps1
+-- docs/
|   +-- labelmate_logo.png
|   +-- labelmate_icon.png
|   +-- record.gif
+-- app/
    +-- settings.py
    +-- bbox.py
    +-- label_io.py
    +-- utils.py
    +-- yolo_inference.py
    +-- workers.py
    +-- canvas.py
    +-- main_window.py
```

## Known Limitations

- Optimized for single-class labeling workflows; multi-class is supported through `classes.txt`.
- Image folders only; video files are not supported yet.
- GPU usage is delegated to Ultralytics through the selected device option.

## License

[MIT](LICENSE) (c) 2026 malifw
