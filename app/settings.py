from pathlib import Path

DEFAULT_CONF = 0.05
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 1280
DEFAULT_MAX_DET = 50
DEFAULT_DEVICE = "auto"
DEFAULT_CLASS = "drone"

HANDLE_RADIUS = 6
HANDLE_DRAW = 8
MIN_BBOX_PX = 5
UNDO_LIMIT = 50

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Preset model download directory (next to main.py)
MODELS_DIR = Path(__file__).parent.parent / "models"

PRESET_MODELS = [
    {"name": "yolo11n.pt", "label": "YOLO11 Nano",    "size": "5 MB",   "note": "Fastest · Latest"},
    {"name": "yolo11s.pt", "label": "YOLO11 Small",   "size": "18 MB",  "note": "Fast · Latest"},
    {"name": "yolo11m.pt", "label": "YOLO11 Medium",  "size": "39 MB",  "note": "Balanced · Latest"},
    {"name": "yolo11l.pt", "label": "YOLO11 Large",   "size": "49 MB",  "note": "Accurate · Latest"},
    {"name": "yolov8n.pt", "label": "YOLOv8 Nano",    "size": "6 MB",   "note": "Fastest"},
    {"name": "yolov8s.pt", "label": "YOLOv8 Small",   "size": "22 MB",  "note": "Fast"},
    {"name": "yolov8m.pt", "label": "YOLOv8 Medium",  "size": "50 MB",  "note": "Balanced"},
    {"name": "yolov8l.pt", "label": "YOLOv8 Large",   "size": "84 MB",  "note": "Accurate"},
    {"name": "yolov8x.pt", "label": "YOLOv8 XLarge",  "size": "131 MB", "note": "Most accurate"},
]
