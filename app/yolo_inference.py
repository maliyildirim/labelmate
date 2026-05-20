import logging
from pathlib import Path
from typing import List, Optional

from .bbox import BBox
from .settings import DEFAULT_CONF, DEFAULT_DEVICE, DEFAULT_IOU, DEFAULT_IMGSZ, DEFAULT_MAX_DET

log = logging.getLogger(__name__)


class YOLOModel:
    def __init__(self) -> None:
        self._model = None
        self._path: Optional[Path] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def names(self) -> dict:
        """Return {class_id: name} from the loaded model, or empty dict."""
        if self._model is None:
            return {}
        return getattr(self._model, "names", {}) or {}

    def load(self, model_path: str) -> None:
        from ultralytics import YOLO
        self._path = Path(model_path)
        log.info("Loading model: %s", model_path)
        self._model = YOLO(str(self._path))
        log.info("Model loaded.")

    def warmup(self, imgsz: int = 640) -> None:
        if self._model is None:
            return
        import numpy as np
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self._model.predict(source=dummy, imgsz=imgsz, verbose=False)
        log.info("Warmup done.")

    def predict(
        self,
        image_path: str,
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        max_det: int = DEFAULT_MAX_DET,
        device: str = DEFAULT_DEVICE,
    ) -> List[BBox]:
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        dev = None if device == "auto" else device
        results = self._model.predict(
            source=image_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=dev,
            verbose=False,
        )
        bboxes: List[BBox] = []
        if not results:
            return bboxes

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return bboxes

        img_h, img_w = r.orig_shape
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bb = BBox(
                class_id=int(box.cls[0]),
                x_center=0.0,
                y_center=0.0,
                width=0.0,
                height=0.0,
                confidence=float(box.conf[0]),
            )
            bb.set_pixel_rect(x1, y1, x2, y2, img_w, img_h)
            bboxes.append(bb)

        log.info("Predicted %d boxes for %s", len(bboxes), image_path)
        return bboxes
