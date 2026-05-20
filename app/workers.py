import logging
import urllib.request
from pathlib import Path
from typing import List

from PySide6.QtCore import QThread, Signal

from .bbox import BBox
from .yolo_inference import YOLOModel

log = logging.getLogger(__name__)


class InferenceWorker(QThread):
    results_ready = Signal(list, str)   # List[BBox], image_path
    error_occurred = Signal(str, str)   # message, image_path

    def __init__(
        self,
        model: YOLOModel,
        image_path: str,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        device: str,
    ) -> None:
        super().__init__()
        self._model = model
        self._image_path = image_path
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._max_det = max_det
        self._device = device

    def run(self) -> None:
        try:
            bboxes = self._model.predict(
                self._image_path,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                max_det=self._max_det,
                device=self._device,
            )
            self.results_ready.emit(bboxes, self._image_path)
        except Exception as exc:
            log.exception("Inference error for %s", self._image_path)
            self.error_occurred.emit(str(exc), self._image_path)


class WarmupWorker(QThread):
    done = Signal()

    def __init__(self, model: YOLOModel, imgsz: int = 640) -> None:
        super().__init__()
        self._model = model
        self._imgsz = imgsz

    def run(self) -> None:
        try:
            self._model.warmup(self._imgsz)
        except Exception:
            log.exception("Warmup failed")
        self.done.emit()


class DownloadWorker(QThread):
    progress = Signal(int)    # 0–100
    succeeded = Signal(str)   # saved model path
    failed = Signal(str)      # error message

    _BASE_URL = "https://github.com/ultralytics/assets/releases/latest/download/"

    def __init__(self, model_name: str, dest: Path) -> None:
        super().__init__()
        self._url = self._BASE_URL + model_name
        self._dest = dest

    def run(self) -> None:
        tmp = self._dest.with_suffix(".tmp")
        try:
            self._dest.parent.mkdir(parents=True, exist_ok=True)

            def _hook(count: int, block: int, total: int) -> None:
                if total > 0:
                    self.progress.emit(min(100, int(count * block * 100 / total)))

            urllib.request.urlretrieve(self._url, str(tmp), reporthook=_hook)
            tmp.rename(self._dest)
            self.succeeded.emit(str(self._dest))
        except Exception as exc:
            log.exception("Download failed: %s", self._url)
            tmp.unlink(missing_ok=True)
            self.failed.emit(str(exc))
