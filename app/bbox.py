import copy
from typing import Optional


class BBox:
    __slots__ = ("class_id", "x_center", "y_center", "width", "height", "confidence")

    def __init__(
        self,
        class_id: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        confidence: Optional[float] = None,
    ):
        self.class_id = class_id
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.confidence = confidence

    # ── coordinate helpers ───────────────────────────────────────────────────

    def pixel_rect(self, img_w: float, img_h: float):
        """Return (x1, y1, x2, y2) in image pixels."""
        hw = self.width / 2 * img_w
        hh = self.height / 2 * img_h
        cx = self.x_center * img_w
        cy = self.y_center * img_h
        return cx - hw, cy - hh, cx + hw, cy + hh

    def set_pixel_rect(self, x1: float, y1: float, x2: float, y2: float,
                       img_w: float, img_h: float) -> None:
        """Set from pixel rect. Sorts corners, clamps to image, normalises."""
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0.0, min(img_w, x1))
        y1 = max(0.0, min(img_h, y1))
        x2 = max(0.0, min(img_w, x2))
        y2 = max(0.0, min(img_h, y2))
        self.x_center = (x1 + x2) / 2.0 / img_w
        self.y_center = (y1 + y2) / 2.0 / img_h
        self.width = (x2 - x1) / img_w
        self.height = (y2 - y1) / img_h

    # ── serialisation ────────────────────────────────────────────────────────

    @classmethod
    def from_yolo_line(cls, line: str) -> Optional["BBox"]:
        parts = line.strip().split()
        if len(parts) < 5:
            return None
        try:
            return cls(
                class_id=int(parts[0]),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        except (ValueError, IndexError):
            return None

    def to_yolo_line(self) -> str:
        return (
            f"{self.class_id} "
            f"{self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}\n"
        )

    def clone(self) -> "BBox":
        return copy.copy(self)

    def __repr__(self) -> str:
        return (
            f"BBox(cls={self.class_id}, "
            f"xc={self.x_center:.4f}, yc={self.y_center:.4f}, "
            f"w={self.width:.4f}, h={self.height:.4f})"
        )
