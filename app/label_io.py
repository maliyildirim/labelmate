from pathlib import Path
from typing import List, Optional

from .bbox import BBox


def read_classes(path: Path) -> List[str]:
    if not path.exists():
        return ["drone"]
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l] or ["drone"]


def write_classes(path: Path, classes: List[str]) -> None:
    path.write_text("\n".join(classes) + "\n", encoding="utf-8")


def read_label(path: Path) -> Optional[List[BBox]]:
    """Return list of BBox if file exists, else None."""
    if not path.exists():
        return None
    bboxes: List[BBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            bb = BBox.from_yolo_line(line)
            if bb is not None:
                bboxes.append(bb)
    return bboxes


def write_label(path: Path, bboxes: List[BBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for bb in bboxes:
            f.write(bb.to_yolo_line())
