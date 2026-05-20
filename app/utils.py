import shutil
from pathlib import Path
from typing import List

from .settings import IMAGE_EXTS


def get_image_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files), key=lambda p: p.name.lower())


def copy_image_to(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
