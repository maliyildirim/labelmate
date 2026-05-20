"""
Canvas widget — image display + bbox interaction.

Coordinate spaces
-----------------
  image  : float pixels, origin top-left of original image
  widget : float pixels, origin top-left of QWidget

Transforms
----------
  widget = image * scale + offset
  image  = (widget - offset) / scale

All BBox data is stored in normalised [0,1] coords.
During interaction we convert to image-pixel space, operate there,
then write back to normalised.
"""
import copy
import logging
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QColor, QCursor, QFont, QFontMetrics, QPainter, QPen, QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .bbox import BBox
from .settings import HANDLE_DRAW, HANDLE_RADIUS, MIN_BBOX_PX, UNDO_LIMIT

log = logging.getLogger(__name__)

# One colour per class (cycles if more classes than colours)
_COLORS = [
    QColor(255, 80,  80),   # red
    QColor( 80, 220,  80),  # green
    QColor( 80, 130, 255),  # blue
    QColor(255, 220,  60),  # yellow
    QColor(255,  80, 220),  # magenta
    QColor( 60, 220, 220),  # cyan
    QColor(255, 160,  60),  # orange
    QColor(160, 255,  80),  # lime
    QColor(255, 130, 180),  # pink
    QColor(180,  80, 255),  # purple
]

# Handle layout indices
# 0=TL 1=TC 2=TR 3=MR 4=BR 5=BC 6=BL 7=ML
_HANDLE_CURSORS = [
    Qt.SizeFDiagCursor,
    Qt.SizeVerCursor,
    Qt.SizeBDiagCursor,
    Qt.SizeHorCursor,
    Qt.SizeFDiagCursor,
    Qt.SizeVerCursor,
    Qt.SizeBDiagCursor,
    Qt.SizeHorCursor,
]


def _color(class_id: int) -> QColor:
    return _COLORS[class_id % len(_COLORS)]


class Canvas(QWidget):
    bboxes_changed = Signal()
    bbox_selected  = Signal(int)   # index, or -1 for deselect
    mode_changed   = Signal(str)   # 'select' | 'draw'

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color:#1a1a1a;")

        # Image state
        self._pixmap: Optional[QPixmap] = None
        self._img_w: int = 0
        self._img_h: int = 0

        # Transform
        self._scale: float = 1.0
        self._offset: QPointF = QPointF(0, 0)

        # Annotation state
        self._bboxes: List[BBox] = []
        self._classes: List[str] = ["drone"]
        self._selected_idx: Optional[int] = None
        self._mode: str = "select"
        self._default_class_id: int = 0

        # Drag state
        self._drag_type: Optional[str] = None  # 'move'|'resize'|'draw'|'pan'
        self._drag_handle: int = -1
        self._drag_start_img: QPointF = QPointF()
        self._drag_bbox_rect: Optional[Tuple[float, float, float, float]] = None
        self._draw_p0: Optional[QPointF] = None
        self._draw_p1: Optional[QPointF] = None
        self._pan_last: Optional[QPointF] = None

        # Undo / redo
        self._undo_stack: List[List[BBox]] = []
        self._redo_stack: List[List[BBox]] = []

        # Overlay text shown during inference
        self._overlay: str = ""

        # Mouse position for crosshair (widget coords)
        self._mouse_pos: Optional[QPointF] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def set_image(self, image_path: str) -> None:
        px = QPixmap(image_path)
        if px.isNull():
            log.warning("Cannot load image: %s", image_path)
            self._pixmap = None
            self._img_w = self._img_h = 0
        else:
            self._pixmap = px
            self._img_w = px.width()
            self._img_h = px.height()
        self._bboxes = []
        self._selected_idx = None
        self._drag_type = None
        self._draw_p0 = self._draw_p1 = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.fit_to_screen()
        self.update()

    def set_bboxes(self, bboxes: List[BBox]) -> None:
        self._bboxes = [bb.clone() for bb in bboxes]
        self._selected_idx = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.update()
        self.bboxes_changed.emit()

    def get_bboxes(self) -> List[BBox]:
        return [bb.clone() for bb in self._bboxes]

    def set_classes(self, classes: List[str]) -> None:
        self._classes = classes

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.mode_changed.emit(mode)
        self.setCursor(Qt.CrossCursor if mode == "draw" else Qt.ArrowCursor)
        self.update()

    def set_overlay(self, text: str) -> None:
        self._overlay = text
        self.update()

    def fit_to_screen(self) -> None:
        if self._pixmap is None or self.width() <= 0 or self.height() <= 0:
            return
        sx = self.width()  / self._img_w
        sy = self.height() / self._img_h
        self._scale = min(sx, sy) * 0.97
        self._offset = QPointF(
            (self.width()  - self._img_w * self._scale) / 2,
            (self.height() - self._img_h * self._scale) / 2,
        )
        self.update()

    def delete_selected(self) -> None:
        if self._selected_idx is None:
            return
        if 0 <= self._selected_idx < len(self._bboxes):
            self._push_undo()
            self._bboxes.pop(self._selected_idx)
            self._selected_idx = None
            self.bboxes_changed.emit()
            self.bbox_selected.emit(-1)
            self.update()

    def clear_all(self) -> None:
        if not self._bboxes:
            return
        self._push_undo()
        self._bboxes.clear()
        self._selected_idx = None
        self.bboxes_changed.emit()
        self.bbox_selected.emit(-1)
        self.update()

    def change_selected_class(self, class_id: int) -> None:
        if self._selected_idx is None:
            return
        if 0 <= self._selected_idx < len(self._bboxes):
            self._push_undo()
            self._bboxes[self._selected_idx].class_id = class_id
            self.bboxes_changed.emit()
            self.bbox_selected.emit(self._selected_idx)
            self.update()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append([bb.clone() for bb in self._bboxes])
        self._bboxes = self._undo_stack.pop()
        self._selected_idx = None
        self.bboxes_changed.emit()
        self.bbox_selected.emit(-1)
        self.update()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append([bb.clone() for bb in self._bboxes])
        self._bboxes = self._redo_stack.pop()
        self._selected_idx = None
        self.bboxes_changed.emit()
        self.bbox_selected.emit(-1)
        self.update()

    # ── Coordinate helpers ───────────────────────────────────────────────────

    def _i2w(self, p: QPointF) -> QPointF:
        return QPointF(p.x() * self._scale + self._offset.x(),
                       p.y() * self._scale + self._offset.y())

    def _w2i(self, p: QPointF) -> QPointF:
        return QPointF((p.x() - self._offset.x()) / self._scale,
                       (p.y() - self._offset.y()) / self._scale)

    def _bbox_wrect(self, bb: BBox) -> QRectF:
        x1, y1, x2, y2 = bb.pixel_rect(self._img_w, self._img_h)
        tl = self._i2w(QPointF(x1, y1))
        br = self._i2w(QPointF(x2, y2))
        return QRectF(tl, br)

    def _handle_pts(self, r: QRectF):
        x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
        mx, my = (x1+x2)/2, (y1+y2)/2
        return [
            QPointF(x1,y1), QPointF(mx,y1), QPointF(x2,y1),
            QPointF(x2,my),
            QPointF(x2,y2), QPointF(mx,y2), QPointF(x1,y2),
            QPointF(x1,my),
        ]

    def _hit_handle(self, pos: QPointF, r: QRectF) -> int:
        for i, pt in enumerate(self._handle_pts(r)):
            if abs(pos.x()-pt.x()) <= HANDLE_RADIUS and abs(pos.y()-pt.y()) <= HANDLE_RADIUS:
                return i
        return -1

    def _hit_bbox(self, pos: QPointF) -> int:
        for i in range(len(self._bboxes)-1, -1, -1):
            if self._bbox_wrect(self._bboxes[i]).contains(pos):
                return i
        return -1

    # ── Undo ────────────────────────────────────────────────────────────────

    def _push_undo(self) -> None:
        self._undo_stack.append([bb.clone() for bb in self._bboxes])
        self._redo_stack.clear()
        if len(self._undo_stack) > UNDO_LIMIT:
            self._undo_stack.pop(0)

    # ── Mouse events ────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if self._pixmap is None:
            return
        pos = event.position()

        # Pan: middle mouse or right mouse in select mode
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._drag_type = "pan"
            self._pan_last = pos
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() != Qt.LeftButton:
            return

        # Draw mode
        if self._mode == "draw":
            self._draw_p0 = self._w2i(pos)
            self._draw_p1 = self._draw_p0
            self._drag_type = "draw"
            return

        # Select mode — check selected bbox handles first
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._bboxes):
            r = self._bbox_wrect(self._bboxes[self._selected_idx])
            h = self._hit_handle(pos, r)
            if h >= 0:
                self._push_undo()
                self._drag_type = "resize"
                self._drag_handle = h
                self._drag_start_img = self._w2i(pos)
                bb = self._bboxes[self._selected_idx]
                self._drag_bbox_rect = bb.pixel_rect(self._img_w, self._img_h)
                return

        # Hit test bboxes
        hit = self._hit_bbox(pos)
        if hit >= 0:
            self._push_undo()
            self._selected_idx = hit
            self.bbox_selected.emit(hit)
            self._drag_type = "move"
            self._drag_start_img = self._w2i(pos)
            bb = self._bboxes[hit]
            self._drag_bbox_rect = bb.pixel_rect(self._img_w, self._img_h)
            self.setCursor(Qt.SizeAllCursor)
            self.update()
            return

        # Click empty area — deselect
        if self._selected_idx is not None:
            self._selected_idx = None
            self.bbox_selected.emit(-1)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._pixmap is None:
            return
        pos = event.position()
        img_pt = self._w2i(pos)

        if self._drag_type == "pan":
            if self._pan_last is not None:
                self._offset += pos - self._pan_last
                self._pan_last = pos
                self.update()
            return

        if self._drag_type == "draw":
            self._draw_p1 = img_pt
            self.update()
            return

        if self._drag_type == "move" and self._selected_idx is not None:
            dx = img_pt.x() - self._drag_start_img.x()
            dy = img_pt.y() - self._drag_start_img.y()
            ox1, oy1, ox2, oy2 = self._drag_bbox_rect
            bw, bh = ox2 - ox1, oy2 - oy1
            nx1 = max(0.0, min(self._img_w - bw, ox1 + dx))
            ny1 = max(0.0, min(self._img_h - bh, oy1 + dy))
            self._bboxes[self._selected_idx].set_pixel_rect(
                nx1, ny1, nx1+bw, ny1+bh, self._img_w, self._img_h
            )
            self.bboxes_changed.emit()
            self.update()
            return

        if self._drag_type == "resize" and self._selected_idx is not None:
            ox1, oy1, ox2, oy2 = self._drag_bbox_rect
            ix, iy = img_pt.x(), img_pt.y()
            h = self._drag_handle
            if   h == 0: ox1, oy1 = ix, iy
            elif h == 1: oy1 = iy
            elif h == 2: ox2, oy1 = ix, iy
            elif h == 3: ox2 = ix
            elif h == 4: ox2, oy2 = ix, iy
            elif h == 5: oy2 = iy
            elif h == 6: ox1, oy2 = ix, iy
            elif h == 7: ox1 = ix
            self._bboxes[self._selected_idx].set_pixel_rect(
                ox1, oy1, ox2, oy2, self._img_w, self._img_h
            )
            self.bboxes_changed.emit()
            self.update()
            return

        # Hover cursor update
        if self._mode == "select" and self._drag_type is None:
            self._update_cursor(pos)

        # Always track mouse position for crosshair
        self._mouse_pos = pos
        self.update()

    def leaveEvent(self, event) -> None:
        self._mouse_pos = None
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_type == "pan":
            self._drag_type = None
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor)
            return

        if self._drag_type == "draw" and event.button() == Qt.LeftButton:
            if self._draw_p0 and self._draw_p1:
                x1 = min(self._draw_p0.x(), self._draw_p1.x())
                y1 = min(self._draw_p0.y(), self._draw_p1.y())
                x2 = max(self._draw_p0.x(), self._draw_p1.x())
                y2 = max(self._draw_p0.y(), self._draw_p1.y())
                if (x2-x1) >= MIN_BBOX_PX and (y2-y1) >= MIN_BBOX_PX:
                    self._push_undo()
                    bb = BBox(self._default_class_id, 0.0, 0.0, 0.0, 0.0)
                    bb.set_pixel_rect(x1, y1, x2, y2, self._img_w, self._img_h)
                    self._bboxes.append(bb)
                    self._selected_idx = len(self._bboxes) - 1
                    self.bboxes_changed.emit()
                    self.bbox_selected.emit(self._selected_idx)
            self._draw_p0 = self._draw_p1 = None
            self._drag_type = None
            self.update()
            return

        if self._drag_type in ("move", "resize"):
            self._drag_type = None
            self._drag_bbox_rect = None
            self.update()
            return

        self._drag_type = None

    def wheelEvent(self, event) -> None:
        if self._pixmap is None:
            return
        pos = event.position()
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        new_scale = max(0.03, min(60.0, self._scale * factor))
        ratio = new_scale / self._scale
        self._offset = QPointF(
            pos.x() - ratio * (pos.x() - self._offset.x()),
            pos.y() - ratio * (pos.y() - self._offset.y()),
        )
        self._scale = new_scale
        self.update()

    def _update_cursor(self, pos: QPointF) -> None:
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self._bboxes):
            r = self._bbox_wrect(self._bboxes[self._selected_idx])
            h = self._hit_handle(pos, r)
            if h >= 0:
                self.setCursor(_HANDLE_CURSORS[h])
                return
        hit = self._hit_bbox(pos)
        self.setCursor(Qt.SizeAllCursor if hit >= 0 else Qt.ArrowCursor)

    # ── Paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(26, 26, 26))

        if self._pixmap is not None:
            dst = QRectF(
                self._offset.x(), self._offset.y(),
                self._img_w * self._scale, self._img_h * self._scale,
            )
            p.drawPixmap(dst, self._pixmap, QRectF(self._pixmap.rect()))

            for i, bb in enumerate(self._bboxes):
                self._paint_bbox(p, bb, i == self._selected_idx)

            if self._drag_type == "draw" and self._draw_p0 and self._draw_p1:
                self._paint_draw_rect(p)

            if self._mode == "draw" and self._mouse_pos is not None:
                self._paint_crosshair(p)

        if self._overlay:
            self._paint_overlay(p)

        p.end()

    def _paint_bbox(self, p: QPainter, bb: BBox, selected: bool) -> None:
        r = self._bbox_wrect(bb)
        c = _color(bb.class_id)

        fill = QColor(c); fill.setAlpha(55 if selected else 30)
        p.fillRect(r, fill)

        pen = QPen(c, 2.5 if selected else 1.5)
        p.setPen(pen); p.setBrush(Qt.NoBrush); p.drawRect(r)

        cls_name = self._classes[bb.class_id] if bb.class_id < len(self._classes) else f"cls{bb.class_id}"
        conf_str = f" {bb.confidence:.2f}" if bb.confidence is not None else ""
        self._paint_label(p, f"{cls_name}{conf_str}", r, c)

        if selected:
            self._paint_handles(p, r)

    def _paint_label(self, p: QPainter, text: str, r: QRectF, c: QColor) -> None:
        font = QFont("Arial", 9)
        p.setFont(font)
        fm = QFontMetrics(font)
        tw = fm.horizontalAdvance(text) + 8
        th = fm.height() + 4
        lx, ly = r.left(), r.top() - th
        if ly < 0:
            ly = r.top()
        bg = QColor(c); bg.setAlpha(210)
        p.fillRect(QRectF(lx, ly, tw, th), bg)
        p.setPen(QColor(255, 255, 255))
        p.drawText(QRectF(lx+4, ly+2, tw, th), Qt.AlignLeft | Qt.AlignVCenter, text)

    def _paint_handles(self, p: QPainter, r: QRectF) -> None:
        hd = HANDLE_DRAW
        for pt in self._handle_pts(r):
            hr = QRectF(pt.x()-hd, pt.y()-hd, hd*2, hd*2)
            p.fillRect(hr, QColor(255, 255, 255))
            p.setPen(QPen(QColor(0, 0, 0), 1)); p.setBrush(Qt.NoBrush)
            p.drawRect(hr)

    def _paint_draw_rect(self, p: QPainter) -> None:
        tl = self._i2w(self._draw_p0)
        br = self._i2w(self._draw_p1)
        r = QRectF(tl, br).normalized()
        p.setPen(QPen(QColor(255, 220, 0), 2, Qt.DashLine))
        p.setBrush(QColor(255, 220, 0, 25))
        p.drawRect(r)

    def _paint_crosshair(self, p: QPainter) -> None:
        x, y = self._mouse_pos.x(), self._mouse_pos.y()
        w, h = float(self.width()), float(self.height())
        pen = QPen(QColor(220, 220, 220, 160), 1, Qt.DashLine)
        pen.setDashPattern([6, 4])
        p.setPen(pen)
        p.drawLine(QPointF(0, y), QPointF(w, y))   # yatay
        p.drawLine(QPointF(x, 0), QPointF(x, h))   # dikey

    def _paint_overlay(self, p: QPainter) -> None:
        p.fillRect(self.rect(), QColor(0, 0, 0, 130))
        p.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 14, QFont.Bold)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignCenter, self._overlay)

    # ── Resize event ────────────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        old = event.oldSize()
        if not old.isValid() or old.width() <= 0:
            if self._pixmap is not None:
                self.fit_to_screen()
        else:
            dx = (event.size().width()  - old.width())  / 2
            dy = (event.size().height() - old.height()) / 2
            self._offset += QPointF(dx, dy)
        super().resizeEvent(event)
        self.update()
