import logging
import shutil
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QEvent, QTimer
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSpinBox,
    QSplitter, QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from .bbox import BBox
from .canvas import Canvas
from .label_io import read_classes, read_label, write_classes, write_label
from .model_dialog import ModelDialog
from .settings import (
    DEFAULT_CLASS, DEFAULT_CONF, DEFAULT_DEVICE, DEFAULT_IMGSZ,
    DEFAULT_IOU, DEFAULT_MAX_DET,
)
from .utils import copy_image_to, get_image_files
from .workers import InferenceWorker, WarmupWorker
from .yolo_inference import YOLOModel

log = logging.getLogger(__name__)

_STYLE = """
QMainWindow,QWidget{background:#2b2b2b;color:#e0e0e0;}
QToolBar{background:#333;border:none;spacing:3px;padding:2px;}
QListWidget{background:#1e1e1e;border:1px solid #555;font-size:11px;}
QListWidget::item:selected{background:#0078d4;color:#fff;}
QListWidget::item:hover{background:#3a3a3a;}
QGroupBox{border:1px solid #555;border-radius:4px;margin-top:6px;font-size:11px;}
QGroupBox::title{subcontrol-origin:margin;left:6px;padding:0 2px;}
QDoubleSpinBox,QSpinBox,QComboBox{background:#3c3c3c;border:1px solid #666;
  color:#e0e0e0;padding:1px 3px;}
QLabel{color:#ddd;}
QPushButton{background:#505050;border:1px solid #777;padding:4px 8px;
  border-radius:3px;color:#e0e0e0;}
QPushButton:hover{background:#666;}
QPushButton:pressed{background:#444;}
QPushButton:checked{background:#0078d4;border-color:#005fa3;}
QCheckBox{color:#e0e0e0;}
QStatusBar{background:#222;color:#aaa;font-size:11px;}
QSplitter::handle{background:#444;}
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LabelMate")
        self.resize(1400, 900)

        # App state
        self._images: List[Path] = []
        self._current_idx: Optional[int] = None
        self._images_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._classes: List[str] = [DEFAULT_CLASS]
        self._model = YOLOModel()
        self._worker: Optional[InferenceWorker] = None
        self._warmup_worker: Optional[WarmupWorker] = None
        self._active_workers: set = set()   # keeps Python refs alive until thread finishes
        self._current_image_path: Optional[str] = None
        self._nav_lock: bool = False   # prevent re-entrant navigation

        self._build_ui()
        self._setup_shortcuts()
        self.setStyleSheet(_STYLE)
        from PySide6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_toolbar()

        splitter = QSplitter(Qt.Horizontal)

        # Left: image list
        splitter.addWidget(self._build_left_panel())

        # Center: canvas
        self._canvas = Canvas()
        self._canvas.bboxes_changed.connect(self._on_bboxes_changed)
        self._canvas.bbox_selected.connect(self._on_bbox_selected)
        self._canvas.mode_changed.connect(self._on_mode_changed)
        splitter.addWidget(self._canvas)

        # Right: annotation panel
        splitter.addWidget(self._build_right_panel())

        splitter.setSizes([210, 960, 230])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Select a model and an images folder to start.")

    def _build_toolbar(self) -> None:
        tb: QToolBar = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)

        def _btn(label: str, slot) -> QPushButton:
            b = QPushButton(label)
            b.clicked.connect(slot)
            tb.addWidget(b)
            return b

        self._model_btn = _btn("Select Model", self._select_model)
        self._images_btn = _btn("Images Folder", self._select_images_folder)
        self._output_btn = _btn("Output Folder", self._select_output_folder)
        tb.addSeparator()

        # Mode buttons
        self._sel_btn = QPushButton("Select (V)")
        self._sel_btn.setCheckable(True); self._sel_btn.setChecked(True)
        self._sel_btn.clicked.connect(lambda: self._set_mode("select"))
        tb.addWidget(self._sel_btn)

        self._draw_btn = QPushButton("Draw [Ctrl]")
        self._draw_btn.setCheckable(True)
        self._draw_btn.setEnabled(False)   # sadece gösterge, tıklanamaz
        tb.addWidget(self._draw_btn)
        tb.addSeparator()

        # Inference settings
        def _lbl(t): tb.addWidget(QLabel(t))

        _lbl("Conf:")
        self._conf_sb = QDoubleSpinBox()
        self._conf_sb.setRange(0.01, 1.0); self._conf_sb.setSingleStep(0.05)
        self._conf_sb.setValue(DEFAULT_CONF); self._conf_sb.setFixedWidth(58)
        tb.addWidget(self._conf_sb)

        _lbl("  IoU:")
        self._iou_sb = QDoubleSpinBox()
        self._iou_sb.setRange(0.01, 1.0); self._iou_sb.setSingleStep(0.05)
        self._iou_sb.setValue(DEFAULT_IOU); self._iou_sb.setFixedWidth(58)
        tb.addWidget(self._iou_sb)

        _lbl("  ImgSz:")
        self._imgsz_cb = QComboBox()
        self._imgsz_cb.addItems(["640", "960", "1280", "1920"])
        self._imgsz_cb.setCurrentText(str(DEFAULT_IMGSZ))
        self._imgsz_cb.setFixedWidth(68)
        tb.addWidget(self._imgsz_cb)

        _lbl("  MaxDet:")
        self._maxdet_sb = QSpinBox()
        self._maxdet_sb.setRange(1, 300); self._maxdet_sb.setValue(DEFAULT_MAX_DET)
        self._maxdet_sb.setFixedWidth(52)
        tb.addWidget(self._maxdet_sb)

        _lbl("  Device:")
        self._device_cb = QComboBox()
        self._device_cb.addItems(["auto", "cpu", "0", "1"])
        self._device_cb.setFixedWidth(58)
        tb.addWidget(self._device_cb)
        tb.addSeparator()

        self._auto_assist_cb = QCheckBox("Auto Assist")
        self._auto_assist_cb.setChecked(True)
        tb.addWidget(self._auto_assist_cb)

        self._empty_label_cb = QCheckBox("Empty Label")
        tb.addWidget(self._empty_label_cb)

        self._copy_images_cb = QCheckBox("Copy Images")
        tb.addWidget(self._copy_images_cb)

    def _build_left_panel(self) -> QWidget:
        w = QWidget(); w.setMaximumWidth(240)
        lay = QVBoxLayout(w); lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(4)

        hdr = QLabel("Images")
        hdr.setStyleSheet("font-weight:bold;font-size:12px;")
        lay.addWidget(hdr)

        self._count_lbl = QLabel("0 images")
        self._count_lbl.setStyleSheet("font-size:10px;color:#888;")
        lay.addWidget(self._count_lbl)

        self._img_list = QListWidget()
        self._img_list.setAlternatingRowColors(False)
        self._img_list.currentRowChanged.connect(self._on_list_row_changed)
        lay.addWidget(self._img_list)

        return w

    def _build_right_panel(self) -> QWidget:
        w = QWidget(); w.setFixedWidth(230)
        lay = QVBoxLayout(w); lay.setContentsMargins(6, 6, 6, 6); lay.setSpacing(6)

        hdr = QLabel("Annotation")
        hdr.setStyleSheet("font-weight:bold;font-size:12px;")
        lay.addWidget(hdr)

        self._box_count_lbl = QLabel("Boxes: 0")
        lay.addWidget(self._box_count_lbl)

        # Selected box info
        grp = QGroupBox("Selected Box")
        form = QFormLayout(grp); form.setSpacing(4); form.setContentsMargins(6,12,6,6)

        self._class_cb = QComboBox()
        self._class_cb.addItems(self._classes)
        self._class_cb.currentIndexChanged.connect(self._on_class_changed)
        form.addRow("Class:", self._class_cb)

        self._conf_lbl  = QLabel("—"); form.addRow("Conf:",    self._conf_lbl)
        self._xc_lbl    = QLabel("—"); form.addRow("X center:", self._xc_lbl)
        self._yc_lbl    = QLabel("—"); form.addRow("Y center:", self._yc_lbl)
        self._bw_lbl    = QLabel("—"); form.addRow("Width:",    self._bw_lbl)
        self._bh_lbl    = QLabel("—"); form.addRow("Height:",   self._bh_lbl)

        lay.addWidget(grp)

        # Buttons
        def _btn(label: str, slot, color: str = "") -> QPushButton:
            b = QPushButton(label)
            b.clicked.connect(slot)
            if color:
                b.setStyleSheet(f"background:{color};")
            lay.addWidget(b)
            return b

        _btn("Delete Selected  [Del]",  self._canvas.delete_selected)
        _btn("Clear All  [C]",          self._clear_all_with_confirm)
        _btn("Re-run Assist  [Ctrl+R]", self._rerun_assist)

        lay.addSpacing(6)
        _btn("Save  [Ctrl+S]",          self._save_current, "#2d6a2d")
        _btn("Save & Next  [Space]",    self._save_and_next, "#1a5a8a")

        lay.addStretch()
        return w

    # ── Shortcuts ────────────────────────────────────────────────────────────

    def _setup_shortcuts(self) -> None:
        def sc(key, fn):
            QShortcut(QKeySequence(key), self, fn)

        sc("Space",     self._save_and_next)
        sc("A",         self._prev_image)
        sc(Qt.Key_Left, self._prev_image)
        sc("D",         self._next_image)
        sc(Qt.Key_Right,self._next_image)
        sc("V",         lambda: self._set_mode("select"))
        sc(Qt.Key_Escape, lambda: self._set_mode("select"))
        sc(Qt.Key_Delete, self._canvas.delete_selected)
        sc("Ctrl+S",    self._save_current)
        sc("Ctrl+R",    self._rerun_assist)
        sc("Ctrl+Z",    self._canvas.undo)
        sc("Ctrl+Y",    self._canvas.redo)
        sc("F",         self._canvas.fit_to_screen)
        sc("C",         self._clear_all_with_confirm)

    # ── Mode ─────────────────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        self._canvas.set_mode(mode)

    def _on_mode_changed(self, mode: str) -> None:
        self._sel_btn.setChecked(mode == "select")
        self._draw_btn.setChecked(mode == "draw")

    # ── Ctrl geçici draw modu (uygulama geneli) ──────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        t = event.type()
        if t == QEvent.Type.KeyPress and not event.isAutoRepeat():
            if event.key() == Qt.Key_Control:
                self._canvas.set_mode("draw")
        elif t == QEvent.Type.KeyRelease and not event.isAutoRepeat():
            if event.key() == Qt.Key_Control:
                self._canvas.set_mode("select")
        return super().eventFilter(obj, event)

    # ── File / folder selection ──────────────────────────────────────────────

    def _select_model(self) -> None:
        dlg = ModelDialog(self)
        dlg.model_selected.connect(self._load_model_path)
        dlg.exec()

    def _load_model_path(self, path: str) -> None:
        self._model_btn.setText("Loading…")
        self._model_btn.setEnabled(False)
        try:
            self._model.load(path)
            self._model_btn.setText(f"Model: {Path(path).name}")
            self._status.showMessage(f"Model loaded: {Path(path).name}", 4000)
            self._apply_model_classes()
            self._start_warmup()
        except Exception as exc:
            log.exception("Model load failed")
            QMessageBox.critical(self, "Model Error", str(exc))
            self._model_btn.setText("Select Model")
        finally:
            self._model_btn.setEnabled(True)

    def _apply_model_classes(self) -> None:
        """Populate class list from model.names — only when no output folder is set yet."""
        if self._output_dir is not None:
            return  # classes.txt in output folder takes precedence
        names = self._model.names
        if not names:
            return
        classes = [names.get(i, f"cls{i}") for i in range(max(names) + 1)]
        self._classes = classes
        self._canvas.set_classes(classes)
        self._class_cb.blockSignals(True)
        self._class_cb.clear()
        self._class_cb.addItems(classes)
        self._class_cb.blockSignals(False)
        log.info("Applied %d class names from model", len(classes))

    def _start_warmup(self) -> None:
        w = WarmupWorker(self._model, 640)
        self._warmup_worker = w
        self._active_workers.add(w)

        def _warmup_done():
            self._active_workers.discard(w)
            self._status.showMessage("Model ready.", 3000)

        w.done.connect(_warmup_done)
        w.finished.connect(w.deleteLater)
        w.start()

    def _select_images_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if not folder:
            return
        self._images_dir = Path(folder)
        self._load_images(self._images_dir)

    def _select_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return
        self._output_dir = Path(folder)
        self._output_btn.setText(f"Output: {Path(folder).name}")

        # Ensure classes.txt
        classes_path = self._output_dir / "classes.txt"
        if not classes_path.exists():
            write_classes(classes_path, self._classes)
        else:
            self._classes = read_classes(classes_path)
            self._canvas.set_classes(self._classes)
            self._class_cb.blockSignals(True)
            self._class_cb.clear()
            self._class_cb.addItems(self._classes)
            self._class_cb.blockSignals(False)

        self._status.showMessage(f"Output: {folder}", 3000)
        # Refresh list colours now that we can check label existence
        self._refresh_list_colors()

    # ── Image list ───────────────────────────────────────────────────────────

    def _load_images(self, folder: Path) -> None:
        self._save_current_silent()
        self._images = get_image_files(folder)
        self._current_idx = None
        self._current_image_path = None

        self._img_list.blockSignals(True)
        self._img_list.clear()
        for img in self._images:
            item = QListWidgetItem(img.name)
            item.setToolTip(str(img))
            self._img_list.addItem(item)
        self._img_list.blockSignals(False)

        self._count_lbl.setText(f"{len(self._images)} images")
        self._images_btn.setText(f"Images: {folder.name}")
        self._refresh_list_colors()

        if self._images:
            self._navigate_to(0)

    def _refresh_list_colors(self) -> None:
        for i, img in enumerate(self._images):
            self._set_item_color(i, img)

    def _set_item_color(self, idx: int, img_path: Path) -> None:
        item = self._img_list.item(idx)
        if item is None:
            return
        if self._find_existing_label(img_path) is not None:
            item.setForeground(QColor(80, 220, 80))    # green — label exists
        else:
            item.setForeground(QColor(180, 180, 180))  # gray  — not labeled yet

    def _on_list_row_changed(self, row: int) -> None:
        if self._nav_lock or row < 0 or row == self._current_idx:
            return
        self._navigate_to(row)

    # ── Navigation ───────────────────────────────────────────────────────────

    def _navigate_to(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._images):
            return
        self._save_current_silent()
        self._open_image(idx)

    def _open_image(self, idx: int) -> None:
        self._current_idx = idx
        path = self._images[idx]
        self._current_image_path = str(path)

        # Sync list selection without triggering _on_list_row_changed
        self._nav_lock = True
        self._img_list.setCurrentRow(idx)
        self._nav_lock = False

        self._canvas.set_image(str(path))
        self._update_title()

        lp = self._find_existing_label(path)
        if lp is not None:
            bboxes = read_label(lp) or []
            self._canvas.set_bboxes(bboxes)
            self._status.showMessage(f"{path.name}  ({len(bboxes)} boxes loaded)")
        elif self._auto_assist_cb.isChecked() and self._model.is_loaded:
            self._run_inference(str(path))
        else:
            self._canvas.set_bboxes([])
            self._status.showMessage(f"{path.name}")

    def _next_image(self) -> None:
        if self._current_idx is not None:
            self._navigate_to(self._current_idx + 1)

    def _prev_image(self) -> None:
        if self._current_idx is not None:
            self._navigate_to(self._current_idx - 1)

    def _save_and_next(self) -> None:
        self._save_current()
        if self._current_idx is not None:
            next_idx = self._current_idx + 1
            if next_idx < len(self._images):
                self._navigate_to(next_idx)
            else:
                self._status.showMessage("Last image — all done!", 4000)

    # ── Label path ───────────────────────────────────────────────────────────

    def _label_path(self, image_path: Path) -> Optional[Path]:
        """Save path — always output_dir/labels/stem.txt."""
        if self._output_dir is None:
            return None
        return self._output_dir / "labels" / (image_path.stem + ".txt")

    def _find_existing_label(self, image_path: Path) -> Optional[Path]:
        """Return an existing label file for this image, checking two locations:
        1. output_dir/labels/stem.txt  (our save format)
        2. output_dir/stem.txt         (labels placed directly in output folder)
        Returns the first match, or None.
        """
        if self._output_dir is None:
            return None
        candidates = [
            self._output_dir / "labels" / (image_path.stem + ".txt"),
            self._output_dir / (image_path.stem + ".txt"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    # ── Save ─────────────────────────────────────────────────────────────────

    def _save_current(self) -> None:
        if self._current_idx is None:
            return
        if self._output_dir is None:
            QMessageBox.warning(self, "No Output Folder",
                                "Please select an output folder before saving.")
            return
        self._do_save(self._images[self._current_idx])

    def _save_current_silent(self) -> None:
        """Save without UI messages; used during navigation."""
        if self._current_idx is None or self._output_dir is None:
            return
        self._do_save(self._images[self._current_idx])

    def _do_save(self, img_path: Path) -> None:
        lp = self._label_path(img_path)
        if lp is None:
            return
        bboxes = self._canvas.get_bboxes()
        if not bboxes and not self._empty_label_cb.isChecked():
            return
        try:
            write_label(lp, bboxes)
            if self._copy_images_cb.isChecked():
                copy_image_to(img_path, self._output_dir / "images")
            self._set_item_color(self._current_idx, img_path)
            self._status.showMessage(
                f"Saved: {lp.name}  ({len(bboxes)} boxes)", 2000
            )
        except Exception as exc:
            log.exception("Save failed")
            self._status.showMessage(f"Save error: {exc}", 5000)

    # ── Inference ────────────────────────────────────────────────────────────

    def _run_inference(self, image_path: str) -> None:
        if not self._model.is_loaded:
            self._status.showMessage("No model loaded.", 3000)
            return
        self._current_image_path = image_path
        self._canvas.set_overlay("Running YOLO Assist…")
        self._canvas.set_bboxes([])

        w = InferenceWorker(
            model=self._model,
            image_path=image_path,
            imgsz=int(self._imgsz_cb.currentText()),
            conf=self._conf_sb.value(),
            iou=self._iou_sb.value(),
            max_det=self._maxdet_sb.value(),
            device=self._device_cb.currentText(),
        )
        self._active_workers.add(w)

        def _worker_done():
            self._active_workers.discard(w)

        w.results_ready.connect(self._on_inference_done)
        w.error_occurred.connect(self._on_inference_error)
        w.finished.connect(_worker_done)
        w.finished.connect(w.deleteLater)
        self._worker = w
        w.start()

    def _on_inference_done(self, bboxes: List[BBox], image_path: str) -> None:
        if image_path != self._current_image_path:
            return  # stale result
        self._canvas.set_overlay("")
        self._canvas.set_bboxes(bboxes)
        n = len(bboxes)
        self._status.showMessage(
            f"{Path(image_path).name}  — YOLO found {n} box{'es' if n!=1 else ''}"
        )

    def _on_inference_error(self, msg: str, image_path: str) -> None:
        if image_path != self._current_image_path:
            return
        self._canvas.set_overlay("")
        log.error("Inference error: %s", msg)
        self._status.showMessage(f"Inference error: {msg}", 6000)

    def _rerun_assist(self) -> None:
        if self._current_idx is None:
            return
        if not self._model.is_loaded:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
        self._run_inference(str(self._images[self._current_idx]))

    # ── Canvas signal handlers ───────────────────────────────────────────────

    def _on_bboxes_changed(self) -> None:
        n = len(self._canvas.get_bboxes())
        self._box_count_lbl.setText(f"Boxes: {n}")
        # Refresh selected box info
        self._on_bbox_selected(
            self._canvas._selected_idx if self._canvas._selected_idx is not None else -1
        )

    def _on_bbox_selected(self, idx: int) -> None:
        if idx < 0:
            self._conf_lbl.setText("—")
            self._xc_lbl.setText("—"); self._yc_lbl.setText("—")
            self._bw_lbl.setText("—"); self._bh_lbl.setText("—")
            self._class_cb.blockSignals(True)
            self._class_cb.setCurrentIndex(0)
            self._class_cb.blockSignals(False)
            return

        bboxes = self._canvas.get_bboxes()
        if idx >= len(bboxes):
            return
        bb = bboxes[idx]

        self._class_cb.blockSignals(True)
        self._class_cb.setCurrentIndex(
            bb.class_id if bb.class_id < self._class_cb.count() else 0
        )
        self._class_cb.blockSignals(False)

        self._conf_lbl.setText(f"{bb.confidence:.3f}" if bb.confidence is not None else "—")
        self._xc_lbl.setText(f"{bb.x_center:.5f}")
        self._yc_lbl.setText(f"{bb.y_center:.5f}")
        self._bw_lbl.setText(f"{bb.width:.5f}")
        self._bh_lbl.setText(f"{bb.height:.5f}")

    def _on_class_changed(self, class_id: int) -> None:
        self._canvas.change_selected_class(class_id)

    # ── Other actions ────────────────────────────────────────────────────────

    def _clear_all_with_confirm(self) -> None:
        bbs = self._canvas.get_bboxes()
        if not bbs:
            return
        r = QMessageBox.question(
            self, "Clear All",
            f"Delete all {len(bbs)} boxes?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if r == QMessageBox.Yes:
            self._canvas.clear_all()

    def _update_title(self) -> None:
        if self._current_idx is not None:
            n = len(self._images)
            name = self._images[self._current_idx].name
            self.setWindowTitle(f"YOLO Label Assist  [{self._current_idx+1}/{n}]  {name}")
        else:
            self.setWindowTitle("LabelMate")

    # ── Close guard ──────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._save_current_silent()
        super().closeEvent(event)
