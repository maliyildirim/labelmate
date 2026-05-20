import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QProgressBar, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout,
)

from .settings import MODELS_DIR, PRESET_MODELS
from .workers import DownloadWorker

log = logging.getLogger(__name__)

_STYLE = """
QDialog,QWidget{background:#2b2b2b;color:#e0e0e0;}
QTableWidget{background:#1e1e1e;border:1px solid #555;
  gridline-color:#3a3a3a;font-size:12px;}
QTableWidget::item:selected{background:#0078d4;color:#fff;}
QHeaderView::section{background:#333;color:#ccc;padding:5px;
  border:none;border-bottom:1px solid #555;font-size:11px;}
QPushButton{background:#505050;border:1px solid #777;padding:5px 12px;
  border-radius:3px;color:#e0e0e0;}
QPushButton:hover{background:#666;}
QPushButton:pressed{background:#444;}
QPushButton:disabled{color:#666;background:#383838;border-color:#555;}
QProgressBar{background:#1e1e1e;border:1px solid #555;
  height:14px;border-radius:3px;text-align:center;color:#ccc;font-size:10px;}
QProgressBar::chunk{background:#0078d4;border-radius:2px;}
QLabel{color:#ccc;}
"""

_COL_LABEL  = 0
_COL_SIZE   = 1
_COL_NOTE   = 2
_COL_STATUS = 3


class ModelDialog(QDialog):
    """Preset model browser + file picker.  Emits model_selected(path) on load."""

    model_selected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Model")
        self.setModal(True)
        self.resize(580, 400)
        self._worker: Optional[DownloadWorker] = None
        self._build_ui()
        self.setStyleSheet(_STYLE)
        self._refresh_table()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        info = QLabel(
            "Choose a preset YOLO model — it will be downloaded once and saved locally.\n"
            "Or click <b>Browse…</b> to load your own <code>.pt</code> / <code>.onnx</code> file."
        )
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        lay.addWidget(info)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Model", "Size", "Speed / Quality", "Status"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(_COL_LABEL,  QHeaderView.Stretch)
        hh.setSectionResizeMode(_COL_SIZE,   QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(_COL_NOTE,   QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(_COL_STATUS, QHeaderView.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setStyleSheet(
            "QTableWidget{alternate-background-color:#242424;}"
        )
        self._table.doubleClicked.connect(self._load_selected)
        lay.addWidget(self._table)

        # Progress row
        self._progress_lbl = QLabel("")
        lay.addWidget(self._progress_lbl)
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        lay.addWidget(self._progress_bar)

        # Button row
        btn_row = QHBoxLayout()

        self._action_btn = QPushButton("Load")
        self._action_btn.clicked.connect(self._load_selected)
        btn_row.addWidget(self._action_btn)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_file)
        btn_row.addWidget(browse_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)

        lay.addLayout(btn_row)

        self._table.selectionModel().selectionChanged.connect(self._update_action_btn)
        self._update_action_btn()

    # ── Table ────────────────────────────────────────────────────────────────

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        for info in PRESET_MODELS:
            row = self._table.rowCount()
            self._table.insertRow(row)

            for col, key in ((_COL_LABEL, "label"), (_COL_SIZE, "size"), (_COL_NOTE, "note")):
                item = QTableWidgetItem(info[key])
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                self._table.setItem(row, col, item)

            ready = (MODELS_DIR / info["name"]).exists()
            status = QTableWidgetItem("✓  Ready" if ready else "Not downloaded")
            status.setForeground(Qt.green if ready else Qt.gray)
            status.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self._table.setItem(row, _COL_STATUS, status)

        self._table.resizeRowsToContents()

    def _update_action_btn(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            self._action_btn.setText("Load")
            return
        info = PRESET_MODELS[row]
        ready = (MODELS_DIR / info["name"]).exists()
        self._action_btn.setText("Load" if ready else "Download & Load")

    # ── Actions ──────────────────────────────────────────────────────────────

    def _load_selected(self) -> None:
        row = self._table.currentRow()
        if row < 0:
            return
        info = PRESET_MODELS[row]
        dest = MODELS_DIR / info["name"]
        if dest.exists():
            self._emit_and_close(str(dest))
        else:
            self._start_download(info, dest)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "",
            "Model files (*.pt *.onnx *.torchscript);;All files (*)",
        )
        if path:
            self._emit_and_close(path)

    def _emit_and_close(self, path: str) -> None:
        self.model_selected.emit(path)
        self.accept()

    # ── Download ─────────────────────────────────────────────────────────────

    def _start_download(self, info: dict, dest: Path) -> None:
        self._set_busy(True, f"Downloading {info['label']} ({info['size']})…")

        w = DownloadWorker(info["name"], dest)
        self._worker = w
        w.progress.connect(self._progress_bar.setValue)
        w.succeeded.connect(self._on_download_ok)
        w.failed.connect(self._on_download_fail)
        w.finished.connect(w.deleteLater)
        w.start()

    def _on_download_ok(self, path: str) -> None:
        self._set_busy(False)
        self._refresh_table()
        self._emit_and_close(path)

    def _on_download_fail(self, msg: str) -> None:
        self._set_busy(False)
        self._progress_lbl.setText(f"Download failed: {msg}")
        log.error("Model download failed: %s", msg)

    def _set_busy(self, busy: bool, label: str = "") -> None:
        self._progress_bar.setVisible(busy)
        self._progress_bar.setValue(0)
        self._progress_lbl.setText(label)
        self._action_btn.setEnabled(not busy)
        self._table.setEnabled(not busy)
