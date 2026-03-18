# Made by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
# FreeTrace GUI — launches the freetrace binary with a graphical interface.
# Launch with:  python gui.py
"""
FreeTrace GUI — run localization and tracking by clicking.
Requires PyQt6: pip install PyQt6
"""
import atexit # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
import json
import math
import os
import signal
import sys
import subprocess
import shutil

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QProcess, QPointF, QRectF
from PyQt6.QtGui import (
    QPixmap, QFont, QColor, QPalette, QIcon, QPainter, QPolygon,
    QPen, QBrush, QPainterPath,
)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
    QDoubleSpinBox, QSpinBox, QFileDialog, QTextEdit, QSplitter,
    QTabWidget, QScrollArea, QProgressBar, QMessageBox, QRadioButton,
    QButtonGroup, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsPathItem, QSlider,
)

# Base window size — font sizes are defined relative to this
_BASE_W, _BASE_H = 1920, 1080

# Generate arrow icon PNGs for spin box buttons (CSS border-triangles don't work in Qt) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
import tempfile as _tempfile
_arrow_dir = _tempfile.mkdtemp(prefix="freetrace_arrows_")
_arrow_up_path = os.path.join(_arrow_dir, "arrow_up.png")
_arrow_down_path = os.path.join(_arrow_dir, "arrow_down.png")
_arrows_generated = False


def _generate_arrow_icons():
    global _arrows_generated
    if _arrows_generated:
        return
    size = 12
    for path, points in [
        (_arrow_up_path, [QPoint(1, size - 2), QPoint(size - 1, size - 2), QPoint(size // 2, 2)]),
        (_arrow_down_path, [QPoint(1, 2), QPoint(size - 1, 2), QPoint(size // 2, size - 2)]),
    ]:
        pix = QPixmap(size, size)
        pix.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#cccccc"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(QPolygon(points))
        painter.end()
        pix.save(path)
    _arrows_generated = True # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


def _find_freetrace_binary(): # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    """Find the freetrace binary bundled inside .app or next to gui.py."""
    if getattr(sys, 'frozen', False):
        # Frozen by PyInstaller — binary is inside .app/Contents/MacOS/
        script_dir = os.path.dirname(sys.executable)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [ # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        # Inside .app/Contents/MacOS/ (renamed to avoid case collision with PyInstaller exe)
        os.path.join(script_dir, "freetrace-bin"),
        # Same dir as gui.py or next to exe
        os.path.join(script_dir, "freetrace"),
        os.path.join(script_dir, "freetrace.exe"),
        # build/ subdirectory (development)
        os.path.join(script_dir, "build", "freetrace"),
        os.path.join(script_dir, "build", "freetrace.exe"),
    ] # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    # Try PATH
    found = shutil.which("freetrace")
    if found:
        return found
    return None # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


# ---------------------------------------------------------------------------
# Worker thread — runs FreeTrace C++ binary without blocking the UI
# ---------------------------------------------------------------------------
class FreeTraceWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, str)   # percent, stage label
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, binary: str, args: list, output_dir: str, batch: bool):
        super().__init__()
        self.binary = binary
        self.args = args
        self.output_dir = output_dir
        self.batch = batch
        self._process = None
        self._cancel = False

    def cancel(self): # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        self._cancel = True
        if self._process and self._process.poll() is None:
            try:
                if sys.platform == "win32":
                    self._process.terminate()
                else:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=3)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                try:
                    self._process.kill()
                    self._process.wait()
                except (ProcessLookupError, OSError):
                    pass # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

    def run(self):
        try:
            cmd = [self.binary] + self.args
            self.log.emit(f"$ {' '.join(cmd)}")
            self.log.emit("")
            self.progress.emit(5, "Running...")

            env = os.environ.copy() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
            # Set library path so freetrace finds bundled dylibs/shared libs
            bin_dir = os.path.dirname(self.binary)
            lib_dir = os.path.join(bin_dir, "lib")
            if sys.platform == "darwin":
                env["DYLD_LIBRARY_PATH"] = lib_dir + ":" + env.get("DYLD_LIBRARY_PATH", "")
            elif sys.platform == "linux":
                env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
            # Start in new process group so we can kill the whole tree # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
            popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            self._process = subprocess.Popen(cmd, **popen_kwargs)

            # Register atexit handler so subprocess is killed even on unexpected exit
            def _cleanup_process(proc=self._process):
                if proc.poll() is None:
                    try:
                        if sys.platform == "win32":
                            proc.kill()
                        else:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        pass
            atexit.register(_cleanup_process) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

            seen_warnings = set()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
            for line in self._process.stdout:
                line = line.rstrip("\n")
                # Show each libtiff warning only once per run
                if line.startswith("TIFFReadDirectory: Warning"):
                    if line not in seen_warnings:
                        seen_warnings.add(line)
                        self.log.emit(line)
                    continue
                self.log.emit(line)

                # Parse progress from output # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
                if "Localization" in line and "===" in line:
                    self.progress.emit(10, "Localization")
                elif "Tracking" in line and "===" in line:
                    self.progress.emit(50, "Tracking")
                elif "Starting trajectory inference" in line:
                    self.progress.emit(55, "Trajectory reconstruction")
                elif "Estimating H for" in line:
                    self.progress.emit(80, "Estimating H for trajectories")
                elif "Estimating K" in line:
                    self.progress.emit(85, "Estimating K for trajectories")
                elif "Batch complete" in line:
                    self.progress.emit(95, "Finishing") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

                if self._cancel:
                    break

            self._process.wait()
            rc = self._process.returncode

            if self._cancel:
                self.finished.emit(False, "Cancelled by user.")
            elif rc == 0:
                self.progress.emit(100, "Done")
                self.finished.emit(True, self.output_dir)
            else:
                self.finished.emit(False, f"Process exited with code {rc}")

        except Exception as e:
            self.log.emit(str(e))
            self.finished.emit(False, f"Error: {e}")


# ---------------------------------------------------------------------------
# Preview worker — runs localization on first N frames via C++ binary
# ---------------------------------------------------------------------------
class PreviewWorker(QThread):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
    """Run C++ localization on the first N frames for quick preview."""
    log = pyqtSignal(str)
    progress = pyqtSignal(int)          # percent 0-100
    finished = pyqtSignal(bool, str)    # success, message
    result_ready = pyqtSignal(object, object)  # (images_array, coords_per_frame dict)

    def __init__(self, binary: str, video_path: str, window_size: int,
                 threshold: float, n_frames: int = 50):
        super().__init__()
        self.binary = binary
        self.video_path = video_path
        self.window_size = window_size
        self.threshold = threshold
        self.n_frames = n_frames

    def run(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        import traceback
        import tifffile
        try:
            # Read and slice first N frames
            self.log.emit("Reading video...")
            if self.video_path.lower().endswith(".nd2"):
                import nd2
                all_imgs = np.array(nd2.imread(self.video_path))
            else:
                with tifffile.TiffFile(self.video_path) as tif:
                    all_imgs = tif.asarray()
            if len(all_imgs.shape) == 2:
                all_imgs = all_imgs[np.newaxis, ...]
            total = len(all_imgs) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
            n = min(self.n_frames, total)
            start = max(0, total // 2 - n // 2)
            imgs = all_imgs[start:start + n]
            self.log.emit(f"Loaded {n} frames for preview (frames {start}–{start + n - 1} of {total}).")
            self.progress.emit(15)

            # Normalize for display (same as FreeTrace read_tif)
            s_min = imgs.min()
            s_max = imgs.max()
            display_imgs = ((imgs.astype(np.float32) - s_min) / max(s_max - s_min, 1e-9))
            frame_maxes = display_imgs.max(axis=(1, 2))
            frame_maxes[frame_maxes == 0] = 1.0
            display_imgs /= frame_maxes.reshape(-1, 1, 1)

            # Write first N frames to temp TIFF
            preview_dir = os.path.join(os.path.dirname(self.video_path), "_preview_tmp")
            os.makedirs(preview_dir, exist_ok=True)
            temp_tiff = os.path.join(preview_dir, "preview.tiff")
            tifffile.imwrite(temp_tiff, imgs)
            self.log.emit("Running C++ localization...")
            self.progress.emit(25)

            # Run C++ binary in localize mode
            env = os.environ.copy()
            bin_dir = os.path.dirname(self.binary)
            lib_dir = os.path.join(bin_dir, "lib")
            if sys.platform == "darwin":
                env["DYLD_LIBRARY_PATH"] = lib_dir + ":" + env.get("DYLD_LIBRARY_PATH", "")
            elif sys.platform == "linux":
                env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

            cmd = [
                self.binary, "localize", temp_tiff, preview_dir,
                "--window", str(self.window_size),
                "--threshold", str(self.threshold),
                "--shift", "1",
            ]
            self.log.emit(f"$ {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line.startswith("TIFFReadDirectory: Warning"):
                    self.log.emit(line)
            proc.wait()
            self.progress.emit(80)

            if proc.returncode != 0:
                self.finished.emit(False, f"C++ localize exited with code {proc.returncode}")
                return

            # Parse _loc.csv (columns: frame,x,y,...)
            loc_csv = os.path.join(preview_dir, "preview_loc.csv")
            if not os.path.exists(loc_csv):
                self.finished.emit(False, f"Localization CSV not found: {loc_csv}")
                return

            loc_df = pd.read_csv(loc_csv)
            coords_per_frame = {}
            for frame_num in range(1, n + 1):
                frame_data = loc_df[loc_df['frame'] == frame_num]
                coords_per_frame[frame_num - 1] = list(
                    zip(frame_data['y'].values, frame_data['x'].values)
                )
            self.progress.emit(95)

            # Cleanup temp files
            try:
                os.remove(temp_tiff)
                os.remove(loc_csv)
                density_png = os.path.join(preview_dir, "preview_loc_2d_density.png")
                if os.path.exists(density_png):
                    os.remove(density_png)
                os.rmdir(preview_dir)
            except OSError:
                pass

            total = sum(len(v) for v in coords_per_frame.values())
            self.log.emit(f"Preview done — {total} molecules found in {n} frames.")
            self.result_ready.emit(display_imgs, coords_per_frame)
            self.finished.emit(True, "Preview complete.")
        except Exception:
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, "Preview failed — see log.")
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18


# ---------------------------------------------------------------------------
# Collapsible section widget
# ---------------------------------------------------------------------------
class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title_text = title
        self._toggle = QPushButton(f"\u25bc  {title}")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.toggled.connect(self._on_toggle)
        self._apply_toggle_style(14)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 4, 8, 4)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toggle)
        layout.addWidget(self._body)

    def _apply_toggle_style(self, font_px: int):
        self._toggle.setStyleSheet(
            f"QPushButton {{ text-align:left; font-weight:bold; font-size:{font_px}px;"
            "border:none; background:#2d2d2d; color:#ccc; padding:6px 8px; border-radius:4px; }"
            "QPushButton:checked { background:#3a3a3a; }"
        )

    def set_font_size(self, font_px: int):
        self._apply_toggle_style(font_px)

    def _on_toggle(self, checked):
        self._body.setVisible(checked)
        self._toggle.setText(
            f"{'\u25bc' if checked else '\u25b6'}  {self._toggle.text()[3:]}"
        )

    def add_widget(self, widget):
        self._body_layout.addWidget(widget)

    def add_layout(self, layout):
        self._body_layout.addLayout(layout)


# ---------------------------------------------------------------------------
# H-K Gating Canvas — interactive scatter plot with freehand boundary drawing
# ---------------------------------------------------------------------------
class HKGatingCanvas(QGraphicsView):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
    """Interactive H-K scatter plot with freehand gating.

    Users draw boundary curves that divide the H-K space into multiple regions.
    Each additional boundary further subdivides existing regions.
    Right-click removes the last drawn boundary.
    """
    gating_changed = pyqtSignal()

    _MARGIN_LEFT = 60
    _MARGIN_BOTTOM = 50
    _MARGIN_TOP = 30
    _MARGIN_RIGHT = 30
    _PLOT_W = 500
    _PLOT_H = 400

    # Color palette for multiple regions
    _REGION_COLORS = [
        QColor(100, 180, 255, 200),   # blue
        QColor(255, 120, 80, 200),    # orange
        QColor(100, 220, 100, 200),   # green
        QColor(200, 100, 255, 200),   # purple
        QColor(255, 220, 60, 200),    # yellow
        QColor(255, 100, 200, 200),   # pink
        QColor(100, 220, 220, 200),   # cyan
        QColor(220, 180, 100, 200),   # tan
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background:#1a1a1a; border:none;")

        self._traj_indices = np.array([])
        self._H = np.array([])
        self._K = np.array([])
        self._log_K = np.array([])

        self._h_min, self._h_max = 0.0, 1.0
        self._logk_min, self._logk_max = -3.0, 3.0

        # Drawing state — multiple boundaries
        self._drawing = False
        self._current_boundary = []       # QPointF list for the line being drawn
        self._current_path_item = None    # live preview path item
        self._boundaries = []             # list of finalized boundary point lists
        self._boundary_path_items = []    # list of finalized QGraphicsPathItem
        self._dot_items = []
        self._region_labels = None

        self._color_default = QColor(180, 180, 180, 160)

    def set_data(self, traj_indices, H, K):
        self._traj_indices = np.array(traj_indices)
        self._H = np.array(H, dtype=float)
        self._K = np.array(K, dtype=float)
        safe_K = np.clip(self._K, 1e-10, None)
        self._log_K = np.log10(safe_K)

        if len(self._log_K) > 0:
            self._logk_min = float(np.floor(np.min(self._log_K) - 0.5))
            self._logk_max = float(np.ceil(np.max(self._log_K) + 0.5))
        self._region_labels = None
        self._clear_boundary()
        self._draw_plot()

    def _h_to_x(self, h):
        return self._MARGIN_LEFT + (h - self._h_min) / (self._h_max - self._h_min) * self._PLOT_W

    def _logk_to_y(self, logk):
        frac = (logk - self._logk_min) / (self._logk_max - self._logk_min)
        return self._MARGIN_TOP + (1.0 - frac) * self._PLOT_H

    def _x_to_h(self, x):
        return self._h_min + (x - self._MARGIN_LEFT) / self._PLOT_W * (self._h_max - self._h_min)

    def _y_to_logk(self, y):
        frac = 1.0 - (y - self._MARGIN_TOP) / self._PLOT_H
        return self._logk_min + frac * (self._logk_max - self._logk_min)

    def _draw_plot(self):
        self._scene.clear()
        self._dot_items = []
        self._boundary_path_items = []

        total_w = self._MARGIN_LEFT + self._PLOT_W + self._MARGIN_RIGHT
        total_h = self._MARGIN_TOP + self._PLOT_H + self._MARGIN_BOTTOM
        self._scene.setSceneRect(0, 0, total_w, total_h)

        pen_axis = QPen(QColor(150, 150, 150), 1.5)
        pen_grid = QPen(QColor(60, 60, 60), 0.5, Qt.PenStyle.DashLine)
        pen_text = QColor(180, 180, 180)

        self._scene.addRect(
            QRectF(self._MARGIN_LEFT, self._MARGIN_TOP, self._PLOT_W, self._PLOT_H),
            QPen(Qt.PenStyle.NoPen), QBrush(QColor(30, 30, 30))
        )

        for h_val in np.arange(0.0, 1.01, 0.1):
            x = self._h_to_x(h_val)
            self._scene.addLine(x, self._MARGIN_TOP, x, self._MARGIN_TOP + self._PLOT_H, pen_grid)
            txt = self._scene.addSimpleText(f"{h_val:.1f}")
            txt.setBrush(pen_text)
            txt.setPos(x - 10, self._MARGIN_TOP + self._PLOT_H + 5)

        for logk_val in range(int(self._logk_min), int(self._logk_max) + 1):
            y = self._logk_to_y(logk_val)
            self._scene.addLine(self._MARGIN_LEFT, y, self._MARGIN_LEFT + self._PLOT_W, y, pen_grid)
            txt = self._scene.addSimpleText(f"1e{logk_val}")
            txt.setBrush(pen_text)
            txt.setPos(self._MARGIN_LEFT - 45, y - 8)

        self._scene.addLine(
            self._MARGIN_LEFT, self._MARGIN_TOP + self._PLOT_H,
            self._MARGIN_LEFT + self._PLOT_W, self._MARGIN_TOP + self._PLOT_H, pen_axis
        )
        self._scene.addLine(
            self._MARGIN_LEFT, self._MARGIN_TOP,
            self._MARGIN_LEFT, self._MARGIN_TOP + self._PLOT_H, pen_axis
        )

        x_label = self._scene.addSimpleText("H (Hurst exponent)")
        x_label.setBrush(pen_text)
        x_label.setPos(self._MARGIN_LEFT + self._PLOT_W / 2 - 60, self._MARGIN_TOP + self._PLOT_H + 28)

        y_label = self._scene.addSimpleText("K")
        y_label.setBrush(pen_text)
        y_label.setPos(5, self._MARGIN_TOP + self._PLOT_H / 2 - 8)

        dot_r = 3.0
        for i in range(len(self._H)):
            x = self._h_to_x(self._H[i])
            y = self._logk_to_y(self._log_K[i])
            if x < self._MARGIN_LEFT or x > self._MARGIN_LEFT + self._PLOT_W:
                self._dot_items.append(None)
                continue
            if y < self._MARGIN_TOP or y > self._MARGIN_TOP + self._PLOT_H:
                self._dot_items.append(None)
                continue
            color = self._color_default
            if self._region_labels is not None:
                idx = int(self._region_labels[i]) % len(self._REGION_COLORS)
                color = self._REGION_COLORS[idx]
            dot = QGraphicsEllipseItem(x - dot_r, y - dot_r, dot_r * 2, dot_r * 2)
            dot.setPen(QPen(Qt.PenStyle.NoPen))
            dot.setBrush(QBrush(color))
            self._scene.addItem(dot)
            self._dot_items.append(dot)

        # Redraw all finalized boundaries
        for boundary in self._boundaries:
            self._draw_finalized_boundary(boundary)

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._scene.sceneRect().width() > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _clamp_to_plot(self, pos):
        x = max(self._MARGIN_LEFT, min(pos.x(), self._MARGIN_LEFT + self._PLOT_W))
        y = max(self._MARGIN_TOP, min(pos.y(), self._MARGIN_TOP + self._PLOT_H))
        return QPointF(x, y)

    def mousePressEvent(self, event):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        if event.button() == Qt.MouseButton.LeftButton and len(self._H) > 0:
            self._drawing = True
            self._current_boundary = []
            pos = self._clamp_to_plot(self.mapToScene(event.pos()))
            self._current_boundary.append(pos)
        elif event.button() == Qt.MouseButton.RightButton and len(self._boundaries) > 0:
            # Undo last boundary
            self._boundaries.pop()
            if self._boundary_path_items:
                item = self._boundary_path_items.pop()
                if item.scene():
                    self._scene.removeItem(item)
            self._classify_points()
            self._update_dot_colors()
            self.gating_changed.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing:
            pos = self._clamp_to_plot(self.mapToScene(event.pos()))
            self._current_boundary.append(pos)
            self._draw_current_boundary()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            if len(self._current_boundary) >= 2:
                self._extend_boundary_to_edges(self._current_boundary)
                # Finalize: move current boundary into the list
                self._boundaries.append(self._current_boundary)
                # Remove live preview, draw finalized version
                if self._current_path_item and self._current_path_item.scene():
                    self._scene.removeItem(self._current_path_item)
                    self._current_path_item = None
                self._draw_finalized_boundary(self._current_boundary)
                self._current_boundary = []
                self._classify_points()
                self._update_dot_colors()
                self.gating_changed.emit()
            else:
                self._current_boundary = []
                if self._current_path_item and self._current_path_item.scene():
                    self._scene.removeItem(self._current_path_item)
                    self._current_path_item = None
        super().mouseReleaseEvent(event)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

    def _draw_current_boundary(self):
        """Draw the live preview of the boundary being drawn."""
        if self._current_path_item and self._current_path_item.scene():
            self._scene.removeItem(self._current_path_item)
        path = QPainterPath()
        path.moveTo(self._current_boundary[0])
        for pt in self._current_boundary[1:]:
            path.lineTo(pt)
        pen = QPen(QColor(255, 255, 0, 220), 2.0)
        self._current_path_item = self._scene.addPath(path, pen)

    def _draw_finalized_boundary(self, boundary):
        """Draw a finalized boundary curve on the scene."""
        path = QPainterPath()
        path.moveTo(boundary[0])
        for pt in boundary[1:]:
            path.lineTo(pt)
        pen = QPen(QColor(255, 255, 0, 220), 2.0)
        item = self._scene.addPath(path, pen)
        self._boundary_path_items.append(item)

    def _extend_boundary_to_edges(self, boundary):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Extend boundary endpoints to the nearest plot edge or existing boundary."""
        if len(boundary) < 2:
            return
        plot_left = self._MARGIN_LEFT
        plot_right = self._MARGIN_LEFT + self._PLOT_W
        plot_top = self._MARGIN_TOP
        plot_bottom = self._MARGIN_TOP + self._PLOT_H

        def _ray_seg_intersect(px, py, dx, dy, ax, ay, bx, by):
            sx, sy = bx - ax, by - ay
            denom = dx * sy - dy * sx
            if abs(denom) < 1e-12:
                return None
            t = ((ax - px) * sy - (ay - py) * sx) / denom
            s = ((ax - px) * dy - (ay - py) * dx) / denom
            if t > 1e-6 and 0.0 <= s <= 1.0:
                return t
            return None

        def _extend_to_edge(pt, direction_pt):
            dx = pt.x() - direction_pt.x()
            dy = pt.y() - direction_pt.y()
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                return pt
            dx /= length
            dy /= length
            candidates = []

            # Plot border intersections
            if abs(dx) > 1e-9:
                t = (plot_left - pt.x()) / dx
                if t > 0:
                    yy = pt.y() + t * dy
                    if plot_top <= yy <= plot_bottom:
                        candidates.append((t, QPointF(plot_left, yy)))
                t = (plot_right - pt.x()) / dx
                if t > 0:
                    yy = pt.y() + t * dy
                    if plot_top <= yy <= plot_bottom:
                        candidates.append((t, QPointF(plot_right, yy)))
            if abs(dy) > 1e-9:
                t = (plot_top - pt.y()) / dy
                if t > 0:
                    xx = pt.x() + t * dx
                    if plot_left <= xx <= plot_right:
                        candidates.append((t, QPointF(xx, plot_top)))
                t = (plot_bottom - pt.y()) / dy
                if t > 0:
                    xx = pt.x() + t * dx
                    if plot_left <= xx <= plot_right:
                        candidates.append((t, QPointF(xx, plot_bottom)))

            # Existing boundary intersections
            for existing in self._boundaries:
                for k in range(len(existing) - 1):
                    t = _ray_seg_intersect(
                        pt.x(), pt.y(), dx, dy,
                        existing[k].x(), existing[k].y(),
                        existing[k + 1].x(), existing[k + 1].y(),
                    )
                    if t is not None:
                        hit = QPointF(pt.x() + t * dx, pt.y() + t * dy)
                        candidates.append((t, hit))

            if candidates:
                candidates.sort(key=lambda c: c[0])
                return candidates[0][1]
            return pt

        # Use a point further along the boundary for a stable direction # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        # (freehand drawing produces noisy endpoints)
        n_dir = min(len(boundary) - 1, max(5, len(boundary) // 5))
        start_ext = _extend_to_edge(boundary[0], boundary[n_dir])
        end_ext = _extend_to_edge(boundary[-1], boundary[-1 - n_dir])
        boundary.insert(0, start_ext)
        boundary.append(end_ext)

    def _classify_points(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Classify scatter points into multiple regions using flood fill.

        Rasterizes all boundary curves onto a grid, then flood-fills each
        connected region with a unique label. Each data point is assigned
        the label of the grid cell it falls in.
        """
        if not self._boundaries:
            self._region_labels = None
            return

        grid_w = int(self._PLOT_W)
        grid_h = int(self._PLOT_H)
        grid = np.zeros((grid_h, grid_w), dtype=int)  # 0 = unfilled, -1 = wall

        # Rasterize all boundary segments as walls using DDA + cross pattern
        for boundary in self._boundaries:
            for j in range(len(boundary) - 1):
                x0 = boundary[j].x() - self._MARGIN_LEFT
                y0 = boundary[j].y() - self._MARGIN_TOP
                x1 = boundary[j + 1].x() - self._MARGIN_LEFT
                y1 = boundary[j + 1].y() - self._MARGIN_TOP
                dx, dy = x1 - x0, y1 - y0
                steps = max(int(abs(dx)), int(abs(dy)), 1)
                x_inc, y_inc = dx / steps, dy / steps
                x, y = x0, y0
                for _ in range(steps + 1):
                    ix, iy = int(round(x)), int(round(y))
                    for di, dj in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = iy + di, ix + dj
                        if 0 <= ni < grid_h and 0 <= nj < grid_w:
                            grid[ni, nj] = -1
                    x += x_inc
                    y += y_inc

        # Flood fill each connected region (DFS with explicit stack)
        region_id = 0
        for r in range(grid_h):
            for c in range(grid_w):
                if grid[r, c] == 0:
                    region_id += 1
                    stack = [(r, c)]
                    grid[r, c] = region_id
                    while stack:
                        cr, cc = stack.pop()
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < grid_h and 0 <= nc < grid_w and grid[nr, nc] == 0:
                                grid[nr, nc] = region_id
                                stack.append((nr, nc))

        # Assign labels to data points
        n_pts = len(self._H)
        self._region_labels = np.zeros(n_pts, dtype=int)
        for i in range(n_pts):
            px = self._h_to_x(self._H[i]) - self._MARGIN_LEFT
            py = self._logk_to_y(self._log_K[i]) - self._MARGIN_TOP
            gx = int(np.clip(round(px), 0, grid_w - 1))
            gy = int(np.clip(round(py), 0, grid_h - 1))
            label = grid[gy, gx]
            if label <= 0:
                # Point sits on a wall pixel — find nearest region
                for radius in range(1, 20):
                    found = False
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            nr, nc = gy + dr, gx + dc
                            if 0 <= nr < grid_h and 0 <= nc < grid_w and grid[nr, nc] > 0:
                                label = grid[nr, nc]
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            self._region_labels[i] = max(0, label - 1)  # convert 1-based to 0-based

        # Reorder region labels by ascending mean K  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        unique_labels = np.unique(self._region_labels)
        if len(unique_labels) > 1:
            mean_k_per_region = []
            for lbl in unique_labels:
                mask = self._region_labels == lbl
                mean_k_per_region.append((lbl, np.mean(self._K[mask])))
            sorted_labels = sorted(mean_k_per_region, key=lambda x: x[1])
            remap = {old_lbl: new_lbl for new_lbl, (old_lbl, _) in enumerate(sorted_labels)}
            self._region_labels = np.array([remap[l] for l in self._region_labels], dtype=int)
        # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

    def _update_dot_colors(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        for i, dot in enumerate(self._dot_items):
            if dot is None:
                continue
            if self._region_labels is not None:
                idx = int(self._region_labels[i]) % len(self._REGION_COLORS)
                color = self._REGION_COLORS[idx]
            else:
                color = self._color_default
            dot.setBrush(QBrush(color))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

    def _clear_boundary(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        self._boundaries = []
        for item in self._boundary_path_items:
            if item.scene():
                self._scene.removeItem(item)
        self._boundary_path_items = []
        self._current_boundary = []
        if self._current_path_item and self._current_path_item.scene():
            self._scene.removeItem(self._current_path_item)
            self._current_path_item = None
        self._region_labels = None

    def clear_gating(self):
        self._clear_boundary()
        self._update_dot_colors()
        self.gating_changed.emit()

    def get_region_data(self):
        result = {
            'H': self._H,
            'K': self._K,
            'traj_indices': self._traj_indices,
            'labels': self._region_labels,
        }
        if self._region_labels is not None:
            unique_labels = sorted(set(self._region_labels.tolist()))
            result['n_regions'] = len(unique_labels)
            result['regions'] = {
                r: np.where(self._region_labels == r)[0] for r in unique_labels
            }
        else:
            result['n_regions'] = 1
            result['regions'] = {0: np.arange(len(self._H))}
        return result  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

    def get_boundaries_data(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Export boundaries as list of point lists in data coordinates (H, logK)."""
        result = []
        for boundary in self._boundaries:
            pts = []
            for p in boundary:
                h = self._x_to_h(p.x())
                logk = self._y_to_logk(p.y())
                pts.append([h, logk])
            result.append(pts)
        return result

    def set_boundaries_data(self, boundaries_data):
        """Import boundaries from data coordinates (H, logK) and reclassify."""
        self._clear_boundary()
        for pts in boundaries_data:
            boundary = [QPointF(self._h_to_x(h), self._logk_to_y(logk)) for h, logk in pts]
            if len(boundary) >= 2:
                self._boundaries.append(boundary)
        self._draw_plot()
        if self._boundaries:
            self._classify_points()
            self._update_dot_colors()
            self.gating_changed.emit()
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class FreeTraceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FreeTrace v1.6.1.0") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        # Set window icon  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15 23:55
        _base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        _icon_path = os.path.join(_base, "icon", "freetrace_icon.png")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))
        # Scale initial size to ~70% of screen, with a reasonable minimum # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        screen = QApplication.primaryScreen().availableGeometry()
        init_w = min(int(screen.width() * 0.7), _BASE_W)
        init_h = min(int(screen.height() * 0.7), _BASE_H)
        self.setMinimumSize(640, 480)
        self.resize(init_w, init_h) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        self._worker = None
        self._preview_worker = None
        self._output_dir = None
        self._result_widgets = []
        self._binary = _find_freetrace_binary()
        # Debounce timer
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._apply_fonts)
        _generate_arrow_icons() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        self._setup_ui()
        self._apply_fonts()

    # ------------------------------------------------------------------
    # Scale helpers
    # ------------------------------------------------------------------
    def _scale(self) -> float:
        s = math.sqrt(self.width() * self.height()) / math.sqrt(_BASE_W * _BASE_H)
        return max(0.6, min(2.5, s)) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

    def _f(self, base_px: int) -> int:
        return max(8, round(base_px * self._scale()))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self): # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        # Top-level tabs: FreeTrace | Analysis
        self._main_tabs = QTabWidget()
        root.addWidget(self._main_tabs)

        self._main_tabs.setObjectName("mainTabs") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        self._main_tabs.addTab(self._build_freetrace_tab(), "FreeTrace")
        self._main_tabs.addTab(self._build_analysis_tab(), "Analysis") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

    def _build_freetrace_tab(self):
        tab = QWidget()
        tab_layout = QHBoxLayout(tab)
        tab_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        tab_layout.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([380, 670])
        return tab

    # ---- Analysis tab (sub-tabs: Class | Basic Stats | Adv Stats) --------
    def _build_analysis_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sub-tab widget inside Analysis
        self._analysis_tabs = QTabWidget()
        self._analysis_tabs.setObjectName("analysisTabs")
        self._analysis_tabs.addTab(self._build_class_tab(), "Class")
        self._analysis_tabs.addTab(self._build_basic_stats_tab(), "Basic Stats")
        self._analysis_tabs.addTab(self._build_adv_stats_tab(), "Adv Stats")
        layout.addWidget(self._analysis_tabs)

        return widget

    # ---- Class sub-tab (H-K gating) --------------------------------------
    def _build_class_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Toolbar row (top)
        toolbar = QHBoxLayout()
        self._analysis_load_btn = QPushButton("Load Data")
        self._analysis_load_btn.clicked.connect(self._on_load_data)
        toolbar.addWidget(self._analysis_load_btn)

        self._analysis_clear_btn = QPushButton("Clear Boundary")
        self._analysis_clear_btn.clicked.connect(self._on_clear_gating)
        toolbar.addWidget(self._analysis_clear_btn)

        self._analysis_load_boundary_btn = QPushButton("Load Boundary")
        self._analysis_load_boundary_btn.clicked.connect(self._on_load_boundary)
        toolbar.addWidget(self._analysis_load_boundary_btn)

        self._analysis_export_btn = QPushButton("Export Classification")
        self._analysis_export_btn.clicked.connect(self._on_export_classification)
        toolbar.addWidget(self._analysis_export_btn)

        toolbar.addStretch()

        self._analysis_info_label = QLabel("Draw a boundary on the H-K plot to classify trajectories.")
        self._analysis_info_label.setStyleSheet("color:#888;")
        toolbar.addWidget(self._analysis_info_label)

        layout.addLayout(toolbar)

        # Vertical splitter: top = two canvases, bottom = stats
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: two large windows side by side
        canvas_splitter = QSplitter(Qt.Orientation.Horizontal)

        # H-K gating canvas (left) with title # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        hk_container = QWidget()
        hk_layout = QVBoxLayout(hk_container)
        hk_layout.setContentsMargins(0, 0, 0, 0)
        hk_layout.setSpacing(2)
        hk_title = QLabel("Trajectory Classification by H and K — Draw Lines to Define Regions")
        hk_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hk_title.setStyleSheet("color:#ccc; font-size:12px; font-weight:bold; padding:4px;")
        hk_layout.addWidget(hk_title)
        self._hk_canvas = HKGatingCanvas()
        self._hk_canvas.gating_changed.connect(self._on_gating_changed)
        self._hk_canvas.setMinimumSize(300, 250)
        hk_layout.addWidget(self._hk_canvas)
        canvas_splitter.addWidget(hk_container)

        # Trajectory visualization (right) — scroll area with one view per video
        self._traj_scroll = QScrollArea()
        self._traj_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._traj_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._traj_scroll.setStyleSheet("background:#1a1a1a; border:none;")
        self._traj_scroll.setMinimumSize(300, 250)
        self._traj_scroll.setWidgetResizable(False)
        canvas_splitter.addWidget(self._traj_scroll)
        self._traj_views = []  # list of (QGraphicsView, QGraphicsScene)

        canvas_splitter.setSizes([500, 500])
        main_splitter.addWidget(canvas_splitter)

        # Bottom: statistics panel
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_widget = QWidget()
        self._stats_layout = QVBoxLayout(stats_widget)
        self._stats_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._stats_label = QLabel("No data loaded.\n\nClick 'Load Data' or run FreeTrace first.")
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet("color:#aaa; font-size:13px; padding:8px;")
        self._stats_layout.addWidget(self._stats_label)

        stats_scroll.setWidget(stats_widget)
        stats_scroll.setMinimumHeight(80)
        main_splitter.addWidget(stats_scroll)

        # Give most space to the canvases, less to stats
        main_splitter.setSizes([500, 150])
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter)

        # Store loaded datasets (multi-video support)
        self._loaded_datasets = []  # list of dicts with keys:
        # 'video_name', 'diffusion_path', 'traces_path', 'diffusion_df', 'traces_df'

        return widget

    # ---- Basic Stats sub-tab ---------------------------------------------
    def _build_basic_stats_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        placeholder = QLabel("Basic Statistics — coming soon.")
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet("color:#888; font-size:14px; padding:20px;")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)
        layout.addStretch()

        return widget

    # ---- Adv Stats sub-tab ------------------------------------------------
    def _build_adv_stats_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        placeholder = QLabel("Advanced Statistics — coming soon.")
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet("color:#888; font-size:14px; padding:20px;")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)
        layout.addStretch()

        return widget

    # ---- left panel (controls) ----------------------------------------
    def _build_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(int(self.width() * 0.4)) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # Title
        self._title_label = QLabel("FreeTrace")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("color:#7ec8e3; margin:6px 0;")
        layout.addWidget(self._title_label)

        self._subtitle_label = QLabel("Single-molecule tracking \u00b7 fBm inference")
        self._subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._subtitle_label)

        # Binary status
        if self._binary:
            status_text = f"Binary: {os.path.basename(self._binary)}"
            status_color = "#4caf50"
        else:
            status_text = "freetrace binary not found!"
            status_color = "#f44336"
        self._binary_label = QLabel(status_text)
        self._binary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._binary_label.setStyleSheet(f"color:{status_color}; font-style:italic;")
        layout.addWidget(self._binary_label)

        # Input mode selection
        self._io_sec = CollapsibleSection("Input / Output")
        mode_row = QHBoxLayout()
        self._mode_file = QRadioButton("Single file")
        self._mode_batch = QRadioButton("Batch (folder)")
        self._mode_file.setChecked(True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self._mode_file)
        mode_group.addButton(self._mode_batch)
        self._mode_file.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_file)
        mode_row.addWidget(self._mode_batch)
        self._io_sec.add_layout(mode_row)

        io_grid = QGridLayout()
        io_grid.setColumnStretch(1, 1)

        self._input_label = QLabel("Input video:")
        io_grid.addWidget(self._input_label, 0, 0)
        self._input_path = QLineEdit("")
        io_grid.addWidget(self._input_path, 0, 1)
        self._btn_input = QPushButton("Browse")
        self._btn_input.clicked.connect(self._browse_input)
        io_grid.addWidget(self._btn_input, 0, 2)

        io_grid.addWidget(QLabel("Output folder:"), 1, 0)
        self._output_path = QLineEdit("outputs")
        io_grid.addWidget(self._output_path, 1, 1)
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self._browse_output)
        io_grid.addWidget(btn_out, 1, 2)

        self._io_sec.add_layout(io_grid)
        layout.addWidget(self._io_sec)

        # Basic parameters
        self._basic_sec = CollapsibleSection("Basic Parameters")
        basic_grid = QGridLayout()
        basic_grid.setColumnStretch(1, 1)

        basic_grid.addWidget(QLabel("Window size:"), 0, 0)
        self._window_size = QSpinBox()
        self._window_size.setRange(3, 21)
        self._window_size.setSingleStep(2)
        self._window_size.setValue(7)
        self._window_size.setToolTip("Sliding window size for particle localisation (odd number).")
        basic_grid.addWidget(self._window_size, 0, 1)

        basic_grid.addWidget(QLabel("Detection threshold:"), 1, 0)
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.1, 10.0)
        self._threshold.setSingleStep(0.1)
        self._threshold.setValue(1.0)
        self._threshold.setToolTip("Signal-to-noise threshold for particle detection.")
        basic_grid.addWidget(self._threshold, 1, 1)

        basic_grid.addWidget(QLabel("Min trajectory length:"), 2, 0)
        self._cutoff = QSpinBox()
        self._cutoff.setRange(1, 50)
        self._cutoff.setValue(3)
        self._cutoff.setToolTip("Minimum number of frames a trajectory must span to be kept.")
        basic_grid.addWidget(self._cutoff, 2, 1)

        self._basic_sec.add_layout(basic_grid)
        layout.addWidget(self._basic_sec)

        # Advanced parameters
        self._adv_sec = CollapsibleSection("Advanced Parameters")
        adv_grid = QGridLayout()
        adv_grid.setColumnStretch(1, 1)

        adv_grid.addWidget(QLabel("Graph depth (\u0394t):"), 0, 0)
        self._graph_depth = QSpinBox()
        self._graph_depth.setRange(1, 10)
        self._graph_depth.setValue(3)
        self._graph_depth.setToolTip("Number of future frames considered for trajectory reconnection.")
        adv_grid.addWidget(self._graph_depth, 0, 1)

        adv_grid.addWidget(QLabel("Jump threshold (0=auto):"), 1, 0)
        self._jump_threshold = QDoubleSpinBox()
        self._jump_threshold.setRange(0, 500)
        self._jump_threshold.setSingleStep(1.0)
        self._jump_threshold.setValue(0)
        self._jump_threshold.setToolTip("Max jump distance in pixels. 0 = inferred automatically.")
        adv_grid.addWidget(self._jump_threshold, 1, 1)

        self._fbm_mode = QCheckBox("fBm mode (NN inference for alpha/K)")
        self._fbm_mode.setChecked(True)
        self._fbm_mode.setToolTip("Use fractional Brownian motion model for tracking. Requires ONNX model files.")
        adv_grid.addWidget(self._fbm_mode, 2, 0, 1, 2)

        self._adv_sec.add_layout(adv_grid)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        layout.addWidget(self._adv_sec)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%  %v")
        layout.addWidget(self._progress_bar)

        self._stage_label = QLabel("")
        self._stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._stage_label)

        # Buttons  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        btn_row = QHBoxLayout()

        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setMinimumHeight(40)
        self._preview_btn.setStyleSheet(
            "QPushButton { background:#1565c0; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#1e88e5; }"
            "QPushButton:disabled { background:#555; color:#888; }"
        )
        self._preview_btn.clicked.connect(self._on_preview)
        btn_row.addWidget(self._preview_btn)

        self._run_btn = QPushButton("\u25b6  Run FreeTrace")
        self._run_btn.setMinimumHeight(40)
        self._run_btn.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#43a047; }"
            "QPushButton:disabled { background:#555; color:#888; }"
        )
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("\u25a0  Stop")
        self._stop_btn.setMinimumHeight(40)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background:#c62828; color:white; border-radius:6px; }"
            "QPushButton:hover { background:#e53935; }"
            "QPushButton:disabled { background:#555; color:#888; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        layout.addStretch()
        return panel

    # ---- right panel (log + output images) ----------------------------
    def _build_right_panel(self):
        tabs = QTabWidget()

        # Log tab
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("background:#1a1a1a; color:#ccc; border:none;")
        tabs.addTab(self._log, "Log")

        # Results tab
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_widget = QWidget()
        self._results_layout = QVBoxLayout(results_widget)
        self._results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._no_results_label = QLabel("Results will appear here after running FreeTrace.")
        self._no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._results_layout.addWidget(self._no_results_label)

        results_scroll.setWidget(results_widget)
        tabs.addTab(results_scroll, "Results")

        # Preview tab  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(4, 4, 4, 4)
        preview_layout.setSpacing(4)

        self._preview_info_label = QLabel("Click 'Preview' to run localization on the middle 50 frames.") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        self._preview_info_label.setStyleSheet("color:#999; font-size:12px;")
        self._preview_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self._preview_info_label)

        self._preview_view = QGraphicsView()
        self._preview_scene = QGraphicsScene()
        self._preview_view.setScene(self._preview_scene)
        self._preview_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._preview_view.setStyleSheet("background:#0a0a0a; border:1px solid #333;")
        preview_layout.addWidget(self._preview_view, 1)

        slider_row = QHBoxLayout()
        self._preview_frame_label = QLabel("Frame: -")
        self._preview_frame_label.setStyleSheet("color:#ccc; font-size:11px;")
        self._preview_frame_label.setMinimumWidth(80)
        slider_row.addWidget(self._preview_frame_label)

        self._preview_slider = QSlider(Qt.Orientation.Horizontal)
        self._preview_slider.setMinimum(0)
        self._preview_slider.setMaximum(0)
        self._preview_slider.valueChanged.connect(self._on_preview_frame_changed)
        slider_row.addWidget(self._preview_slider)
        preview_layout.addLayout(slider_row)

        self._preview_images = None
        self._preview_coords = None
        tabs.addTab(preview_widget, "Preview")
        # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

        self._tabs = tabs
        return tabs

    # ------------------------------------------------------------------
    # Dynamic font scaling
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start(80)

    def _apply_fonts(self):
        f = self._f
        self._apply_dark_theme(f)

        self._title_label.setFont(QFont("Arial", f(18), QFont.Weight.Bold))
        self._run_btn.setFont(QFont("Arial", f(12), QFont.Weight.Bold))
        self._stop_btn.setFont(QFont("Arial", f(12), QFont.Weight.Bold))
        self._log.setFont(QFont("Courier New", f(13)))

        self._subtitle_label.setStyleSheet(
            f"color:#888; font-size:{f(14)}px; margin-bottom:4px;"
        )
        self._stage_label.setStyleSheet(f"color:#888; font-size:{f(13)}px;")
        try: # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
            self._no_results_label.setStyleSheet(
                f"color:#666; font-size:{f(15)}px; margin:40px;"
            )
        except RuntimeError:
            pass
        try:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
            self._analysis_info_label.setStyleSheet(
                f"color:#888; font-size:{f(13)}px;"
            )
        except RuntimeError:
            pass

        for sec in (self._io_sec, self._basic_sec, self._adv_sec):
            sec.set_font_size(f(14))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_mode_changed(self, checked):
        if self._mode_file.isChecked():
            self._input_label.setText("Input video:")
        else:
            self._input_label.setText("Input folder:")

    def _browse_input(self):
        if self._mode_file.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "Select input video", "",
                "Supported files (*.tiff *.tif *.nd2);;TIFF files (*.tiff *.tif);;ND2 files (*.nd2);;All files (*)"
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "Select input folder")
        if path:
            self._input_path.setText(path)

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._output_path.setText(path)

    def _on_run(self):
        if not self._binary:
            QMessageBox.warning(
                self, "Binary not found",
                "Cannot find the freetrace binary.\n"
                "Build the project first (cmake --build build/)\n"
                "or place the binary next to gui.py."
            )
            return

        input_path = self._input_path.text().strip()
        if not input_path:
            QMessageBox.warning(self, "No input", "Please select an input file or folder.")
            return

        batch = self._mode_batch.isChecked()
        if batch and not os.path.isdir(input_path):
            QMessageBox.warning(self, "Not a directory", f"Batch mode requires a folder:\n{input_path}")
            return
        if not batch and not os.path.isfile(input_path):
            QMessageBox.warning(self, "File not found", f"Cannot find:\n{input_path}")
            return

        output_dir = self._output_path.text().strip() or "outputs"

        # Build command arguments
        args = []
        if batch:
            args.append("batch")
        args.extend([input_path, output_dir])
        args.extend(["--window", str(self._window_size.value())])
        args.extend(["--threshold", str(self._threshold.value())])
        args.extend(["--shift", "1"])
        args.extend(["--depth", str(self._graph_depth.value())])
        args.extend(["--cutoff", str(self._cutoff.value())])

        if self._jump_threshold.value() > 0:
            args.extend(["--jump", str(self._jump_threshold.value())])

        if not self._fbm_mode.isChecked():
            args.append("--no-fbm")


        # Setup UI
        self._log.clear()
        self._log.append(f"<b>Input:</b> {input_path}")
        self._log.append(f"<b>Output:</b> {output_dir}")
        self._log.append(f"<b>Mode:</b> {'Batch' if batch else 'Single file'}")
        self._log.append(f"<b>fBm mode:</b> {self._fbm_mode.isChecked()}")
        self._log.append("-" * 60)
        self._progress_bar.setValue(0)
        self._stage_label.setText("")
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._tabs.setCurrentIndex(0)

        self._worker = FreeTraceWorker(self._binary, args, output_dir, batch)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._update_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._append_log("Cancellation requested...")

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.wait(5000)
        super().closeEvent(event)

    def _append_log(self, text: str):
        self._log.append(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _update_progress(self, value: int, label: str):
        self._progress_bar.setValue(value)
        self._stage_label.setText(label)

    def _on_finished(self, success: bool, message: str):
        self._reset_buttons()
        if success:
            self._output_dir = message
            self._append_log(f"Done. Results saved to: {message}")
            self._load_results(message)
            self._tabs.setCurrentIndex(1)
        else:
            self._append_log(f"Failed: {message}")
            QMessageBox.critical(self, "FreeTrace error", message)

    def _reset_buttons(self):
        self._run_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Preview  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
    # ------------------------------------------------------------------
    def _on_preview(self):
        if not self._binary:
            QMessageBox.warning(self, "Binary not found",
                "Cannot find the freetrace binary.\n"
                "Build the project first or place the binary next to gui.py.")
            return
        input_path = self._input_path.text().strip()
        if not os.path.isfile(input_path):
            QMessageBox.warning(self, "File not found", f"Cannot find:\n{input_path}")
            return
        if self._preview_worker and self._preview_worker.isRunning():
            return

        self._preview_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._stage_label.setText("Preview")
        self._tabs.setCurrentIndex(0)
        self._append_log("-" * 40)
        self._append_log("<b>Starting preview...</b>")

        self._preview_worker = PreviewWorker(
            binary=self._binary,
            video_path=input_path,
            window_size=self._window_size.value(),
            threshold=self._threshold.value(),
            n_frames=50,
        )
        self._preview_worker.log.connect(self._append_log)
        self._preview_worker.progress.connect(lambda v: self._update_progress(v, "Preview"))
        self._preview_worker.result_ready.connect(self._on_preview_result)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.start()

    def _on_preview_result(self, images, coords_per_frame):
        self._preview_images = images
        self._preview_coords = coords_per_frame
        n = len(images)
        self._preview_slider.setMaximum(n - 1)
        self._preview_slider.setValue(0)
        self._preview_info_label.setText(f"{n} frames loaded — use slider to browse")
        self._show_preview_frame(0)

    def _on_preview_finished(self, success, message):
        self._reset_buttons()
        if success:
            self._tabs.setCurrentIndex(2)  # switch to Preview tab
        else:
            self._append_log(f"✗ {message}")

    def _on_preview_frame_changed(self, frame_idx):
        self._preview_frame_label.setText(f"Frame: {frame_idx}")
        if self._preview_images is not None:
            self._show_preview_frame(frame_idx)

    def _show_preview_frame(self, frame_idx):
        """Render a single preview frame with red localization dots."""
        if self._preview_images is None or frame_idx >= len(self._preview_images):
            return

        img = self._preview_images[frame_idx]
        h, w = img.shape

        img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        from PyQt6.QtGui import QImage
        qimg = QImage(img8.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg.copy())

        self._preview_scene.clear()
        self._preview_scene.setSceneRect(0, 0, w, h)
        self._preview_scene.addPixmap(pixmap)

        # Draw localization dots
        if self._preview_coords and frame_idx in self._preview_coords:
            coords = self._preview_coords[frame_idx]
            red_pen = QPen(Qt.PenStyle.NoPen)
            red_brush = QBrush(QColor(255, 50, 50, 200))
            dot_r = max(1.5, min(w, h) / 200)
            for y, x in coords:
                self._preview_scene.addEllipse(
                    x - dot_r, y - dot_r, dot_r * 2, dot_r * 2,
                    red_pen, red_brush,
                )

        self._preview_view.fitInView(
            self._preview_scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

    # ------------------------------------------------------------------
    # Analysis tab slots  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
    # ------------------------------------------------------------------
    def _on_load_data(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Load FreeTrace output data. Clears previous data, supports multi-select."""
        start_dir = self._output_dir or ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select FreeTrace output CSV(s) (*_diffusion.csv or *_traces.csv)",
            start_dir,
            "FreeTrace CSV (*_diffusion.csv *_traces.csv);;All CSV (*.csv);;All files (*)"
        )
        if not paths:
            return
        self._loaded_datasets = []
        for path in paths:
            self._load_data_from_file(path)

    def _load_data_from_file(self, selected_path):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Load a single video's data and add it to the loaded datasets."""
        try:
            if '_traces.csv' in selected_path:
                traces_path = selected_path
                diffusion_path = selected_path.replace('_traces.csv', '_diffusion.csv')
            elif '_diffusion.csv' in selected_path:
                diffusion_path = selected_path
                traces_path = selected_path.replace('_diffusion.csv', '_traces.csv')
            else:
                QMessageBox.warning(self, "Unrecognized file",
                                    "Please select a file ending with _diffusion.csv or _traces.csv")
                return

            # Skip if same video already loaded in this batch
            for ds in self._loaded_datasets:
                if ds['diffusion_path'] == diffusion_path:
                    return

            if not os.path.exists(diffusion_path):
                QMessageBox.warning(self, "File not found",
                                    f"Diffusion file not found:\n{diffusion_path}")
                return
            if not os.path.exists(traces_path):
                QMessageBox.warning(self, "File not found",
                                    f"Traces file not found:\n{traces_path}")
                return

            # Validate diffusion CSV format # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
            df = pd.read_csv(diffusion_path)
            required_diff_cols = {'traj_idx', 'H', 'K'}
            if not required_diff_cols.issubset(df.columns):
                missing = required_diff_cols - set(df.columns)
                QMessageBox.warning(self, "Invalid diffusion file",
                                    f"Not a valid FreeTrace diffusion output.\n"
                                    f"Missing columns: {', '.join(sorted(missing))}\n"
                                    f"Expected: traj_idx, H, K")
                return
            if not all(df[c].dtype.kind in ('i', 'f') for c in ['H', 'K']):
                QMessageBox.warning(self, "Invalid diffusion file",
                                    "Columns H and K must be numeric.")
                return

            # Validate traces CSV format
            traces_df = pd.read_csv(traces_path)
            required_trace_cols = {'traj_idx', 'frame', 'x', 'y'}
            if not required_trace_cols.issubset(traces_df.columns):
                missing = required_trace_cols - set(traces_df.columns)
                QMessageBox.warning(self, "Invalid traces file",
                                    f"Not a valid FreeTrace traces output.\n"
                                    f"Missing columns: {', '.join(sorted(missing))}\n"
                                    f"Expected: traj_idx, frame, x, y, z")
                return
            if not all(traces_df[c].dtype.kind in ('i', 'f') for c in ['frame', 'x', 'y']):
                QMessageBox.warning(self, "Invalid traces file",
                                    "Columns frame, x, y must be numeric.")
                return

            fname = os.path.basename(diffusion_path)
            video_name = fname.replace('_diffusion.csv', '')

            self._loaded_datasets.append({
                'video_name': video_name,
                'diffusion_path': diffusion_path,
                'traces_path': traces_path,
                'diffusion_df': df,
                'traces_df': traces_df,
            })

            self._rebuild_canvas_data()
        except Exception as e:
            QMessageBox.critical(self, "Error loading data", str(e))

    def _rebuild_canvas_data(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Combine all loaded datasets and update the H-K canvas."""
        if not self._loaded_datasets:
            return

        all_H, all_K = [], []
        for ds in self._loaded_datasets:
            df = ds['diffusion_df']
            all_H.append(df['H'].values)
            all_K.append(df['K'].values)

        combined_H = np.concatenate(all_H)
        combined_K = np.concatenate(all_K)
        combined_idx = np.arange(len(combined_H))

        self._hk_canvas.set_data(combined_idx, combined_H, combined_K)

        total = len(combined_H)
        n_vids = len(self._loaded_datasets)
        self._analysis_info_label.setText(
            f"Loaded {total} trajectories from {n_vids} video(s). "
            f"Draw a boundary to classify."
        )
        self._update_stats_display()
        self._draw_trajectories()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

    def _on_clear_gating(self):
        self._hk_canvas.clear_gating()

    def _on_load_boundary(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Load boundary information from a JSON file."""
        start_dir = self._output_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select boundary file",
            start_dir,
            "Boundary JSON (*_boundaries.json);;All JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            boundaries = data.get('boundaries', [])
            if not boundaries:
                QMessageBox.warning(self, "No boundaries", "No boundary data found in file.")
                return
            self._hk_canvas.set_boundaries_data(boundaries)
        except Exception as e:
            QMessageBox.critical(self, "Error loading boundary", str(e))

    def _on_gating_changed(self):
        self._update_stats_display()
        self._draw_trajectories()

    def _draw_trajectories(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Draw trajectories with one panel per loaded video, inside a scroll area."""
        self._traj_views = []

        if not self._loaded_datasets:
            container = QWidget()
            lay = QVBoxLayout(container)
            lbl = QLabel("No trajectory data available.")
            lbl.setStyleSheet("color:#999;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)
            self._traj_scroll.setWidget(container)
            return

        data = self._hk_canvas.get_region_data()
        labels = data['labels']
        region_colors = HKGatingCanvas._REGION_COLORS

        # Build per-dataset label maps
        offset = 0
        ds_label_maps = []
        for ds in self._loaded_datasets:
            n = len(ds['diffusion_df'])
            label_map = {}
            if labels is not None:
                ds_labels = labels[offset:offset + n]
                for i, tidx in enumerate(ds['diffusion_df']['traj_idx'].values):
                    label_map[int(tidx)] = int(ds_labels[i])
            ds_label_maps.append(label_map)
            offset += n

        # Panel width: fill available space or scroll when too many
        viewport_w = self._traj_scroll.viewport().width()
        n_vids = len(self._loaded_datasets)
        min_panel_w = 400
        panel_w = max(viewport_w // n_vids - 4, min_panel_w)

        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        max_panels = min(len(self._loaded_datasets), 10)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

        rng_colors = {}
        for ds_idx, ds in enumerate(self._loaded_datasets[:max_panels]):
            # Per-video panel: label on top, view below
            panel = QWidget()
            panel.setFixedWidth(panel_w)
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(2, 2, 2, 2)
            panel_layout.setSpacing(2)

            title = QLabel(ds['video_name'])
            title.setStyleSheet("color:#ccc; font-size:12px; font-weight:bold;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            panel_layout.addWidget(title)

            scene = QGraphicsScene()
            view = QGraphicsView(scene)
            view.setRenderHint(QPainter.RenderHint.Antialiasing)
            view.setStyleSheet("background:#0a0a0a; border:1px solid #333;")
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            panel_layout.addWidget(view)

            self._traj_views.append((view, scene))

            # Draw this video's trajectories
            df = ds['traces_df']
            label_map = ds_label_maps[ds_idx]

            x_max = df['x'].max()
            y_max = df['y'].max()
            canvas_w = max(x_max + 10, 100)
            canvas_h = max(y_max + 10, 100)

            scene.setSceneRect(0, 0, canvas_w, canvas_h)
            scene.addRect(
                QRectF(0, 0, canvas_w, canvas_h),
                QPen(Qt.PenStyle.NoPen), QBrush(QColor(0, 0, 0))
            )

            for tidx in df['traj_idx'].unique():
                traj_data = df[df['traj_idx'] == tidx].sort_values('frame')
                positions = list(zip(traj_data['x'].values, traj_data['y'].values))
                if len(positions) < 2:
                    continue

                if labels is not None:
                    region = label_map.get(int(tidx), 0)
                    color = region_colors[region % len(region_colors)]
                else:
                    key = (ds_idx, int(tidx))
                    if key not in rng_colors:
                        rng = np.random.default_rng(hash(key) & 0x7FFFFFFF)
                        rgb = rng.integers(low=50, high=256, size=3)
                        rng_colors[key] = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), 200)
                    color = rng_colors[key]

                pen = QPen(color, 0.5)
                path = QPainterPath()
                path.moveTo(positions[0][0], positions[0][1])
                for x, y in positions[1:]:
                    path.lineTo(x, y)
                scene.addPath(path, pen)

            view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            container_layout.addWidget(panel)

        if len(self._loaded_datasets) > max_panels:
            note = QLabel(f"Showing {max_panels} / {len(self._loaded_datasets)} videos")
            note.setStyleSheet("color:#ff9; font-size:11px;")
            note.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(note)

        total_w = max_panels * (panel_w + 4)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        container.setMinimumWidth(total_w)
        container.setFixedHeight(self._traj_scroll.viewport().height())
        self._traj_scroll.setWidget(container)
        # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

    def _update_stats_display(self):
        data = self._hk_canvas.get_region_data()
        H, K = data['H'], data['K']
        labels = data['labels']

        if len(H) == 0:
            self._stats_label.setText("No data loaded.")
            return

        lines = []
        lines.append(f"<b>Total trajectories:</b> {len(H)}")
        lines.append(f"<b>H range:</b> [{np.min(H):.3f}, {np.max(H):.3f}]")
        lines.append(f"<b>K range:</b> [{np.min(K):.4g}, {np.max(K):.4g}]")
        lines.append("")

        if labels is not None:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
            region_colors = HKGatingCanvas._REGION_COLORS
            _color_names = ["blue", "orange", "green", "purple", "yellow", "pink", "cyan", "tan"]
            unique_regions = sorted(set(labels.tolist()))
            for region_id in unique_regions:
                cidx = region_id % len(region_colors)
                c = region_colors[cidx]
                color_hex = c.name()
                cname = _color_names[cidx] if cidx < len(_color_names) else f"color {cidx}"
                region_name = f"Region {region_id} ({cname})"
                mask = labels == region_id
                n = int(np.sum(mask))
                lines.append(f"<span style='color:{color_hex}'><b>--- {region_name} ---</b></span>")
                lines.append(f"  Count: <b>{n}</b> ({100*n/len(H):.1f}%)")
                if n > 0:
                    h_sub = H[mask]
                    k_sub = K[mask]
                    lines.append(f"  H: mean={np.mean(h_sub):.3f}, "
                                 f"median={np.median(h_sub):.3f}, "
                                 f"std={np.std(h_sub):.3f}")
                    lines.append(f"  K: mean={np.mean(k_sub):.4g}, "
                                 f"median={np.median(k_sub):.4g}, "
                                 f"std={np.std(k_sub):.4g}")
                    lines.append(f"  H range: [{np.min(h_sub):.3f}, {np.max(h_sub):.3f}]")
                    lines.append(f"  K range: [{np.min(k_sub):.4g}, {np.max(k_sub):.4g}]")
                lines.append("")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        else:
            lines.append("<i>No boundary drawn yet.</i>")
            lines.append("Click and drag on the H-K plot to draw a boundary.")

        self._stats_label.setText("<br>".join(lines))

    def _on_export_classification(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Export classified trajectories to CSV files, per video and per region."""
        data = self._hk_canvas.get_region_data()
        if data['labels'] is None:
            QMessageBox.information(self, "No classification",
                                    "Draw a boundary first to classify trajectories.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select export folder",
                                                     self._output_dir or "")
        if not save_dir:
            return

        try:
            labels = data['labels']
            offset = 0
            exported_files = []

            for ds in self._loaded_datasets:
                df_diff = ds['diffusion_df']
                n = len(df_diff)
                ds_labels = labels[offset:offset + n]
                vname = ds['video_name']

                unique_regions = sorted(set(ds_labels.tolist()))
                for region_id in unique_regions:
                    suffix = f"region_{region_id}"
                    mask = ds_labels == region_id

                    region_df = df_diff[mask].copy()
                    region_df.to_csv(
                        os.path.join(save_dir, f"{vname}_{suffix}_diffusion.csv"),
                        index=False
                    )

                    region_traj_ids = set(df_diff['traj_idx'].values[mask].tolist())
                    traj_sub = ds['traces_df'][
                        ds['traces_df']['traj_idx'].isin(region_traj_ids)
                    ]
                    traj_sub.to_csv(
                        os.path.join(save_dir, f"{vname}_{suffix}_traces.csv"),
                        index=False
                    )
                    exported_files.append(f"{vname}_{suffix}")

                offset += n

            # Save boundary information
            boundaries_data = self._hk_canvas.get_boundaries_data()
            boundary_path = os.path.join(save_dir, "classification_boundaries.json")
            with open(boundary_path, 'w') as f:
                json.dump({'boundaries': boundaries_data}, f, indent=2)

            QMessageBox.information(self, "Export complete",
                                    f"Exported {len(exported_files)} region files "
                                    f"from {len(self._loaded_datasets)} video(s) to:\n{save_dir}\n\n"
                                    f"Boundary saved to: classification_boundaries.json")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
        # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

    def _auto_load_analysis(self, output_dir: str):
        """Auto-load H-K data into Analysis tab after a FreeTrace run."""
        if not os.path.isdir(output_dir):
            return
        diffusion_files = [
            f for f in os.listdir(output_dir) if f.endswith('_diffusion.csv')
        ]
        if diffusion_files:
            self._load_data_from_file(
                os.path.join(output_dir, diffusion_files[0])
            )  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17

    def _load_results(self, output_dir: str):
        # Clear previous dynamic result widgets
        for w in self._result_widgets:
            self._results_layout.removeWidget(w)
            w.deleteLater()
        self._result_widgets.clear()

        image_suffixes = {  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15 23:50
            "Trajectory Map": "_traces.png",
            "Localisation Density": "_loc_2d_density.png",
            "H-K Distribution": "_diffusion_distribution.png",
        }

        found = False
        if os.path.isdir(output_dir):
            for title, suffix in image_suffixes.items():
                matches = [
                    f for f in os.listdir(output_dir)
                    if f.endswith(suffix)
                ]
                for fname in sorted(matches):
                    fpath = os.path.join(output_dir, fname)
                    if not os.path.exists(fpath):
                        continue
                    found = True

                    header = QLabel(f"<b>{title}</b> - {fname}")
                    header.setStyleSheet(
                        f"color:#aaa; font-size:{self._f(15)}px; margin-top:12px;"
                    )
                    self._results_layout.addWidget(header)
                    self._result_widgets.append(header)

                    img_label = QLabel()
                    pixmap = QPixmap(fpath)
                    if not pixmap.isNull():
                        pixmap = pixmap.scaledToWidth(
                            600, Qt.TransformationMode.SmoothTransformation
                        )
                        img_label.setPixmap(pixmap)
                    else:
                        img_label.setText(f"(could not load {fname})")
                        img_label.setStyleSheet("color:#888;")
                    img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._results_layout.addWidget(img_label)
                    self._result_widgets.append(img_label)

            # Also show CSV files
            csv_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".csv"))
            if csv_files:
                header = QLabel(f"<b>Output files:</b>")
                header.setStyleSheet(
                    f"color:#aaa; font-size:{self._f(15)}px; margin-top:12px;"
                )
                self._results_layout.addWidget(header)
                self._result_widgets.append(header)
                found = True

                for csv_name in csv_files:
                    csv_label = QLabel(f"  {csv_name}")
                    csv_label.setStyleSheet(f"color:#8bc34a; font-size:{self._f(13)}px;")
                    self._results_layout.addWidget(csv_label)
                    self._result_widgets.append(csv_label)

        self._no_results_label.setVisible(not found)

    # ------------------------------------------------------------------
    # Dark theme
    # ------------------------------------------------------------------
    def _apply_dark_theme(self, f):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: #1e1e1e;
                color: #ddd;
                font-size: {f(14)}px;
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox {{
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: {f(14)}px;
                color: #eee;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid #7ec8e3;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{ /* Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16 */
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: {f(18)}px;
                border-left: 1px solid #555;
                border-bottom: 1px solid #555;
                border-top-right-radius: 4px;
                background: #3a3a3a;
            }}
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
                background: #4a4a4a;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: {f(18)}px;
                border-left: 1px solid #555;
                border-top: 1px solid #555;
                border-bottom-right-radius: 4px;
                background: #3a3a3a;
            }}
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
                background: #4a4a4a;
            }}
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ /* Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16 */
                image: url({_arrow_up_path});
                width: {f(10)}px; height: {f(10)}px;
            }}
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                image: url({_arrow_down_path});
                width: {f(10)}px; height: {f(10)}px;
            }} /* Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16 */
            QPushButton {{
                background: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: {f(14)}px;
                color: #ddd;
            }}
            QPushButton:hover {{ background: #4a4a4a; }}
            QCheckBox, QRadioButton {{ color: #ccc; spacing: 8px; font-size: {f(14)}px; }}
            QCheckBox::indicator, QRadioButton::indicator {{
                width: {f(17)}px; height: {f(17)}px;
                border: 1px solid #666;
                border-radius: 3px;
                background: #2a2a2a;
            }}
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
                background: #7ec8e3;
                border-color: #7ec8e3;
            }}
            QRadioButton::indicator {{ border-radius: {f(17) // 2}px; }}
            QRadioButton::indicator:checked {{ border-radius: {f(17) // 2}px; }}
            QTabWidget::pane {{ border: 1px solid #444; background: #1e1e1e; }}
            QTabWidget#mainTabs::pane {{ /* Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16 */
                border: none; border-top: 2px solid #444; background: #1e1e1e;
            }}
            QTabWidget#mainTabs > QTabBar::tab {{
                background: #2a2a2a; color: #999;
                padding: 10px 28px; font-size: {f(16)}px; font-weight: bold;
                border: 1px solid #444; border-bottom: none;
                border-radius: 6px 6px 0 0;
                margin-right: 4px; min-width: 120px;
            }}
            QTabWidget#mainTabs > QTabBar::tab:selected {{
                background: #1e1e1e; color: #7ec8e3;
                border-bottom: 2px solid #1e1e1e;
            }}
            QTabWidget#mainTabs > QTabBar::tab:hover:!selected {{
                background: #333; color: #ccc;
            }} /* Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16 */
            QTabBar::tab {{
                background: #2a2a2a; color: #aaa;
                padding: 8px 18px; font-size: {f(14)}px;
                border-radius: 4px 4px 0 0;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{ background: #1e1e1e; color: #fff; }}
            QProgressBar {{
                border: 1px solid #555; border-radius: 4px;
                background: #2a2a2a; color: #eee; text-align: center;
                font-size: {f(13)}px; height: {f(22)}px;
            }}
            QProgressBar::chunk {{ background: #2e7d32; border-radius: 3px; }}
            QScrollArea {{ border: none; }}
            QSplitter::handle {{ background: #333; }}
            QLabel {{ color: #ccc; font-size: {f(14)}px; }}
        """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    app.setApplicationName("FreeTrace")
    win = FreeTraceGUI()
    win.show()
    ret = app.exec()
    # Cleanup temp arrow icons
    import shutil as _shutil
    _shutil.rmtree(_arrow_dir, ignore_errors=True)
    sys.exit(ret) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


if __name__ == "__main__":
    main()
