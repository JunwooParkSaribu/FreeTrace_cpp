# Made by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
# FreeTrace GUI — launches the freetrace binary with a graphical interface.
# Launch with:  python gui.py
"""
FreeTrace GUI — run localization and tracking by clicking.
Requires PyQt6: pip install PyQt6
"""
import atexit # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
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
    QGraphicsPathItem,
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
class HKGatingCanvas(QGraphicsView):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
    """Interactive H-K scatter plot with freehand gating.

    Users draw a boundary curve that divides the H-K space into two regions.
    Trajectories are classified based on which side of the boundary they fall.
    """
    gating_changed = pyqtSignal()

    _MARGIN_LEFT = 60
    _MARGIN_BOTTOM = 50
    _MARGIN_TOP = 30
    _MARGIN_RIGHT = 30
    _PLOT_W = 500
    _PLOT_H = 400

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

        self._drawing = False
        self._boundary_points = []
        self._boundary_path_item = None
        self._dot_items = []
        self._region_labels = None

        self._color_a = QColor(100, 180, 255, 200)
        self._color_b = QColor(255, 120, 80, 200)
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
        self._boundary_path_item = None

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
                color = self._color_a if self._region_labels[i] == 0 else self._color_b
            dot = QGraphicsEllipseItem(x - dot_r, y - dot_r, dot_r * 2, dot_r * 2)
            dot.setPen(QPen(Qt.PenStyle.NoPen))
            dot.setBrush(QBrush(color))
            self._scene.addItem(dot)
            self._dot_items.append(dot)

        if self._boundary_points:
            self._draw_boundary_path()

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._scene.sceneRect().width() > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _clamp_to_plot(self, pos):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
        x = max(self._MARGIN_LEFT, min(pos.x(), self._MARGIN_LEFT + self._PLOT_W))
        y = max(self._MARGIN_TOP, min(pos.y(), self._MARGIN_TOP + self._PLOT_H))
        return QPointF(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and len(self._H) > 0:
            self._drawing = True
            self._boundary_points = []
            if self._boundary_path_item and self._boundary_path_item.scene():
                self._scene.removeItem(self._boundary_path_item)
                self._boundary_path_item = None
            pos = self._clamp_to_plot(self.mapToScene(event.pos()))
            self._boundary_points.append(pos)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing:
            pos = self._clamp_to_plot(self.mapToScene(event.pos()))
            self._boundary_points.append(pos)
            self._draw_boundary_path()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            if len(self._boundary_points) >= 2:
                self._extend_boundary_to_edges()
                self._draw_boundary_path()
                self._classify_points()
                self._update_dot_colors()
                self.gating_changed.emit()
        super().mouseReleaseEvent(event)

    def _draw_boundary_path(self):
        if self._boundary_path_item and self._boundary_path_item.scene():
            self._scene.removeItem(self._boundary_path_item)
        path = QPainterPath()
        path.moveTo(self._boundary_points[0])
        for pt in self._boundary_points[1:]:
            path.lineTo(pt)
        pen = QPen(QColor(255, 255, 0, 220), 2.0)
        self._boundary_path_item = self._scene.addPath(path, pen)

    def _extend_boundary_to_edges(self):
        if len(self._boundary_points) < 2:
            return
        plot_left = self._MARGIN_LEFT
        plot_right = self._MARGIN_LEFT + self._PLOT_W
        plot_top = self._MARGIN_TOP
        plot_bottom = self._MARGIN_TOP + self._PLOT_H

        def _extend_to_edge(pt, direction_pt):
            dx = pt.x() - direction_pt.x()
            dy = pt.y() - direction_pt.y()
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-6:
                return pt
            dx /= length
            dy /= length
            candidates = []
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
            if candidates:
                candidates.sort(key=lambda c: c[0])
                return candidates[0][1]
            return pt

        start_ext = _extend_to_edge(self._boundary_points[0], self._boundary_points[1])
        end_ext = _extend_to_edge(self._boundary_points[-1], self._boundary_points[-2])
        self._boundary_points.insert(0, start_ext)
        self._boundary_points.append(end_ext)

    def _classify_points(self):
        if len(self._boundary_points) < 2:
            self._region_labels = None
            return
        n_pts = len(self._H)
        self._region_labels = np.zeros(n_pts, dtype=int)
        bx = [p.x() for p in self._boundary_points]
        by = [p.y() for p in self._boundary_points]
        for i in range(n_pts):
            px = self._h_to_x(self._H[i])
            py = self._logk_to_y(self._log_K[i])
            crossings = 0
            for j in range(len(bx) - 1):
                y1, y2 = by[j], by[j + 1]
                x1, x2 = bx[j], bx[j + 1]
                if (y1 <= py < y2) or (y2 <= py < y1):
                    t = (py - y1) / (y2 - y1)
                    x_intersect = x1 + t * (x2 - x1)
                    if x_intersect > px:
                        crossings += 1
            self._region_labels[i] = crossings % 2

    def _update_dot_colors(self):
        for i, dot in enumerate(self._dot_items):
            if dot is None:
                continue
            if self._region_labels is not None:
                color = self._color_a if self._region_labels[i] == 0 else self._color_b
            else:
                color = self._color_default
            dot.setBrush(QBrush(color))

    def _clear_boundary(self):
        self._boundary_points = []
        if self._boundary_path_item and self._boundary_path_item.scene():
            self._scene.removeItem(self._boundary_path_item)
            self._boundary_path_item = None
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
            mask_a = self._region_labels == 0
            mask_b = self._region_labels == 1
            result['region_a'] = np.where(mask_a)[0]
            result['region_b'] = np.where(mask_b)[0]
        else:
            result['region_a'] = np.arange(len(self._H))
            result['region_b'] = np.array([], dtype=int)
        return result  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class FreeTraceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FreeTrace v1.6.0.5") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
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

    def _build_analysis_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Toolbar row
        toolbar = QHBoxLayout()
        self._analysis_load_btn = QPushButton("Load Data")
        self._analysis_load_btn.clicked.connect(self._on_load_data)
        toolbar.addWidget(self._analysis_load_btn)

        self._analysis_clear_btn = QPushButton("Clear Boundary")
        self._analysis_clear_btn.clicked.connect(self._on_clear_gating)
        toolbar.addWidget(self._analysis_clear_btn)

        self._analysis_export_btn = QPushButton("Export Classification")
        self._analysis_export_btn.clicked.connect(self._on_export_classification)
        toolbar.addWidget(self._analysis_export_btn)

        toolbar.addStretch()

        self._analysis_info_label = QLabel("Draw a boundary on the H-K plot to classify trajectories.")
        self._analysis_info_label.setStyleSheet("color:#888;")
        toolbar.addWidget(self._analysis_info_label)

        layout.addLayout(toolbar)

        # Main content: H-K canvas | trajectory view | stats
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._hk_canvas = HKGatingCanvas()
        self._hk_canvas.gating_changed.connect(self._on_gating_changed)
        self._hk_canvas.setMinimumSize(400, 350)
        content_splitter.addWidget(self._hk_canvas)

        # Trajectory visualization
        self._traj_view = QGraphicsView()
        self._traj_scene = QGraphicsScene()
        self._traj_view.setScene(self._traj_scene)
        self._traj_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._traj_view.setStyleSheet("background:#1a1a1a; border:none;")
        self._traj_view.setMinimumSize(350, 350)
        content_splitter.addWidget(self._traj_view)

        # Statistics panel
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_widget = QWidget()
        self._stats_layout = QVBoxLayout(stats_widget)
        self._stats_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._stats_label = QLabel("No data loaded.\n\nClick 'Load Data' or run FreeTrace first.")
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet("color:#aaa; font-size:13px; padding:12px;")
        self._stats_layout.addWidget(self._stats_label)

        stats_scroll.setWidget(stats_widget)
        stats_scroll.setMinimumWidth(220)
        content_splitter.addWidget(stats_scroll)

        content_splitter.setSizes([400, 400, 250])
        layout.addWidget(content_splitter)

        self._analysis_diffusion_path = None
        self._analysis_traces_path = None
        self._analysis_traces_df = None
        self._analysis_video_name = None

        return tab  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17

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

        # Buttons
        btn_row = QHBoxLayout()
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
        self._stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Analysis tab slots  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
    # ------------------------------------------------------------------
    def _on_load_data(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
        start_dir = self._output_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FreeTrace output CSV (*_diffusion.csv or *_traces.csv)",
            start_dir,
            "FreeTrace CSV (*_diffusion.csv *_traces.csv);;All CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        self._load_data_from_file(path)

    def _load_data_from_file(self, selected_path):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
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

            if not os.path.exists(diffusion_path):
                QMessageBox.warning(self, "File not found",
                                    f"Diffusion file not found:\n{diffusion_path}")
                return
            if not os.path.exists(traces_path):
                QMessageBox.warning(self, "File not found",
                                    f"Traces file not found:\n{traces_path}")
                return

            df = pd.read_csv(diffusion_path)
            if 'H' not in df.columns or 'K' not in df.columns or 'traj_idx' not in df.columns:
                QMessageBox.warning(self, "Invalid file",
                                    "Diffusion CSV must contain columns: traj_idx, H, K")
                return

            self._analysis_diffusion_path = diffusion_path
            self._analysis_traces_path = traces_path
            self._analysis_traces_df = pd.read_csv(traces_path)

            fname = os.path.basename(diffusion_path)
            self._analysis_video_name = fname.replace('_diffusion.csv', '')

            self._hk_canvas.set_data(
                df['traj_idx'].values, df['H'].values, df['K'].values
            )

            n = len(df)
            self._analysis_info_label.setText(
                f"Loaded {n} trajectories. Draw a boundary to classify."
            )
            self._update_stats_display()
            self._draw_trajectories()
        except Exception as e:
            QMessageBox.critical(self, "Error loading data", str(e))

    def _on_clear_gating(self):
        self._hk_canvas.clear_gating()

    def _on_gating_changed(self):
        self._update_stats_display()
        self._draw_trajectories()

    def _draw_trajectories(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
        """Draw trajectories on the trajectory view, colored by region classification."""
        self._traj_scene.clear()
        if self._analysis_traces_df is None:
            txt = self._traj_scene.addSimpleText("No trajectory data available.")
            txt.setBrush(QColor(150, 150, 150))
            return

        df = self._analysis_traces_df
        data = self._hk_canvas.get_region_data()
        labels = data['labels']
        traj_indices = data['traj_indices']

        label_map = {}
        if labels is not None:
            for i, tidx in enumerate(traj_indices):
                label_map[int(tidx)] = int(labels[i])

        x_max = df['x'].max()
        y_max = df['y'].max()
        canvas_w = max(x_max + 10, 100)
        canvas_h = max(y_max + 10, 100)

        self._traj_scene.setSceneRect(0, 0, canvas_w, canvas_h)
        self._traj_scene.addRect(
            QRectF(0, 0, canvas_w, canvas_h),
            QPen(Qt.PenStyle.NoPen), QBrush(QColor(0, 0, 0))
        )

        color_a = QColor(100, 180, 255, 200)
        color_b = QColor(255, 120, 80, 200)
        rng_colors = {}

        for tidx in df['traj_idx'].unique():
            traj_data = df[df['traj_idx'] == tidx].sort_values('frame')
            positions = list(zip(traj_data['x'].values, traj_data['y'].values))
            if len(positions) < 2:
                continue

            if labels is not None:
                region = label_map.get(int(tidx), 0)
                color = color_a if region == 0 else color_b
            else:
                if tidx not in rng_colors:
                    rng = np.random.default_rng(int(tidx))
                    rgb = rng.integers(low=50, high=256, size=3)
                    rng_colors[tidx] = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), 200)
                color = rng_colors[tidx]

            pen = QPen(color, 0.5)
            path = QPainterPath()
            path.moveTo(positions[0][0], positions[0][1])
            for x, y in positions[1:]:
                path.lineTo(x, y)
            self._traj_scene.addPath(path, pen)

        self._traj_view.fitInView(
            self._traj_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

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

        if labels is not None:
            for region_id, region_name, color_hex in [(0, "Region A (blue)", "#64b4ff"),
                                                       (1, "Region B (orange)", "#ff7850")]:
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
                lines.append("")
        else:
            lines.append("<i>No boundary drawn yet.</i>")
            lines.append("Click and drag on the H-K plot to draw a boundary.")

        self._stats_label.setText("<br>".join(lines))

    def _on_export_classification(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-17
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
            H, K = data['H'], data['K']
            traj_idx = data['traj_indices']
            labels = data['labels']
            vname = getattr(self, '_analysis_video_name', 'classified')

            for region_id, suffix in [(0, "region_A"), (1, "region_B")]:
                mask = labels == region_id
                region_df = pd.DataFrame({
                    'traj_idx': traj_idx[mask],
                    'H': H[mask],
                    'K': K[mask],
                })
                region_df.to_csv(
                    os.path.join(save_dir, f"{vname}_{suffix}_diffusion.csv"),
                    index=False
                )
                if self._analysis_traces_df is not None:
                    region_traj_ids = set(traj_idx[mask].tolist())
                    traj_sub = self._analysis_traces_df[
                        self._analysis_traces_df['traj_idx'].isin(region_traj_ids)
                    ]
                    traj_sub.to_csv(
                        os.path.join(save_dir, f"{vname}_{suffix}_traces.csv"),
                        index=False
                    )

            QMessageBox.information(self, "Export complete",
                                    f"Classification exported to:\n{save_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

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
