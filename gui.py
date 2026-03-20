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
from scipy.optimize import minimize  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
import matplotlib  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

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

# Current version — used for update check against GitHub releases  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
_VERSION = "1.6.1.0"
_GITHUB_REPO = "JunwooParkSaribu/FreeTrace_cpp"

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

    candidates = [ # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        # Inside .app/Contents/MacOS/ (renamed to avoid case collision with PyInstaller exe)
        os.path.join(script_dir, "freetrace-bin"),
        # Same dir as gui.py or next to exe
        os.path.join(script_dir, "freetrace"),
        os.path.join(script_dir, "freetrace.exe"),
        # build_gpu/ subdirectory (GPU development build — preferred over CPU)
        os.path.join(script_dir, "build_gpu", "freetrace"),
        os.path.join(script_dir, "build_gpu", "freetrace.exe"),
        # build/ subdirectory (development)
        os.path.join(script_dir, "build", "freetrace"),
        os.path.join(script_dir, "build", "freetrace.exe"),
    ] # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
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
        self._batch_summary = ""  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

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
            if sys.platform == "win32":  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                )
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
            self._error_lines = []  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            for line in self._process.stdout:
                line = line.rstrip("\n")
                # Show each libtiff warning only once per run
                if line.startswith("TIFFReadDirectory: Warning"):
                    if line not in seen_warnings:
                        seen_warnings.add(line)
                        self.log.emit(line)
                    continue
                self.log.emit(line)
                # Collect error/failure lines for the error dialog
                line_lower = line.lower()
                if any(kw in line_lower for kw in ("error", "failed", "cannot", "exception", "err:")):
                    self._error_lines.append(line)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20

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
                    self._batch_summary = line  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                    self.progress.emit(95, "Finishing")

                if self._cancel:
                    break

            self._process.wait()
            rc = self._process.returncode

            if self._cancel:
                self.finished.emit(False, "Cancelled by user.")
            elif rc == 0:
                self.progress.emit(100, "Done")
                self.finished.emit(True, self.output_dir)
            elif self.batch and self._batch_summary:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                # Partial success in batch mode — not a critical error
                self.progress.emit(100, "Done (with errors)")
                self.finished.emit(True, f"BATCH_PARTIAL|{self.output_dir}|{self._batch_summary}")
            else:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
                if self._error_lines:
                    detail = "\n".join(self._error_lines[-5:])  # last 5 error lines
                    self.finished.emit(False, f"{detail}")
                else:
                    self.finished.emit(False, f"Process exited with code {rc}")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20

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
            self.log.emit("Reading video...")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
            if self.video_path.lower().endswith(".nd2"):
                import nd2
                all_imgs = np.array(nd2.imread(self.video_path))
                if len(all_imgs.shape) == 2:
                    all_imgs = all_imgs[np.newaxis, ...]
                total = len(all_imgs)
                n = min(self.n_frames, total)
                start = max(0, total // 2 - n // 2)
                imgs = all_imgs[start:start + n]
            else:
                with tifffile.TiffFile(self.video_path) as tif:
                    total = len(tif.pages)
                    n = min(self.n_frames, total)
                    start = max(0, total // 2 - n // 2)
                    imgs = np.stack([tif.pages[i].asarray() for i in range(start, start + n)])  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
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
            popen_kw = dict(  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            if sys.platform == "win32":
                popen_kw["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                )
            proc = subprocess.Popen(cmd, **popen_kw)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
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
# Preprocessing helpers for Basic Stats (standalone, no FreeTrace dependency)
# ---------------------------------------------------------------------------
def _unit_vector(vector):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _angle_between(v1, v2):
    v1_u = _unit_vector(v1)
    v2_u = _unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def _dot_product_angle(v1, v2):
    ang = _angle_between(v1, v2)
    if ang == np.inf or ang == np.nan or math.isnan(ang):
        return 0
    return 180 - ang


# ---------------------------------------------------------------------------
# Distribution helpers for Adv Stats (standalone, no FreeTrace dependency)
# ---------------------------------------------------------------------------
def _pdf_gaussian(x, mu, sigma, alpha):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    """Scaled Gaussian PDF: alpha * N(x; mu, sigma)."""
    return alpha / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)



def _pdf_cauchy(u, h, t=1, s=1):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    """PDF of the Cauchy distribution for fBm displacement ratio."""
    assert 0 < h < 1
    return 1/(np.pi * np.sqrt(1-(2**(2*h-1)-1)**2)) * 1 / (
        (u - (2**(2*h-1) - 1)*(t/s)**h)**2 / ((1 - (2**(2*h-1)-1)**2)*np.sqrt((t/s)**(2*h)))
        + (np.sqrt((t/s)**(2*h))))


def _pdf_cauchy_1mixture(x, h1, alpha):
    """Scaled single-component Cauchy PDF."""
    return alpha * _pdf_cauchy(x, h1)


def _cauchy_location(h1, t=1, s=1):
    """Peak location of the Cauchy distribution given H."""
    return (2**(2*h1-1) - 1)*(t/s)**h1


def _func_to_minimise(params, func, x, y):
    """Objective for scipy.optimize.minimize — L1 norm."""
    y_pred = func(x, *params)
    return np.sum(abs(y_pred - y))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19


def _preprocess_for_stats(data, pixelmicrons, framerate, cutoff_min=3, cutoff_max=99999,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                          has_diffusion=True):
    """Standalone preprocessing for Basic Stats tab.

    Simplified from simple_preprocessing() — excludes TAMSD, Markov chain,
    networkx, tqdm, color_palette, trajectory_visualization.

    Trajectories are split only by state change, not by frame gaps.
    Per-step statistics (jump distance, angles) use only consecutive-frame
    steps (df==1), while per-trajectory metrics (duration, MSD) span the
    full observation including gaps.

    Returns (analysis_data1, analysis_data2, analysis_data3, msd, total_states)
    """
    data = data.dropna()
    if 'state' in data.columns:
        data = data.astype({'state': int})
    else:
        data['state'] = 0

    traj_indices = pd.unique(data['traj_idx'])
    total_states = sorted(data['state'].unique())

    if len(data) == 0:
        return None, None, None, None, total_states

    # State re-ordering w.r.t. K (only if diffusion data available)
    if has_diffusion and 'K' in data.columns and len(total_states) > 1:
        avg_ks = []
        for st in total_states:
            avg_ks.append(data['K'][data['state'] == st].mean())
        avg_ks = np.array(avg_ks)
        prev_states = np.argsort(avg_ks)
        state_reorder = {st: idx for idx, st in enumerate(prev_states)}
        ordered_states = np.empty(len(data['state']), dtype=np.uint8)
        for st_idx in range(len(ordered_states)):
            ordered_states[st_idx] = state_reorder[data['state'].iloc[st_idx]]
        data['state'] = ordered_states
        total_states = sorted(data['state'].unique())

    dim = 2
    # Max trajectory duration (not max absolute frame number)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    max_frame = int(data.groupby('traj_idx')['frame'].apply(lambda f: f.max() - f.min()).max())

    analysis_data1 = {'mean_jump_d': [], 'state': [], 'duration': [], 'traj_id': []}
    if has_diffusion:
        analysis_data1['K'] = []
        analysis_data1['H'] = []
    analysis_data2 = {'2d_displacement': [], 'state': []}
    analysis_data3 = {'angle': [], 'polar_angle': [], 'state': []}
    msd_ragged_ens_trajs = {st: [] for st in total_states}
    msd = {'mean': [], 'std': [], 'nb_data': [], 'state': [], 'time': []}

    n_done = 0
    n_total = len(traj_indices)

    for traj_idx in traj_indices:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        single_traj = data.loc[data['traj_idx'] == traj_idx].copy()
        single_traj = single_traj.sort_values(by=['frame'])

        # Chunk by state only (no splitting by frame gap)
        before_st = single_traj.state.iloc[0]
        state_chunk_idx = [0, len(single_traj)]
        for st_idx, st in enumerate(single_traj.state):
            if st != before_st:
                state_chunk_idx.append(st_idx)
            before_st = st
        state_chunk_idx = sorted(state_chunk_idx)

        for si in range(len(state_chunk_idx) - 1):
            sub_traj = single_traj.iloc[state_chunk_idx[si]:state_chunk_idx[si + 1]].copy()
            if not (cutoff_min <= len(sub_traj) <= cutoff_max):
                continue

            state = sub_traj.state.iloc[0]

            # Convert to microns
            sub_traj.x = sub_traj.x * pixelmicrons
            sub_traj.y = sub_traj.y * pixelmicrons

            frame_diffs = sub_traj.frame.iloc[1:].to_numpy() - sub_traj.frame.iloc[:-1].to_numpy()
            duration = np.sum(frame_diffs) * framerate

            # Coordinate rescale (origin at first point)
            sub_traj.x = sub_traj.x - sub_traj.x.iloc[0]
            sub_traj.y = sub_traj.y - sub_traj.y.iloc[0]

            # All step vectors
            dx = sub_traj.x.iloc[1:].to_numpy() - sub_traj.x.iloc[:-1].to_numpy()
            dy = sub_traj.y.iloc[1:].to_numpy() - sub_traj.y.iloc[:-1].to_numpy()

            # Filter: only consecutive-frame steps (df == 1) for jump distances
            consecutive = (frame_diffs == 1)
            dx_consec = dx[consecutive]
            dy_consec = dy[consecutive]
            jump_distances = np.sqrt(dx_consec ** 2 + dy_consec ** 2)

            # Angles: only from pairs of consecutive steps (both must be df==1)
            vecs_consec = np.vstack([dx_consec, dy_consec]).T
            angles = []
            polar_angles = []
            for vi in range(len(vecs_consec) - 1):
                angles.append(_dot_product_angle(vecs_consec[vi], vecs_consec[vi + 1]))
                cross = vecs_consec[vi][0] * vecs_consec[vi + 1][1] - vecs_consec[vi][1] * vecs_consec[vi + 1][0]
                dot = vecs_consec[vi][0] * vecs_consec[vi + 1][0] + vecs_consec[vi][1] * vecs_consec[vi + 1][1]
                ang = np.degrees(np.arctan2(cross, dot))
                polar_angles.append(ang % 360)

            # Ensemble averaged SD (MSD) — uses absolute positions, handles gaps naturally
            copy_frames = sub_traj.frame.to_numpy()
            copy_frames = copy_frames - copy_frames[0]
            tmp_msd = []
            sq_disp = (sub_traj.x.to_numpy() ** 2 + sub_traj.y.to_numpy() ** 2) / dim / 2
            for frame_val, sd in zip(np.arange(0, copy_frames[-1], 1), sq_disp):
                if frame_val in copy_frames:
                    tmp_msd.append(sd)
                else:
                    tmp_msd.append(None)
            msd_ragged_ens_trajs[state].append(tmp_msd)

            # Store data1 (per-trajectory summary)
            if len(jump_distances) > 0:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                analysis_data1['mean_jump_d'].append(jump_distances.mean())
            else:
                analysis_data1['mean_jump_d'].append(np.nan)  # no consecutive-frame steps → exclude from plot
            analysis_data1['state'].append(state)
            analysis_data1['duration'].append(duration)
            analysis_data1['traj_id'].append(sub_traj.traj_idx.iloc[0])
            if has_diffusion:
                analysis_data1['H'].append(sub_traj.H.iloc[0] if 'H' in sub_traj.columns else 0)
                analysis_data1['K'].append(sub_traj.K.iloc[0] if 'K' in sub_traj.columns else 0)

            # Store data2 (per-step jump distances, consecutive frames only)
            analysis_data2['2d_displacement'].extend(list(jump_distances))
            analysis_data2['state'].extend([state] * len(jump_distances))

            # Store data3 (angles from consecutive step pairs only)
            analysis_data3['angle'].extend(angles)
            analysis_data3['polar_angle'].extend(polar_angles)
            analysis_data3['state'].extend([state] * len(angles))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

        n_done += 1

    # Calculate MSD averages per state
    for state_key in total_states:
        msd_mean = []
        msd_std = []
        msd_nb_data = []
        for t in range(max_frame):
            msd_row_data = []
            for row in range(len(msd_ragged_ens_trajs[state_key])):
                if t < len(msd_ragged_ens_trajs[state_key][row]) and msd_ragged_ens_trajs[state_key][row][t] is not None:
                    msd_row_data.append(msd_ragged_ens_trajs[state_key][row][t])
            msd_mean.append(np.mean(msd_row_data) if msd_row_data else np.nan)
            msd_std.append(np.std(msd_row_data) if msd_row_data else np.nan)
            msd_nb_data.append(len(msd_row_data))

        times = np.arange(0, max_frame) * framerate
        msd['mean'].extend(msd_mean)
        msd['std'].extend(msd_std)
        msd['nb_data'].extend(msd_nb_data)
        msd['state'].extend([state_key] * max_frame)
        msd['time'].extend(times)

    analysis_data1 = pd.DataFrame(analysis_data1).astype({'state': int, 'duration': float, 'traj_id': str})
    analysis_data2 = pd.DataFrame(analysis_data2)
    analysis_data3 = pd.DataFrame(analysis_data3)
    msd = pd.DataFrame(msd)

    return analysis_data1, analysis_data2, analysis_data3, msd, total_states  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19


def _preprocess_for_adv_stats(data, pixelmicrons, framerate, cutoff_min=3, cutoff_max=99999):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    """Preprocessing for Advanced Stats tab — TAMSD, 1D displacements, 1D ratios + Cauchy/Gaussian fits.

    Works with traces-only data (no H/K needed).
    Returns (tamsd_df, displacements_1d_df, ratios_1d_df, cauchy_fits, gaussian_fits, total_states)
    """
    data = data.dropna()
    if 'state' in data.columns:
        data = data.astype({'state': int})
    else:
        data['state'] = 0

    traj_indices = pd.unique(data['traj_idx'])
    total_states = sorted(data['state'].unique())

    if len(data) == 0:
        return None, None, None, None, None, total_states  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

    dim = 2

    # TAMSD ragged arrays per state
    tamsd_ragged = {st: [] for st in total_states}
    # 1D displacements
    disp_1d = {'dx': [], 'dy': [], 'state': []}
    # 1D ratios
    ratios_1d = {'ratio': [], 'state': []}

    for traj_idx in traj_indices:
        single_traj = data.loc[data['traj_idx'] == traj_idx].copy()
        single_traj = single_traj.sort_values(by=['frame'])

        # Chunk by state only
        before_st = single_traj.state.iloc[0]
        state_chunk_idx = [0, len(single_traj)]
        for st_idx, st in enumerate(single_traj.state):
            if st != before_st:
                state_chunk_idx.append(st_idx)
            before_st = st
        state_chunk_idx = sorted(state_chunk_idx)

        for si in range(len(state_chunk_idx) - 1):
            sub_traj = single_traj.iloc[state_chunk_idx[si]:state_chunk_idx[si + 1]].copy()
            if not (cutoff_min <= len(sub_traj) <= cutoff_max):
                continue

            state = sub_traj.state.iloc[0]

            # Convert to microns
            sub_traj.x = sub_traj.x * pixelmicrons
            sub_traj.y = sub_traj.y * pixelmicrons

            # Coordinate rescale (origin at first point)
            x0 = sub_traj.x.iloc[0]
            y0 = sub_traj.y.iloc[0]
            sub_traj.x = sub_traj.x - x0
            sub_traj.y = sub_traj.y - y0

            frames = sub_traj.frame.to_numpy()
            xs = sub_traj.x.to_numpy()
            ys = sub_traj.y.to_numpy()
            n_pts = len(xs)

            # --- TAMSD: for each lag τ, average SD over all valid windows ---
            tmp_tamsd = []
            for tau in range(1, n_pts):
                sd_vals = []
                for i in range(n_pts - tau):
                    actual_gap = frames[i + tau] - frames[i]
                    if actual_gap == tau:
                        sd = ((xs[i + tau] - xs[i])**2 + (ys[i + tau] - ys[i])**2) / dim / 2
                        sd_vals.append(sd)
                if sd_vals:
                    tmp_tamsd.append(np.mean(sd_vals))
                else:
                    tmp_tamsd.append(None)
            tamsd_ragged[state].append(tmp_tamsd)

            # --- 1D Displacements (consecutive frames only, Δt=1) ---
            frame_diffs = frames[1:] - frames[:-1]
            consecutive = (frame_diffs == 1)
            dx_consec = (xs[1:] - xs[:-1])[consecutive]
            dy_consec = (ys[1:] - ys[:-1])[consecutive]
            disp_1d['dx'].extend(dx_consec.tolist())
            disp_1d['dy'].extend(dy_consec.tolist())
            disp_1d['state'].extend([state] * len(dx_consec))

            # --- 1D Ratios (consecutive displacement ratios) ---
            if len(dx_consec) > 1:
                rx = dx_consec[1:] / dx_consec[:-1]
                ry = dy_consec[1:] / dy_consec[:-1]
                ratios = np.concatenate([rx, ry])
                valid = np.isfinite(ratios)
                ratios_1d['ratio'].extend(ratios[valid].tolist())
                ratios_1d['state'].extend([state] * int(valid.sum()))

    # --- Ensemble average TAMSD per state ---
    tamsd = {'mean': [], 'std': [], 'nb_data': [], 'state': [], 'time': []}
    for state_key in total_states:
        tamsd_mean = []
        tamsd_std = []
        tamsd_nb = []
        max_lag = max((len(row) for row in tamsd_ragged[state_key]), default=0)
        for t in range(max_lag):
            row_data = []
            for row in tamsd_ragged[state_key]:
                if t < len(row) and row[t] is not None:
                    row_data.append(row[t])
            tamsd_mean.append(np.mean(row_data) if row_data else np.nan)
            tamsd_std.append(np.std(row_data) if row_data else np.nan)
            tamsd_nb.append(len(row_data))
        times = (np.arange(1, max_lag + 1)) * framerate
        tamsd['mean'].extend(tamsd_mean)
        tamsd['std'].extend(tamsd_std)
        tamsd['nb_data'].extend(tamsd_nb)
        tamsd['state'].extend([state_key] * max_lag)
        tamsd['time'].extend(times.tolist())

    tamsd_df = pd.DataFrame(tamsd)
    displacements_1d_df = pd.DataFrame(disp_1d)
    ratios_1d_df = pd.DataFrame(ratios_1d)

    # --- Cauchy fit per state ---
    cauchy_fits = {}
    for state_key in total_states:
        try:
            subset = ratios_1d_df[ratios_1d_df['state'] == state_key]['ratio'].to_numpy()
            if len(subset) < 10:
                continue
            # Clip to reasonable range for histogram
            subset_clipped = subset[(subset > -10) & (subset < 10)]
            if len(subset_clipped) < 10:
                continue
            counts, bin_edges = np.histogram(subset_clipped, bins=100, density=True)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

            result = minimize(  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                _func_to_minimise, x0=[0.5, 1.0],
                args=(_pdf_cauchy_1mixture, bin_centres, counts),
                method='Nelder-Mead',
            )
            h_est, alpha_est = np.clip(result.x[0], 0.01, 0.99), max(result.x[1], 0.01)
            loc_est = _cauchy_location(h_est)
            x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
            y_fit = _pdf_cauchy_1mixture(x_fit, h_est, alpha_est)
            cauchy_fits[state_key] = {
                'h_est': h_est, 'alpha_est': alpha_est, 'location': loc_est,
                'x_fit': x_fit, 'y_fit': y_fit,
                'bin_centres': bin_centres, 'counts': counts, 'bin_edges': bin_edges,
            }
        except Exception:
            pass

    # --- Gaussian fit per state for 1D displacements ---
    gaussian_fits = {}  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    for state_key in total_states:
        try:
            subset = displacements_1d_df[displacements_1d_df['state'] == state_key]
            dx_arr = subset['dx'].to_numpy()
            dy_arr = subset['dy'].to_numpy()
            if len(dx_arr) < 10:
                continue
            fits = {}
            for axis_name, arr in [('dx', dx_arr), ('dy', dy_arr)]:
                counts, bin_edges = np.histogram(arr, bins=80, density=True)
                bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                # Initial guess: mean, std, amplitude=1
                mu0, sigma0 = np.mean(arr), np.std(arr)
                result = minimize(  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                    _func_to_minimise, x0=[mu0, max(sigma0, 1e-6), 1.0],
                    args=(_pdf_gaussian, bin_centres, counts),
                    method='Nelder-Mead',
                )
                mu_est = result.x[0]
                sigma_est = max(result.x[1], 1e-8)
                alpha_est = max(result.x[2], 0.01)
                x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
                y_fit = _pdf_gaussian(x_fit, mu_est, sigma_est, alpha_est)
                fits[axis_name] = {
                    'mu': mu_est, 'sigma': sigma_est, 'alpha': alpha_est,
                    'x_fit': x_fit, 'y_fit': y_fit,
                }
            gaussian_fits[state_key] = fits
        except Exception:
            pass

    return tamsd_df, displacements_1d_df, ratios_1d_df, cauchy_fits, gaussian_fits, total_states  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19


# ---------------------------------------------------------------------------
# StatsWorker — runs preprocessing in background thread
# ---------------------------------------------------------------------------
class StatsWorker(QThread):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)  # list of per-dataset result dicts
    error = pyqtSignal(str)

    def __init__(self, datasets, pixelmicrons, framerate, cutoff_min):
        """datasets: list of (name, DataFrame, has_diffusion) tuples."""
        super().__init__()
        self._datasets = datasets
        self._pixelmicrons = pixelmicrons
        self._framerate = framerate
        self._cutoff_min = cutoff_min

    def run(self):
        try:
            all_results = []
            n = len(self._datasets)
            for idx, (name, data, has_diff) in enumerate(self._datasets):
                pct = int(10 + 80 * idx / max(n, 1))
                self.progress.emit(pct, f"Processing {name}...")
                result = _preprocess_for_stats(
                    data, self._pixelmicrons, self._framerate,
                    cutoff_min=self._cutoff_min, has_diffusion=has_diff,
                )
                ad1, ad2, ad3, msd, total_states = result
                if ad1 is not None and len(ad1) > 0:
                    all_results.append({
                        'name': name,
                        'analysis_data1': ad1,
                        'analysis_data2': ad2,
                        'analysis_data3': ad3,
                        'msd': msd,
                        'total_states': total_states,
                        'has_diffusion': has_diff,
                    })
            if not all_results:
                self.error.emit("No data remaining after filtering.")
                return
            self.progress.emit(95, "Building results...")
            self.finished.emit(all_results)
        except Exception as e:
            self.error.emit(str(e))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19


class AdvStatsWorker(QThread):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    """Background worker for Advanced Stats — runs TAMSD + 1D displacement + Cauchy fit."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, datasets, pixelmicrons, framerate, cutoff_min):
        """datasets: list of (name, DataFrame) tuples."""
        super().__init__()
        self._datasets = datasets
        self._pixelmicrons = pixelmicrons
        self._framerate = framerate
        self._cutoff_min = cutoff_min

    def run(self):
        try:
            all_results = []
            n = len(self._datasets)
            for idx, (name, data) in enumerate(self._datasets):
                pct = int(10 + 80 * idx / max(n, 1))
                self.progress.emit(pct, f"Processing {name}...")
                result = _preprocess_for_adv_stats(
                    data, self._pixelmicrons, self._framerate,
                    cutoff_min=self._cutoff_min,
                )
                tamsd_df, disp_df, ratios_df, cauchy_fits, gaussian_fits, total_states = result  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                n_trajs = int(data['traj_idx'].nunique())
                if tamsd_df is not None and len(tamsd_df) > 0:
                    all_results.append({
                        'name': name,
                        'tamsd': tamsd_df,
                        'displacements_1d': disp_df,
                        'ratios_1d': ratios_df,
                        'cauchy_fits': cauchy_fits,
                        'gaussian_fits': gaussian_fits,
                        'total_states': total_states,
                        'n_trajectories': n_trajs,
                    })
            if not all_results:
                self.error.emit("No data remaining after filtering.")
                return
            self.progress.emit(95, "Building results...")
            self.finished.emit(all_results)
        except Exception as e:
            self.error.emit(str(e))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19


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
        self._dot_pixmap_item = None      # single pixmap for all dots  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        self._dot_coords = []             # (x, y) per point or None if out of bounds
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

    def _draw_plot(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        self._scene.clear()
        self._dot_pixmap_item = None
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

        self._dot_coords = []  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        for i in range(len(self._H)):
            x = self._h_to_x(self._H[i])
            y = self._logk_to_y(self._log_K[i])
            if (x < self._MARGIN_LEFT or x > self._MARGIN_LEFT + self._PLOT_W or
                    y < self._MARGIN_TOP or y > self._MARGIN_TOP + self._PLOT_H):
                self._dot_coords.append(None)
                continue
            self._dot_coords.append((x, y))
        self._render_dot_pixmap()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

        # Redraw all finalized boundaries
        for boundary in self._boundaries:
            self._draw_finalized_boundary(boundary)

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _render_dot_pixmap(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        """Render all scatter dots onto a single QPixmap for performance."""
        if self._dot_pixmap_item and self._dot_pixmap_item.scene():
            self._scene.removeItem(self._dot_pixmap_item)
            self._dot_pixmap_item = None
        rect = self._scene.sceneRect()
        w, h = int(rect.width()), int(rect.height())
        if w <= 0 or h <= 0:
            return
        pix = QPixmap(w, h)
        pix.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        dot_r = 3.0
        for i, coord in enumerate(self._dot_coords):
            if coord is None:
                continue
            x, y = coord
            if self._region_labels is not None:
                idx = int(self._region_labels[i]) % len(self._REGION_COLORS)
                color = self._REGION_COLORS[idx]
            else:
                color = self._color_default
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(x, y), dot_r, dot_r)
        painter.end()
        self._dot_pixmap_item = self._scene.addPixmap(pix)
        self._dot_pixmap_item.setZValue(1)  # above background, below boundaries
        # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

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
        """Classify scatter points into regions using vectorized flood fill."""
        from scipy.ndimage import label as ndimage_label, distance_transform_edt

        if not self._boundaries:
            self._region_labels = None
            return

        grid_w = int(self._PLOT_W)
        grid_h = int(self._PLOT_H)
        wall = np.zeros((grid_h, grid_w), dtype=bool)

        # Rasterize boundary segments as walls (DDA + cross pattern)
        for boundary in self._boundaries:
            for j in range(len(boundary) - 1):
                x0 = boundary[j].x() - self._MARGIN_LEFT
                y0 = boundary[j].y() - self._MARGIN_TOP
                x1 = boundary[j + 1].x() - self._MARGIN_LEFT
                y1 = boundary[j + 1].y() - self._MARGIN_TOP
                dx, dy = x1 - x0, y1 - y0
                steps = max(int(abs(dx)), int(abs(dy)), 1)
                xs = np.round(np.linspace(x0, x1, steps + 1)).astype(int)
                ys = np.round(np.linspace(y0, y1, steps + 1)).astype(int)
                for di, dj in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                    rs = np.clip(ys + di, 0, grid_h - 1)
                    cs = np.clip(xs + dj, 0, grid_w - 1)
                    wall[rs, cs] = True

        # Flood fill via scipy (C-level, ~100x faster than Python DFS)
        grid, n_regions = ndimage_label(~wall)

        # Vectorized label assignment for all data points at once
        px_all = (self._H - self._h_min) / (self._h_max - self._h_min) * self._PLOT_W
        py_all = (1.0 - (self._log_K - self._logk_min) / (self._logk_max - self._logk_min)) * self._PLOT_H
        gx = np.clip(np.round(px_all).astype(int), 0, grid_w - 1)
        gy = np.clip(np.round(py_all).astype(int), 0, grid_h - 1)
        self._region_labels = grid[gy, gx]

        # Fix points on wall pixels: find nearest labeled region via distance transform
        on_wall = self._region_labels == 0
        if np.any(on_wall):
            # For each region, compute distance; assign wall points to nearest
            _, nearest_indices = distance_transform_edt(wall, return_distances=True, return_indices=True)
            nr, nc = nearest_indices[0], nearest_indices[1]
            wall_gy, wall_gx = gy[on_wall], gx[on_wall]
            self._region_labels[on_wall] = grid[nr[wall_gy, wall_gx], nc[wall_gy, wall_gx]]

        # Convert to 0-based
        self._region_labels = np.maximum(0, self._region_labels - 1)

        # Reorder region labels by ascending mean K
        unique_labels = np.unique(self._region_labels)
        if len(unique_labels) > 1:
            means = np.array([np.mean(self._K[self._region_labels == lbl]) for lbl in unique_labels])
            order = np.argsort(means)
            remap = np.zeros(int(unique_labels.max()) + 1, dtype=int)
            for new_lbl, idx in enumerate(order):
                remap[unique_labels[idx]] = new_lbl
            self._region_labels = remap[self._region_labels]
        # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

    def _update_dot_colors(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        self._render_dot_pixmap()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

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
# Update checker — queries GitHub API for latest release in background
# ---------------------------------------------------------------------------
class UpdateChecker(QThread):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
    """Check GitHub for a newer FreeTrace release. Emits update_available if found."""
    update_available = pyqtSignal(str, str, str)  # latest_version, release_body, release_url

    def run(self):
        try:
            import urllib.request
            import ssl
            import json as _json
            url = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.v3+json"})
            # Try with default SSL first, fall back to unverified if certificates are missing
            try:
                resp = urllib.request.urlopen(req, timeout=5)
            except (ssl.SSLError, urllib.error.URLError):
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                resp = urllib.request.urlopen(req, timeout=5, context=ctx)
            with resp:
                data = _json.loads(resp.read().decode())
            tag = data.get("tag_name", "").lstrip("v")
            if not tag:
                return
            # Compare version tuples
            def _ver_tuple(s):
                return tuple(int(x) for x in s.split("."))
            if _ver_tuple(tag) > _ver_tuple(_VERSION):
                body = data.get("body", "")
                html_url = data.get("html_url", f"https://github.com/{_GITHUB_REPO}/releases/latest")
                self.update_available.emit(tag, body, html_url)
        except Exception:
            pass  # silently ignore — network issues should not affect the GUI
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class FreeTraceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"FreeTrace v{_VERSION}") # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
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
        self._last_applied_scale = None  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        _generate_arrow_icons() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        self._update_banner = None  # placeholder for update notification widget
        self._setup_ui()
        self._apply_fonts()
        # Start background update check  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
        self._update_checker = UpdateChecker()
        self._update_checker.update_available.connect(self._show_update_banner)
        self._update_checker.start()

    # ------------------------------------------------------------------
    # Update notification banner
    # ------------------------------------------------------------------
    def _show_update_banner(self, latest_ver: str, body: str, url: str):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
        """Show a dismissible banner at the top of the window when an update is available."""
        if self._update_banner is not None:
            return  # already showing

        banner = QWidget()
        banner.setObjectName("updateBanner")
        banner.setStyleSheet(  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "#updateBanner { background-color: #1e1e1e; border-bottom: 2px solid #555; }"
            "#updateBanner QLabel { background-color: #1e1e1e; color: #ddd; font-size: 13px; }"
            "#updateBanner QPushButton { background-color: #1e1e1e; }"
        )
        layout = QVBoxLayout(banner)
        layout.setContentsMargins(12, 8, 12, 8)

        # Top row: message + dismiss button
        top_row = QHBoxLayout()
        title_label = QLabel(
            f"<b>FreeTrace v{latest_ver} is available</b>  (current: v{_VERSION})"
        )
        top_row.addWidget(title_label)
        top_row.addStretch()

        download_btn = QPushButton("Download")
        download_btn.setStyleSheet(
            "QPushButton { background-color: #2980b9 !important; color: white; border: none; "
            "padding: 4px 12px; border-radius: 3px; font-size: 12px; }"
            "QPushButton:hover { background-color: #3498db !important; }"
        )
        download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        download_btn.clicked.connect(lambda: __import__('webbrowser').open(url))
        top_row.addWidget(download_btn)

        dismiss_btn = QPushButton("X")
        dismiss_btn.setFixedSize(36, 36)
        dismiss_btn.setStyleSheet(
            "QPushButton { background: transparent !important; color: #e74c3c; "
            "border: 1px solid #e74c3c; border-radius: 4px; font-size: 20px; font-weight: bold; }"
            "QPushButton:hover { color: #ff6b6b; border-color: #ff6b6b; }"
        )
        dismiss_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        dismiss_btn.clicked.connect(lambda: self._dismiss_update_banner())
        top_row.addWidget(dismiss_btn)
        layout.addLayout(top_row)

        # Changelog section (collapsible)
        if body.strip():
            changelog_toggle = QPushButton("▶ What's new")
            changelog_toggle.setStyleSheet(
                "QPushButton { background: transparent !important; color: #7ec8e3; border: none; "
                "font-size: 12px; text-align: left; padding: 2px 0px; }"
                "QPushButton:hover { color: #aed6f1; }"
            )
            changelog_toggle.setCursor(Qt.CursorShape.PointingHandCursor)

            changelog_text = QTextEdit()
            changelog_text.setReadOnly(True)
            changelog_text.setMarkdown(body)
            changelog_text.setMaximumHeight(200)
            changelog_text.setStyleSheet(
                "QTextEdit { background-color: #2a2a2a; color: #ddd; border: 1px solid #555; "
                "border-radius: 3px; font-size: 11px; padding: 6px; }"
            )
            changelog_text.setVisible(False)

            def _toggle_changelog():
                vis = not changelog_text.isVisible()
                changelog_text.setVisible(vis)
                changelog_toggle.setText("▼ What's new" if vis else "▶ What's new")

            changelog_toggle.clicked.connect(_toggle_changelog)
            layout.addWidget(changelog_toggle)
            layout.addWidget(changelog_text)

        self._update_banner = banner
        # Insert at the very top of the central widget layout
        central_layout = self.centralWidget().layout()
        central_layout.insertWidget(0, banner)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20

    def _dismiss_update_banner(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
        if self._update_banner is not None:
            self._update_banner.setParent(None)
            self._update_banner.deleteLater()
            self._update_banner = None

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
        self._main_tabs.currentChanged.connect(self._on_main_tab_changed)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

    def _on_main_tab_changed(self, index):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        """Rescale help image when switching to Analysis tab with Help sub-tab active."""
        if self._main_tabs.tabText(index) == "Analysis":
            if self._analysis_tabs.tabText(self._analysis_tabs.currentIndex()) == "Help":
                QTimer.singleShot(0, self._rescale_help_image)

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
        self._analysis_tabs.addTab(self._build_help_tab(), "Help")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._analysis_tabs.addTab(self._build_class_tab(), "Class")
        self._analysis_tabs.addTab(self._build_basic_stats_tab(), "Basic Stats")
        self._analysis_tabs.addTab(self._build_adv_stats_tab(), "Adv Stats")
        self._analysis_tabs.currentChanged.connect(self._on_analysis_tab_changed)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        layout.addWidget(self._analysis_tabs)

        return widget

    # ---- Help sub-tab (variable explanation image) -------------------------  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    def _build_help_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Scroll area for the help content (vertical only)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e1e; }")

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self._help_image_label = QLabel()
        self._help_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._help_image_label.setWordWrap(True)
        self._help_pixmap = None

        # Try to load help image from icon/ directory
        help_img_path = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for name in ('help.png', 'help.jpg', 'help.svg'):
            candidate = os.path.join(base_dir, 'icon', name)
            if os.path.isfile(candidate):
                help_img_path = candidate
                break

        if help_img_path:
            pixmap = QPixmap(help_img_path)
            if not pixmap.isNull():  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                self._help_pixmap = pixmap
                self._help_image_label.setPixmap(pixmap)
            else:
                self._help_image_label.setText("Failed to load help image.")
                self._help_image_label.setStyleSheet("color:#ff8800; font-size:14px; padding:20px;")
        else:
            self._help_image_label.setText(
                "Place a help image (help.png, help.jpg, or help.svg) in the icon/ folder\n"
                "to display variable explanations here.")
            self._help_image_label.setStyleSheet("color:#888; font-size:14px; padding:20px;")
        self._help_scroll = scroll  # keep reference for resize

        container_layout.addWidget(self._help_image_label)

        # Summary text below image
        help_text = QLabel()
        help_text.setWordWrap(True)
        help_text.setTextFormat(Qt.TextFormat.RichText)
        help_text.setStyleSheet("color:#cccccc; font-size:13px; padding:12px 20px;")
        help_text.setText(  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            "<p style='font-size:13px; color:#aaaaaa;'>Suggestions for additional statistics or modifications are welcome — "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            "please contact <a style='color:#66ccff;' href='mailto:junwoo.park@sorbonne-universite.fr'>"
            "junwoo.park@sorbonne-universite.fr</a></p>"
            "<p style='font-size:14px;'><b>Paper:</b> "
            "https://doi.org/10.64898/2026.01.08.698486</p>"
            "<h3 style='color:#66ccff;'>Loading Data</h3>"
            "<p><b>Class tab</b> — Requires both <code>_diffusion.csv</code> and "
            "<code>_traces.csv</code>. Select either file; the other is loaded automatically. "
            "Multiple datasets can be loaded at once.</p>"
            "<p><b>Basic Stats tab</b> — Requires <code>_traces.csv</code>; "
            "<code>_diffusion.csv</code> is optional. If only traces are available, "
            "all trajectory-based plots (jump distance, duration, EA-SD, angles) work normally. "
            "H and K distributions require <code>_diffusion.csv</code>.</p>"
            "<p><b>Advanced Stats tab</b> — Requires only <code>_traces.csv</code> "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "(<code>_diffusion.csv</code> is not used).</p>"
            "<h3 style='color:#66ccff;'>Key Concepts</h3>"
            "<p><b>Consecutive frames only (Δt = 1)</b> — Jump distance, mean jump distance, "
            "1D displacement, 1D displacement ratio, and angle distributions use only steps "
            "between consecutive frames. Steps spanning frame gaps (Δt &gt; 1) are excluded "
            "because they are not directly comparable: a molecule diffusing for 2 frames "
            "covers a different distance than one diffusing for 1 frame.</p>"
            "<p><b>Frame gaps</b> — Trajectories are not split at frame gaps. "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "This preserves trajectory identity and avoids artificially inflating "
            "trajectory counts. Duration and MSD span the full observation including gaps.</p>"
            "<h3 style='color:#66ccff;'>Class Tab</h3>"
            "<p>The Class tab provides an interactive H-K scatter plot for gating trajectories "
            "by their diffusion properties. Load one or more FreeTrace output datasets "
            "(<code>_diffusion.csv</code> + <code>_traces.csv</code>) and visualise the "
            "Hurst exponent (H) vs. diffusion coefficient (K) for each trajectory. "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "Note that 2H = α (the anomalous diffusion exponent).</p>"
            "<p><b>Gating</b> — Draw boundaries on the H-K scatter plot to select subsets of "
            "trajectories. Gated trajectories can be exported or further analysed.</p>"
            "<p><b>Load Boundary</b> — Import a previously saved gating boundary.</p>"
            "<h3 style='color:#66ccff;'>Basic Stats Tab</h3>"
            "<p><b>H &amp; K distributions</b> — Per-trajectory Hurst exponent and diffusion "
            "coefficient (requires <code>_diffusion.csv</code>). "
            "K is computed in pixel &amp; frame scale, not converted to μm &amp; s.</p>"
            "<p><b>Jump Distance</b> — Per-step Euclidean displacement √(Δx² + Δy²), Δt = 1 only. "
            "Assumes isotropic motion.</p>"
            "<p><b>Mean Jump Distance</b> — Average jump distance per trajectory (one value per trajectory).</p>"
            "<p><b>Duration</b> — Total observation time in frames: the sum of frame differences "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "across the trajectory (last frame − first frame).</p>"
            "<p><b>EA-SD</b> — Ensemble-Averaged Squared Displacement: the average of "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "squared displacements over all trajectories at each time point. "
            "Here, SD stands for Squared Displacement, not standard deviation.</p>"
            "<p><b>Angle / Polar Angle</b> — Deflection angle (0°–180°) and signed turning "
            "angle (0°–360°) between consecutive step pairs, both Δt = 1. "
            "Uniform if isotropic &amp; Brownian.</p>"
            "<h3 style='color:#66ccff;'>Advanced Stats Tab</h3>"  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            "<p>The Advanced Stats tab provides computationally intensive analyses that "
            "complement the Basic Stats tab. It requires only <code>_traces.csv</code> "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "(no diffusion data needed).</p>"
            "<p><b>TA-EA-SD (Time-Averaged Ensemble-Averaged Squared Displacement)</b> — "
            "For each trajectory, the squared displacement at lag τ is averaged over all "
            "valid time windows of size τ (the <i>time-average</i>). These per-trajectory "
            "means are then averaged across all trajectories in the ensemble (the "
            "<i>ensemble-average</i>). This two-stage averaging is more robust than the "
            "EA-SD in Basic Stats, especially for short trajectories with frame gaps: "
            "EA-SD uses only the displacement from origin at each time point, while "
            "TA-EA-SD exploits all overlapping windows. The slope on a log-log plot "
            "gives the anomalous diffusion exponent: slope = 1 for Brownian motion, "
            "&lt; 1 for subdiffusion, &gt; 1 for superdiffusion. Only windows where the "
            "actual frame gap equals the lag τ are included (gaps are skipped, not "
            "interpolated). The shaded region shows ± 1 std across the ensemble. "
            "A log-log version is shown below: on a log-log scale, the slope directly "
            "gives the anomalous diffusion exponent (slope = 1 for Brownian, &lt; 1 for "
            "subdiffusion, &gt; 1 for superdiffusion). The log-log plot omits the std "
            "fill to avoid y-axis distortion.</p>"
            "<p><b>1D Displacement (Δx, Δy)</b> — Projection of each step onto the x and y "
            "axes separately, using only consecutive-frame steps (Δt = 1). For a "
            "homogeneous population of Brownian or fBm molecules, each projection is "
            "Gaussian with zero mean. A non-Gaussian shape (heavy tails, multiple peaks) "
            "indicates a heterogeneous population with mixed diffusion states. Δx and Δy "
            "are shown overlaid; for isotropic motion they should be identical. "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            "A Gaussian N(x; μ, σ) × α is fitted to each histogram using L1 minimisation "
            "(<code>scipy.optimize.minimize</code>, Nelder-Mead). "
            "The stats panel reports empirical location/scale, plus the fitted parameters:</p>"
            "<ul>"
            "<li><b>Location</b> — Fitted mean μ of the Gaussian. Should be ≈ 0 for "
            "unbiased diffusion (no drift).</li>"
            "<li><b>Scale</b> — Fitted standard deviation σ. Related to the diffusion "
            "coefficient: σ² = 2D·Δt for free Brownian motion.</li>"
            "<li><b>Amplitude</b> — Vertical scaling factor α that adjusts the PDF to "
            "the histogram density. For a perfect fit, α ≈ 1.</li>"
            "</ul>"
            "<p><b>1D Displacement Ratio — Cauchy Fit</b> — The ratio of consecutive "
            "1D displacements: Δx(t+1) / Δx(t) (and similarly for Δy). For fractional "
            "Brownian motion with Hurst exponent H, this ratio follows a Cauchy "
            "distribution whose location parameter depends on H:</p>"
            "<p style='margin-left:20px;'><code>location = 2<sup>2H−1</sup> − 1</code></p>"
            "<p>A single-component Cauchy is fitted to the histogram using L1 minimisation "
            "(<code>scipy.optimize.minimize</code>, Nelder-Mead). The fitted parameters are:</p>"
            "<ul>"
            "<li><b>Ĥ</b> — Estimated Hurst exponent from the Cauchy fit. "
            "H = 0.5 → Brownian, H &lt; 0.5 → anti-persistent (subdiffusive), "
            "H &gt; 0.5 → persistent (superdiffusive).</li>"
            "<li><b>Location parameter</b> — The peak (mode) of the fitted Cauchy, "
            "equal to 2<sup>2Ĥ−1</sup> − 1. Negative for H &lt; 0.5, zero at H = 0.5, "
            "positive for H &gt; 0.5.</li>"
            "<li><b>Amplitude factor</b> — A vertical scaling factor that adjusts the "
            "PDF amplitude to match the observed histogram density. It is a fitting "
            "nuisance parameter, not a physical quantity.</li>"
            "</ul>"
            "<p>Deviation of the observed histogram from the fitted Cauchy curve suggests "
            "the underlying motion is not purely fractional Brownian (e.g., confined "
            "diffusion, active transport, or a mixture of diffusion states). "
            "Ratios are clipped to [−10, 10] and data with fewer than 10 valid ratios "
            "per state are excluded from fitting.</p>"
            "<h3 style='color:#66ccff;'>Log-log TA-EA-SD vs Cauchy Fit — When to Use Which?</h3>"  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            "<p>Both the log-log TA-EA-SD plot and the Cauchy ratio fit estimate the "
            "anomalous diffusion exponent (and hence the Hurst exponent H), but they "
            "have different strengths:</p>"
            "<table style='border-collapse:collapse; margin:8px 0;'>"
            "<tr style='border-bottom:1px solid #555;'>"
            "<th style='padding:4px 12px; text-align:left;'></th>"
            "<th style='padding:4px 12px; text-align:left;'>Log-log TA-EA-SD</th>"
            "<th style='padding:4px 12px; text-align:left;'>Cauchy Ratio Fit</th></tr>"
            "<tr><td style='padding:4px 12px;'><b>Output</b></td>"
            "<td style='padding:4px 12px;'>Visual slope (anomalous exponent α)</td>"
            "<td style='padding:4px 12px;'>Single number Ĥ (with α = 2H)</td></tr>"
            "<tr><td style='padding:4px 12px;'><b>Best for</b></td>"
            "<td style='padding:4px 12px;'>Qualitative inspection — spotting regime "
            "changes across time scales (e.g., subdiffusive at short lags, normal at "
            "long lags)</td>"
            "<td style='padding:4px 12px;'>Quantitative H estimation — robust "
            "single-value estimate from the full displacement ratio distribution</td></tr>"
            "<tr><td style='padding:4px 12px;'><b>Limitations</b></td>"
            "<td style='padding:4px 12px;'>Large lags have few averaging windows → "
            "noisy right tail; MSD errors across lags are correlated, so standard "
            "line fitting can underestimate uncertainty; fitting range choice affects "
            "the result</td>"
            "<td style='padding:4px 12px;'>Assumes the motion is fBm (or similar); "
            "collapses all time-scale information into one number, so regime changes "
            "are invisible</td></tr>"
            "</table>"
            "<p><b>Recommendation:</b> use the log-log plot to visually inspect the "
            "diffusion regime across time scales, and the Cauchy fit for the "
            "quantitative Ĥ estimate. If they disagree, the motion may not be "
            "well-described by a single fBm model.</p>"
            "<h3 style='color:#66ccff;'>NN-estimated H vs Cauchy-fitted Ĥ</h3>"  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "<p>FreeTrace provides two independent estimates of the Hurst exponent:</p>"
            "<table style='border-collapse:collapse; margin:8px 0;'>"
            "<tr style='border-bottom:1px solid #555;'>"
            "<th style='padding:4px 12px; text-align:left;'></th>"
            "<th style='padding:4px 12px; text-align:left;'>NN-estimated H</th>"
            "<th style='padding:4px 12px; text-align:left;'>Cauchy-fitted Ĥ</th></tr>"
            "<tr><td style='padding:4px 12px;'><b>Source</b></td>"
            "<td style='padding:4px 12px;'>Neural network inference during tracking "
            "(stored in <code>_diffusion.csv</code>)</td>"
            "<td style='padding:4px 12px;'>Cauchy fit to displacement ratios "
            "(computed from <code>_traces.csv</code>)</td></tr>"
            "<tr><td style='padding:4px 12px;'><b>Granularity</b></td>"
            "<td style='padding:4px 12px;'>Per-trajectory — each trajectory gets its own H</td>"
            "<td style='padding:4px 12px;'>Per-population — one Ĥ per diffusion state</td></tr>"
            "<tr><td style='padding:4px 12px;'><b>Requires</b></td>"
            "<td style='padding:4px 12px;'><code>_diffusion.csv</code> (Basic Stats / Class tab)</td>"
            "<td style='padding:4px 12px;'><code>_traces.csv</code> only (Advanced Stats tab)</td></tr>"
            "<tr><td style='padding:4px 12px;'><b>Best for</b></td>"
            "<td style='padding:4px 12px;'>Per-trajectory H distribution, H-K scatter classification</td>"
            "<td style='padding:4px 12px;'>Independent validation of the ensemble diffusion regime</td></tr>"
            "</table>"
            "<p>Comparing the two: the NN-estimated H distribution (Basic Stats) "
            "shows the spread of H across individual trajectories, while the "
            "Cauchy-fitted Ĥ (Advanced Stats) gives an ensemble-level estimate. "
            "If the Cauchy Ĥ falls near the peak of the NN H distribution, "
            "both methods agree. A significant discrepancy may indicate that the "
            "NN model and the Cauchy assumption capture different aspects of the "
            "motion, or that the population is heterogeneous.</p>"
            "<h3 style='color:#66ccff;'>Common Normalisation</h3>"
            "<p>When enabled, all datasets share the same bin edges and the y-axis is "
            "normalised to the dataset with the most data points. The largest dataset "
            "sums to 100%, smaller datasets sum proportionally less. This preserves "
            "population size information when comparing datasets.</p>"
        )
        container_layout.addWidget(help_text)

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Deferred rescale after layout is complete  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        QTimer.singleShot(0, self._rescale_help_image)

        return widget

    def _rescale_help_image(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        """Rescale help image to fit scroll area viewport width."""
        if self._help_pixmap is not None and hasattr(self, '_help_scroll'):
            w = self._help_scroll.viewport().width() - 10
            if w > 50:
                self._help_image_label.setPixmap(
                    self._help_pixmap.scaledToWidth(w, Qt.TransformationMode.SmoothTransformation))

    def _on_analysis_tab_changed(self, index):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        """Rescale help image when the Help tab becomes visible."""
        if self._analysis_tabs.tabText(index) == "Help":
            QTimer.singleShot(0, self._rescale_help_image)

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
    def _build_basic_stats_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Toolbar
        toolbar = QHBoxLayout()
        self._stats_load_btn = QPushButton("Load Data")
        self._stats_load_btn.clicked.connect(self._on_stats_load_data)
        toolbar.addWidget(self._stats_load_btn)

        toolbar.addWidget(QLabel("Pixel size (μm):"))
        self._stats_pixelsize = QDoubleSpinBox()
        self._stats_pixelsize.setRange(0.001, 10.0)
        self._stats_pixelsize.setValue(1.0)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._stats_pixelsize.setDecimals(4)
        self._stats_pixelsize.setSingleStep(0.01)
        toolbar.addWidget(self._stats_pixelsize)

        toolbar.addWidget(QLabel("Frame rate (s):"))
        self._stats_framerate = QDoubleSpinBox()
        self._stats_framerate.setRange(0.0001, 10.0)
        self._stats_framerate.setValue(1.0)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._stats_framerate.setDecimals(4)
        self._stats_framerate.setSingleStep(0.001)
        toolbar.addWidget(self._stats_framerate)

        toolbar.addWidget(QLabel("Min traj length:"))
        self._stats_cutoff = QSpinBox()
        self._stats_cutoff.setRange(1, 9999)
        self._stats_cutoff.setValue(3)
        toolbar.addWidget(self._stats_cutoff)

        self._stats_common_norm = QCheckBox("Common normalisation")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._stats_common_norm.setChecked(False)
        toolbar.addWidget(self._stats_common_norm)

        self._stats_legend_cb = QCheckBox("Legend")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._stats_legend_cb.setChecked(True)
        self._stats_legend_cb.stateChanged.connect(self._on_stats_legend_toggled)
        toolbar.addWidget(self._stats_legend_cb)

        self._stats_run_btn = QPushButton("▶ Run Basic Stats")
        self._stats_run_btn.clicked.connect(self._on_run_preprocessing)
        toolbar.addWidget(self._stats_run_btn)

        self._stats_save_btn = QPushButton("Save Plots")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._stats_save_btn.clicked.connect(self._on_save_stats_plots)
        self._stats_save_btn.setEnabled(False)
        toolbar.addWidget(self._stats_save_btn)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

        self._stats_status_label = QLabel("")
        self._stats_status_label.setStyleSheet("color:#888;")
        toolbar.addWidget(self._stats_status_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Scroll area for plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:#1e1e1e; border:none;")
        scroll_widget = QWidget()
        self._stats_plot_layout = QVBoxLayout(scroll_widget)
        self._stats_plot_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._stats_plot_layout.setContentsMargins(4, 4, 4, 4)
        self._stats_plot_layout.setSpacing(4)

        self._stats_placeholder = QLabel("Load data and click 'Run Basic Stats' to generate plots.")
        self._stats_placeholder.setWordWrap(True)
        self._stats_placeholder.setStyleSheet("color:#888; font-size:14px; padding:20px;")
        self._stats_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stats_plot_layout.addWidget(self._stats_placeholder)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # State
        self._stats_datasets = []  # separate from Class tab's _loaded_datasets
        self._stats_worker = None
        self._stats_results = None
        self._stats_canvases = []  # keep refs to FigureCanvasQTAgg

        return widget

    def _on_stats_load_data(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        """Load data for Basic Stats — traces required, diffusion optional."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select FreeTrace output CSV(s)", "",
            "CSV files (*_traces.csv *_diffusion.csv);;All files (*)",
        )
        if not paths:
            return
        paths = sorted(paths)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._stats_datasets.clear()
        for p in paths:
            self._load_stats_data_from_file(p)
        self._stats_datasets.sort(key=lambda ds: ds['video_name'])
        n = len(self._stats_datasets)
        if n > 0:
            has_diff = sum(1 for ds in self._stats_datasets if ds['diffusion_df'] is not None)
            self._stats_status_label.setText(
                f"Loaded {n} video(s) ({has_diff} with diffusion data)."
            )

    def _load_stats_data_from_file(self, selected_path):
        """Load data for Basic Stats — accepts _traces.csv (required), _diffusion.csv (optional)."""
        try:
            if '_traces.csv' in selected_path:
                traces_path = selected_path
                diffusion_path = selected_path.replace('_traces.csv', '_diffusion.csv')
            elif '_diffusion.csv' in selected_path:
                diffusion_path = selected_path
                traces_path = selected_path.replace('_diffusion.csv', '_traces.csv')
            else:
                return

            # Skip duplicates
            for ds in self._stats_datasets:
                if ds['traces_path'] == traces_path:
                    return

            if not os.path.exists(traces_path):
                QMessageBox.warning(self, "File not found", f"Traces file not found:\n{traces_path}")
                return

            traces_df = pd.read_csv(traces_path)
            required_cols = {'traj_idx', 'frame', 'x', 'y'}
            if not required_cols.issubset(traces_df.columns):
                missing = required_cols - set(traces_df.columns)
                QMessageBox.warning(self, "Invalid traces file",
                                    f"Missing columns: {', '.join(sorted(missing))}")
                return

            # Diffusion file is optional
            diffusion_df = None
            if os.path.exists(diffusion_path):
                df = pd.read_csv(diffusion_path)
                if {'traj_idx', 'H', 'K'}.issubset(df.columns):
                    diffusion_df = df

            fname = os.path.basename(traces_path)
            video_name = fname.replace('_traces.csv', '')

            self._stats_datasets.append({
                'video_name': video_name,
                'traces_path': traces_path,
                'diffusion_path': diffusion_path,
                'traces_df': traces_df,
                'diffusion_df': diffusion_df,
            })
        except Exception as e:
            QMessageBox.critical(self, "Error loading data", str(e))

    def _on_run_preprocessing(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        if not self._stats_datasets:
            QMessageBox.warning(self, "No data", "Load data first.")
            return
        if self._stats_worker is not None and self._stats_worker.isRunning():
            return

        # Build per-dataset tuples: (name, merged_df, has_diffusion)
        dataset_tuples = []
        for ds in self._stats_datasets:
            tdf = ds['traces_df'].copy()
            has_diff = ds['diffusion_df'] is not None
            if has_diff:
                ddf = ds['diffusion_df'][['traj_idx', 'H', 'K']].copy()
                if 'state' in ds['diffusion_df'].columns:
                    ddf['state'] = ds['diffusion_df']['state']
                merged = tdf.merge(ddf, on='traj_idx', how='left')
            else:
                merged = tdf.copy()
            if 'z' not in merged.columns:
                merged['z'] = 0.0
            if 'state' not in merged.columns:
                merged['state'] = 0
            dataset_tuples.append((ds['video_name'], merged, has_diff))

        self._stats_run_btn.setEnabled(False)
        self._stats_status_label.setText("Preprocessing...")
        self._stats_worker = StatsWorker(
            dataset_tuples, self._stats_pixelsize.value(), self._stats_framerate.value(),
            self._stats_cutoff.value(),
        )
        self._stats_worker.progress.connect(
            lambda pct, msg: self._stats_status_label.setText(f"{msg} ({pct}%)")
        )
        self._stats_worker.finished.connect(self._stats_worker_finished)
        self._stats_worker.error.connect(self._stats_worker_error)
        self._stats_worker.start()

    def _stats_worker_error(self, msg):
        self._stats_run_btn.setEnabled(True)
        self._stats_status_label.setText(f"Error: {msg}")

    def _stats_worker_finished(self, results_list):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._stats_run_btn.setEnabled(True)
        self._stats_status_label.setText("Done.")
        self._stats_results = results_list
        self._stats_save_btn.setEnabled(True)
        self._render_stats_plots(results_list)

    def _render_stats_plots(self, results_list):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        """Render all Basic Stats plots — overlay multiple datasets with distinct colors."""
        # Clear previous plots
        for canvas in self._stats_canvases:
            canvas.setParent(None)
            canvas.deleteLater()
        self._stats_canvases.clear()
        while self._stats_plot_layout.count():
            item = self._stats_plot_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        n_datasets = len(results_list)
        single = (n_datasets == 1)
        common_norm = self._stats_common_norm.isChecked()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        show_legend = self._stats_legend_cb.isChecked()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        hist_ylabel = 'Percent'
        # Per-dataset colors (tab10 for multi-dataset, per-state colors for single)
        ds_palette = sns.color_palette('tab10', n_colors=max(n_datasets, 1))
        any_has_diff = any(r['has_diffusion'] for r in results_list)

        dark_style = {
            'figure.facecolor': '#1e1e1e',
            'axes.facecolor': '#1a1a1a',
            'text.color': '#cccccc',
            'axes.labelcolor': '#cccccc',
            'xtick.color': '#cccccc',
            'ytick.color': '#cccccc',
            'axes.edgecolor': '#555555',
            'grid.color': '#333333',
        }

        def _make_canvas(fig):
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(350)
            self._stats_canvases.append(canvas)
            return canvas

        def _ds_label(r, state=None):
            """Build legend label: dataset name (+ state if multi-state single dataset)."""
            name = r['name']
            if single and len(r['total_states']) > 1 and state is not None:
                return f'State {state}'
            if not single:
                if state is not None and len(r['total_states']) > 1:
                    return f'{name} (st {state})'
                return name
            return None  # single dataset, single state — no legend

        def _iter_colors(results_list):
            """Yield (dataset_idx, result, state, color) for plotting.

            Single dataset: one color per state (tab10 by state index).
            Multiple datasets: one color per dataset (states share that color, alpha varies).
            """
            if single:
                r = results_list[0]
                st_palette = sns.color_palette('tab10', n_colors=max(len(r['total_states']), 1))
                for si, st in enumerate(r['total_states']):
                    yield 0, r, st, st_palette[si % len(st_palette)]
            else:
                for di, r in enumerate(results_list):
                    for st in r['total_states']:
                        yield di, r, st, ds_palette[di % len(ds_palette)]

        need_legend = not single or (single and len(results_list[0]['total_states']) > 1)

        # Precompute shared bin edges across all datasets (always, for equal bin widths)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        common_bins = {}

        def _common_edges(key, df_getter, n_bins, log_scale=False):
            all_vals = pd.concat([df_getter(r) for r in results_list], ignore_index=True).dropna()
            if len(all_vals) > 0:
                vmin, vmax = all_vals.min(), all_vals.max()
                if log_scale and vmin > 0:
                    return np.geomspace(vmin, vmax, n_bins + 1)
                return np.linspace(vmin, vmax, n_bins + 1)
            return n_bins
        common_bins['H'] = _common_edges('H', lambda r: r['analysis_data1']['H'] if 'H' in r['analysis_data1'].columns else pd.Series(dtype=float), 50)
        common_bins['K'] = _common_edges('K', lambda r: r['analysis_data1']['K'][r['analysis_data1']['K'] > 0] if 'K' in r['analysis_data1'].columns else pd.Series(dtype=float), 50, log_scale=True)
        common_bins['mean_jump_d'] = _common_edges('mean_jump_d', lambda r: r['analysis_data1']['mean_jump_d'], 50)
        common_bins['2d_displacement'] = _common_edges('2d_displacement', lambda r: r['analysis_data2']['2d_displacement'], 50)
        common_bins['duration'] = _common_edges('duration', lambda r: r['analysis_data1']['duration'], 200)
        common_bins['angle'] = _common_edges('angle', lambda r: r['analysis_data3']['angle'], 50)

        def _bins(key, default):
            return common_bins[key] if key in common_bins else default  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

        # Precompute max row counts per data source for common normalisation  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        n_max = {}
        if common_norm:
            n_max['data1'] = max(len(r['analysis_data1']) for r in results_list)
            n_max['data2'] = max(len(r['analysis_data2']) for r in results_list)
            n_max['data3'] = max(len(r['analysis_data3']) for r in results_list)

        def _hist_kwargs(subset, data_key):
            """Return dict with stat/weights for histplot depending on common_norm."""
            if common_norm and data_key in n_max and n_max[data_key] > 0:
                w = np.ones(len(subset)) * (100.0 / n_max[data_key])
                return {'stat': 'count', 'weights': w}
            return {'stat': 'percent'}

        def _bins_safe(key, default):
            """Return bin edges as list (avoids seaborn bug with numpy array + weights)."""
            b = _bins(key, default)
            return list(b) if isinstance(b, np.ndarray) else b

        def _make_fig_with_stats():  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            """Create a figure with 80% plot area (left) and 20% stats panel (right).

            Uses constrained_layout instead of tight_layout (incompatible with GridSpec).
            Do NOT call fig.tight_layout() on figures returned by this function.
            """
            fig = Figure(figsize=(10, 4), dpi=100, constrained_layout=True)
            gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.02)
            ax = fig.add_subplot(gs[0])
            ax_stats = fig.add_subplot(gs[1])
            ax_stats.axis('off')
            return fig, ax, ax_stats

        def _fill_stats_panel(ax_stats, stat_lines):
            """Fill the stats panel with lines of (label, color, mean, std)."""
            y = 0.95
            ax_stats.text(0.05, y, 'Mean \u00b1 Std', color='#aaaaaa', fontsize=9,
                         fontweight='bold', transform=ax_stats.transAxes, va='top')
            y -= 0.08
            for label, color, mean_val, std_val in stat_lines:
                txt = f'{label}' if label else ''
                if txt:
                    ax_stats.text(0.05, y, txt, color=color, fontsize=8,
                                 transform=ax_stats.transAxes, va='top', fontweight='bold')
                    y -= 0.06
                ax_stats.text(0.05, y, f'{mean_val:.4g} \u00b1 {std_val:.4g}', color=color, fontsize=8,
                             transform=ax_stats.transAxes, va='top')
                y -= 0.07  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

        # --- Summary table ---
        summary_section = CollapsibleSection("Summary")
        html = "<table style='color:#ccc; font-size:13px; border-collapse:collapse;'>"
        html += "<tr style='border-bottom:1px solid #555;'>"
        html += "<th style='padding:4px 12px;'>Dataset</th>"
        html += "<th style='padding:4px 12px;'>State</th><th style='padding:4px 12px;'>Count</th>"
        html += "<th style='padding:4px 12px;'>Mean Jump Dist</th>"
        html += "<th style='padding:4px 12px;'>Mean Duration</th>"
        if any_has_diff:
            html += "<th style='padding:4px 12px;'>Mean H</th><th style='padding:4px 12px;'>Median H</th>"
            html += "<th style='padding:4px 12px;'>Mean K</th><th style='padding:4px 12px;'>Median K</th>"
        html += "</tr>"
        for r in results_list:
            ad1 = r['analysis_data1']
            for st in r['total_states']:
                subset = ad1[ad1['state'] == st]
                html += f"<tr><td style='padding:4px 12px;'>{r['name']}</td>"
                html += f"<td style='padding:4px 12px;'>{st}</td>"
                html += f"<td style='padding:4px 12px;'>{len(subset)}</td>"
                html += f"<td style='padding:4px 12px;'>{subset['mean_jump_d'].mean():.4f}</td>"
                html += f"<td style='padding:4px 12px;'>{subset['duration'].mean():.4f}</td>"
                if any_has_diff:
                    if r['has_diffusion'] and 'H' in ad1.columns:
                        html += f"<td style='padding:4px 12px;'>{subset['H'].mean():.4f}</td>"
                        html += f"<td style='padding:4px 12px;'>{subset['H'].median():.4f}</td>"
                        html += f"<td style='padding:4px 12px;'>{subset['K'].mean():.4f}</td>"
                        html += f"<td style='padding:4px 12px;'>{subset['K'].median():.4f}</td>"
                    else:
                        html += "<td style='padding:4px 12px;'>—</td>" * 4
                html += "</tr>"
        html += "</table>"
        summary_label = QLabel(html)
        summary_label.setTextFormat(Qt.TextFormat.RichText)
        summary_section.add_widget(summary_label)
        self._stats_plot_layout.addWidget(summary_section)

        # --- Population Pie Chart ---  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        pie_section = CollapsibleSection("Trajectory Population")
        with plt.style.context(dark_style):
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            labels = []
            sizes = []
            colors = []
            if single and len(results_list[0]['total_states']) > 1:
                r = results_list[0]
                st_palette = sns.color_palette('tab10', n_colors=len(r['total_states']))
                for si, st in enumerate(r['total_states']):
                    count = len(r['analysis_data1'][r['analysis_data1']['state'] == st])
                    labels.append(f'State {st}')
                    sizes.append(count)
                    colors.append(st_palette[si % len(st_palette)])
            else:
                for di, r in enumerate(results_list):
                    count = len(r['analysis_data1'])
                    labels.append(r['name'])
                    sizes.append(count)
                    colors.append(ds_palette[di % len(ds_palette)])
            total = sum(sizes)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            ax.pie(sizes, labels=labels, colors=colors,
                   autopct=lambda pct: f'{pct:.1f}%\n({int(round(pct * total / 100))})',
                   textprops={'color': '#cccccc', 'fontsize': 10},
                   wedgeprops={'edgecolor': '#333333', 'linewidth': 0.5})
            ax.set_title('Trajectory Population')
            fig.tight_layout()
        pie_section.add_widget(_make_canvas(fig))
        self._stats_plot_layout.addWidget(pie_section)

        # --- H Distribution ---  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        h_section = CollapsibleSection("H Distribution")
        if any_has_diff:
            with plt.style.context(dark_style):
                fig, ax, ax_stats = _make_fig_with_stats()
                plotted = False
                stat_lines = []
                for di, r, st, color in _iter_colors(results_list):
                    if not r['has_diffusion'] or 'H' not in r['analysis_data1'].columns:
                        continue
                    subset = r['analysis_data1'][r['analysis_data1']['state'] == st]
                    if len(subset) > 0:
                        sns.histplot(data=subset, x='H', bins=_bins_safe('H', 50),
                                     kde=True, ax=ax, color=color,
                                     label=_ds_label(r, st), alpha=0.5, **_hist_kwargs(subset, 'data1'))
                        stat_lines.append((_ds_label(r, st) or '', color, subset['H'].mean(), subset['H'].std()))
                        plotted = True
                ax.set_xlabel('H')
                ax.set_ylabel(hist_ylabel)
                ax.set_title('H Distribution')
                if need_legend and plotted:
                    ax.legend(loc='upper right')
                if stat_lines:
                    _fill_stats_panel(ax_stats, stat_lines)
            h_section.add_widget(_make_canvas(fig))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        else:
            warn = QLabel("_diffusion.csv required for this plot.")
            warn.setStyleSheet("color:#ff8800; padding:12px;")
            h_section.add_widget(warn)
        self._stats_plot_layout.addWidget(h_section)

        # --- K Distribution ---  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        k_section = CollapsibleSection("K Distribution (pixel && frame scale)")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        if any_has_diff:
            # Always use log-spaced bins for K (linear bins look uneven on log axis)
            if not common_norm:
                all_k = pd.concat([r['analysis_data1']['K'][r['analysis_data1']['K'] > 0]
                                   for r in results_list if r['has_diffusion'] and 'K' in r['analysis_data1'].columns],
                                  ignore_index=True).dropna()
                k_default_bins = list(np.geomspace(all_k.min(), all_k.max(), 51)) if len(all_k) > 0 else 50
            else:
                k_default_bins = _bins_safe('K', 50)
            from scipy.stats import gaussian_kde
            log_k_edges = np.log10(np.array(k_default_bins)) if isinstance(k_default_bins, list) else None
            with plt.style.context(dark_style):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                fig, ax, ax_stats = _make_fig_with_stats()
                plotted = False
                stat_lines = []
                for di, r, st, color in _iter_colors(results_list):
                    if not r['has_diffusion'] or 'K' not in r['analysis_data1'].columns:
                        continue
                    subset = r['analysis_data1'][r['analysis_data1']['state'] == st]
                    k_vals = subset['K'][subset['K'] > 0] if len(subset) > 0 else pd.Series(dtype=float)
                    if len(k_vals) > 0:
                        sns.histplot(x=k_vals, bins=k_default_bins, kde=False,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                                     ax=ax, color=color, label=_ds_label(r, st),
                                     alpha=0.5, **_hist_kwargs(k_vals, 'data1'))
                        for patch in ax.patches:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                            patch.set_edgecolor('black')
                            patch.set_linewidth(0.5)
                        # KDE in log space (seaborn's kde computes in linear space, distorts on log axis)
                        log_k = np.log10(k_vals.values)
                        kde = gaussian_kde(log_k)
                        x_log = np.linspace(log_k.min(), log_k.max(), 200)
                        kde_density = kde(x_log)
                        bin_w = np.mean(np.diff(log_k_edges)) if log_k_edges is not None else np.mean(np.diff(x_log))
                        hkw = _hist_kwargs(k_vals, 'data1')
                        if hkw.get('stat') == 'count' and 'weights' in hkw:
                            kde_scaled = kde_density * bin_w * hkw['weights'][0] * len(k_vals)
                        else:
                            kde_scaled = kde_density * bin_w * 100
                        ax.plot(10**x_log, kde_scaled, color=color, linewidth=1.5)
                        stat_lines.append((_ds_label(r, st) or '', color, k_vals.mean(), k_vals.std()))
                        plotted = True
                ax.set_xscale('log')
                ax.set_xlabel('K')
                ax.set_ylabel(hist_ylabel)
                ax.set_title('K Distribution')
                if need_legend and plotted:
                    ax.legend(loc='upper right')
                if stat_lines:
                    _fill_stats_panel(ax_stats, stat_lines)
            k_section.add_widget(_make_canvas(fig))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        else:
            warn = QLabel("_diffusion.csv required for this plot.")
            warn.setStyleSheet("color:#ff8800; padding:12px;")
            k_section.add_widget(warn)
        self._stats_plot_layout.addWidget(k_section)

        # --- Mean Jump Distance ---  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        mjd_section = CollapsibleSection("Mean Jump Distance (consecutive frames only, Δt = 1)")
        with plt.style.context(dark_style):
            fig, ax, ax_stats = _make_fig_with_stats()
            plotted = False
            stat_lines = []
            for di, r, st, color in _iter_colors(results_list):
                subset = r['analysis_data1'][r['analysis_data1']['state'] == st]
                if len(subset) > 0:
                    sns.histplot(data=subset, x='mean_jump_d', bins=_bins_safe('mean_jump_d', 50),
                                 kde=True, ax=ax, color=color,
                                 label=_ds_label(r, st), alpha=0.5, **_hist_kwargs(subset, 'data1'))
                    stat_lines.append((_ds_label(r, st) or '', color, subset['mean_jump_d'].mean(), subset['mean_jump_d'].std()))
                    plotted = True
            ax.set_xlabel('Mean Jump Distance (μm)')
            ax.set_ylabel(hist_ylabel)
            ax.set_title('Mean Jump Distance')
            if need_legend and plotted:
                ax.legend(loc='upper right')
            if stat_lines:
                _fill_stats_panel(ax_stats, stat_lines)
        mjd_section.add_widget(_make_canvas(fig))
        self._stats_plot_layout.addWidget(mjd_section)

        # --- Jump Distance ---  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        disp_section = CollapsibleSection("Jump Distance (consecutive frames only, Δt = 1)")
        with plt.style.context(dark_style):
            fig, ax, ax_stats = _make_fig_with_stats()
            plotted = False
            stat_lines = []
            for di, r, st, color in _iter_colors(results_list):
                subset = r['analysis_data2'][r['analysis_data2']['state'] == st]
                if len(subset) > 0:
                    sns.histplot(data=subset, x='2d_displacement', bins=_bins_safe('2d_displacement', 50),
                                 kde=True, ax=ax, color=color,
                                 label=_ds_label(r, st), alpha=0.5, **_hist_kwargs(subset, 'data2'))
                    stat_lines.append((_ds_label(r, st) or '', color, subset['2d_displacement'].mean(), subset['2d_displacement'].std()))
                    plotted = True
            ax.set_xlabel('Jump Distance (μm)')
            ax.set_ylabel(hist_ylabel)
            ax.set_title('Jump Distance')
            if need_legend and plotted:
                ax.legend(loc='upper right')
            if stat_lines:
                _fill_stats_panel(ax_stats, stat_lines)
        disp_section.add_widget(_make_canvas(fig))
        self._stats_plot_layout.addWidget(disp_section)

        # --- Duration ---  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        dur_section = CollapsibleSection("Duration")
        with plt.style.context(dark_style):
            fig, ax, ax_stats = _make_fig_with_stats()
            plotted = False
            stat_lines = []
            all_durations = []
            for di, r, st, color in _iter_colors(results_list):
                subset = r['analysis_data1'][r['analysis_data1']['state'] == st]
                if len(subset) > 0:
                    sns.histplot(data=subset, x='duration', bins=_bins_safe('duration', 200),
                                 kde=True, ax=ax, color=color,
                                 label=_ds_label(r, st), alpha=0.5, **_hist_kwargs(subset, 'data1'))
                    all_durations.extend(subset['duration'].tolist())
                    stat_lines.append((_ds_label(r, st) or '', color, subset['duration'].mean(), subset['duration'].std()))
                    plotted = True
            if all_durations:
                ax.set_xlim(0, np.percentile(all_durations, 99))
            ax.set_xlabel('Duration (s)')
            ax.set_ylabel(hist_ylabel)
            ax.set_title('Duration')
            if need_legend and plotted:
                ax.legend(loc='upper right')
            if stat_lines:
                _fill_stats_panel(ax_stats, stat_lines)
        dur_section.add_widget(_make_canvas(fig))
        self._stats_plot_layout.addWidget(dur_section)

        # --- MSD ---
        msd_section = CollapsibleSection("Ensemble-Averaged Squared Displacement")
        with plt.style.context(dark_style):
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            plotted = False
            for di, r, st, color in _iter_colors(results_list):
                msd_df = r['msd']
                msd_clean = msd_df.dropna(subset=['mean'])
                msd_clean = msd_clean[msd_clean['nb_data'] >= 3]
                subset = msd_clean[msd_clean['state'] == st]
                if len(subset) > 0:
                    ax.plot(subset['time'], subset['mean'], color=color,
                            label=_ds_label(r, st))
                    ax.fill_between(subset['time'],
                                    subset['mean'] - subset['std'],
                                    subset['mean'] + subset['std'],
                                    alpha=0.15, color=color)
                    plotted = True
            ax.set_xlabel('Time lag (s)')
            ax.set_ylabel('EA-SD (μm²)')
            ax.set_title('Ensemble-Averaged Squared Displacement')
            if need_legend and plotted:
                ax.legend(loc='upper right')
            fig.tight_layout()
        msd_section.add_widget(_make_canvas(fig))
        self._stats_plot_layout.addWidget(msd_section)

        # --- Angle Distribution ---  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        angle_section = CollapsibleSection("Angle Distribution (consecutive frames only, Δt = 1)")
        with plt.style.context(dark_style):
            fig, ax, ax_stats = _make_fig_with_stats()
            plotted = False
            stat_lines = []
            for di, r, st, color in _iter_colors(results_list):
                subset = r['analysis_data3'][r['analysis_data3']['state'] == st]
                if len(subset) > 0:
                    ang_kwargs = _hist_kwargs(subset, 'data3')
                    if not common_norm:
                        ang_kwargs = {'stat': 'proportion'}
                    sns.histplot(data=subset, x='angle', bins=_bins_safe('angle', 50),
                                 kde=True, ax=ax, color=color,
                                 label=_ds_label(r, st), alpha=0.5, **ang_kwargs)
                    stat_lines.append((_ds_label(r, st) or '', color, subset['angle'].mean(), subset['angle'].std()))
                    plotted = True
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel(hist_ylabel if common_norm else 'Proportion')
            ax.set_title('Angle Distribution')
            if need_legend and plotted:
                ax.legend(loc='upper right')
            if stat_lines:
                _fill_stats_panel(ax_stats, stat_lines)
        angle_section.add_widget(_make_canvas(fig))
        self._stats_plot_layout.addWidget(angle_section)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

        # --- Polar Angle Distribution ---  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        polar_section = CollapsibleSection("Polar Angle Distribution, 0°–360° (consecutive frames only, Δt = 1)")
        with plt.style.context(dark_style):
            fig = Figure(figsize=(6, 6), dpi=100)
            ax = fig.add_subplot(111, projection='polar')
            ax.set_facecolor('#1a1a1a')
            n_bins_polar = 36  # 10° per bin
            bin_edges = np.linspace(0, 2 * np.pi, n_bins_polar + 1)
            n_max_polar = n_max.get('data3', 0) if common_norm else 0  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            for di, r, st, color in _iter_colors(results_list):
                subset = r['analysis_data3'][r['analysis_data3']['state'] == st]
                if len(subset) > 0 and 'polar_angle' in subset.columns:
                    theta_rad = np.radians(subset['polar_angle'].values)
                    counts, _ = np.histogram(theta_rad, bins=bin_edges)
                    denom = n_max_polar if common_norm and n_max_polar > 0 else counts.sum()
                    proportions = counts / denom if denom > 0 else counts
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    width = 2 * np.pi / n_bins_polar
                    ax.bar(bin_centers, proportions, width=width, color=color,
                           alpha=0.5, label=_ds_label(r, st), edgecolor='#555555', linewidth=0.3)
            ax.set_theta_zero_location('E')  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            ax.set_theta_direction(1)  # counterclockwise
            ax.set_yticklabels([])  # hide radial tick labels
            ax.set_title('Polar Angle Distribution', pad=15)
            ax.tick_params(colors='#cccccc')
            ax.spines['polar'].set_color('#555555')
            if need_legend:
                ax.legend(loc='upper left', bbox_to_anchor=(-0.35, 1.15))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            fig.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.05)
        polar_section.add_widget(_make_canvas(fig))
        self._stats_plot_layout.addWidget(polar_section)

        self._stats_plot_layout.addStretch()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

        # Apply legend visibility from checkbox  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        if not show_legend:
            for canvas in self._stats_canvases:
                for ax in canvas.figure.get_axes():
                    leg = ax.get_legend()
                    if leg:
                        leg.set_visible(False)
                canvas.draw_idle()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

    def _on_save_stats_plots(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        """Save all rendered plot canvases as PNG files to a user-selected directory."""
        if not self._stats_canvases:
            QMessageBox.warning(self, "No plots", "Run preprocessing first.")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Select directory to save plots")
        if not save_dir:
            return
        plot_names = ['population_pie']  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        if self._stats_results and any(r.get('has_diffusion', False) for r in self._stats_results):
            plot_names += ['H_distribution', 'K_distribution']
        plot_names += ['mean_jump_distance', 'jump_distance', 'duration',
                       'ensemble_averaged_SD', 'angle_distribution', 'polar_angle_distribution']
        saved = 0
        for i, canvas in enumerate(self._stats_canvases):
            name = plot_names[i] if i < len(plot_names) else f'plot_{i}'
            path = os.path.join(save_dir, f'{name}.png')
            canvas.figure.savefig(path, dpi=150, bbox_inches='tight',
                                  transparent=True)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            saved += 1
        self._stats_status_label.setText(f"Saved {saved} plots to {save_dir}")

    def _on_stats_legend_toggled(self, state):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        show = bool(state)
        for canvas in self._stats_canvases:
            for ax in canvas.figure.get_axes():
                leg = ax.get_legend()
                if leg:
                    leg.set_visible(show)
            canvas.draw_idle()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

    # ---- Adv Stats sub-tab ------------------------------------------------
    def _build_adv_stats_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Toolbar
        toolbar = QHBoxLayout()
        self._adv_stats_load_btn = QPushButton("Load Data")
        self._adv_stats_load_btn.clicked.connect(self._on_adv_stats_load_data)
        toolbar.addWidget(self._adv_stats_load_btn)

        toolbar.addWidget(QLabel("Pixel size (μm):"))
        self._adv_stats_pixelsize = QDoubleSpinBox()
        self._adv_stats_pixelsize.setRange(0.001, 10.0)
        self._adv_stats_pixelsize.setValue(1.0)
        self._adv_stats_pixelsize.setDecimals(4)
        self._adv_stats_pixelsize.setSingleStep(0.01)
        toolbar.addWidget(self._adv_stats_pixelsize)

        toolbar.addWidget(QLabel("Frame rate (s):"))
        self._adv_stats_framerate = QDoubleSpinBox()
        self._adv_stats_framerate.setRange(0.0001, 10.0)
        self._adv_stats_framerate.setValue(1.0)
        self._adv_stats_framerate.setDecimals(4)
        self._adv_stats_framerate.setSingleStep(0.001)
        toolbar.addWidget(self._adv_stats_framerate)

        toolbar.addWidget(QLabel("Min traj length:"))
        self._adv_stats_cutoff = QSpinBox()
        self._adv_stats_cutoff.setRange(1, 9999)
        self._adv_stats_cutoff.setValue(3)
        toolbar.addWidget(self._adv_stats_cutoff)

        self._adv_stats_legend_cb = QCheckBox("Legend")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._adv_stats_legend_cb.setChecked(True)
        self._adv_stats_legend_cb.stateChanged.connect(self._on_adv_stats_legend_toggled)
        toolbar.addWidget(self._adv_stats_legend_cb)

        self._adv_stats_run_btn = QPushButton("▶ Run Advanced Stats")
        self._adv_stats_run_btn.clicked.connect(self._on_run_adv_stats)
        toolbar.addWidget(self._adv_stats_run_btn)

        self._adv_stats_save_btn = QPushButton("Save Plots")
        self._adv_stats_save_btn.clicked.connect(self._on_save_adv_stats_plots)
        self._adv_stats_save_btn.setEnabled(False)
        toolbar.addWidget(self._adv_stats_save_btn)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

        self._adv_stats_status_label = QLabel("")
        self._adv_stats_status_label.setStyleSheet("color:#888;")
        toolbar.addWidget(self._adv_stats_status_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Scroll area for plots
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:#1e1e1e; border:none;")
        scroll_widget = QWidget()
        self._adv_stats_plot_layout = QVBoxLayout(scroll_widget)
        self._adv_stats_plot_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._adv_stats_plot_layout.setContentsMargins(4, 4, 4, 4)
        self._adv_stats_plot_layout.setSpacing(4)

        self._adv_stats_placeholder = QLabel("Load data and click 'Run Advanced Stats' to generate plots.")
        self._adv_stats_placeholder.setWordWrap(True)
        self._adv_stats_placeholder.setStyleSheet("color:#888; font-size:14px; padding:20px;")
        self._adv_stats_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._adv_stats_plot_layout.addWidget(self._adv_stats_placeholder)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # State
        self._adv_stats_datasets = []
        self._adv_stats_worker = None
        self._adv_stats_results = None
        self._adv_stats_canvases = []

        return widget

    def _on_adv_stats_load_data(self):
        """Load data for Advanced Stats — only _traces.csv needed."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select FreeTrace traces CSV(s)", "",
            "CSV files (*_traces.csv);;All files (*)",
        )
        if not paths:
            return
        paths = sorted(paths)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._adv_stats_datasets.clear()
        for p in paths:
            self._load_adv_stats_data_from_file(p)
        self._adv_stats_datasets.sort(key=lambda ds: ds['video_name'])
        n = len(self._adv_stats_datasets)
        if n > 0:
            self._adv_stats_status_label.setText(f"Loaded {n} video(s).")

    def _load_adv_stats_data_from_file(self, selected_path):
        """Load traces CSV for Advanced Stats."""
        try:
            if '_traces.csv' not in selected_path:
                return
            traces_path = selected_path

            # Skip duplicates
            for ds in self._adv_stats_datasets:
                if ds['traces_path'] == traces_path:
                    return

            if not os.path.exists(traces_path):
                QMessageBox.warning(self, "File not found", f"Traces file not found:\n{traces_path}")
                return

            traces_df = pd.read_csv(traces_path)
            required_cols = {'traj_idx', 'frame', 'x', 'y'}
            if not required_cols.issubset(traces_df.columns):
                missing = required_cols - set(traces_df.columns)
                QMessageBox.warning(self, "Invalid traces file",
                                    f"Missing columns: {', '.join(sorted(missing))}")
                return

            if 'state' not in traces_df.columns:
                traces_df['state'] = 0

            fname = os.path.basename(traces_path)
            video_name = fname.replace('_traces.csv', '')

            self._adv_stats_datasets.append({
                'video_name': video_name,
                'traces_path': traces_path,
                'traces_df': traces_df,
            })
        except Exception as e:
            QMessageBox.critical(self, "Error loading data", str(e))

    def _on_run_adv_stats(self):
        """Run Advanced Stats preprocessing in background thread."""
        if not self._adv_stats_datasets:
            QMessageBox.warning(self, "No data", "Load data first.")
            return
        if self._adv_stats_worker is not None and self._adv_stats_worker.isRunning():
            return

        dataset_tuples = []
        for ds in self._adv_stats_datasets:
            tdf = ds['traces_df'].copy()
            if 'state' not in tdf.columns:
                tdf['state'] = 0
            dataset_tuples.append((ds['video_name'], tdf))

        self._adv_stats_run_btn.setEnabled(False)
        self._adv_stats_status_label.setText("Processing...")
        self._adv_stats_worker = AdvStatsWorker(
            dataset_tuples, self._adv_stats_pixelsize.value(),
            self._adv_stats_framerate.value(), self._adv_stats_cutoff.value(),
        )
        self._adv_stats_worker.progress.connect(
            lambda pct, msg: self._adv_stats_status_label.setText(f"{msg} ({pct}%)")
        )
        self._adv_stats_worker.finished.connect(self._adv_stats_worker_finished)
        self._adv_stats_worker.error.connect(self._adv_stats_worker_error)
        self._adv_stats_worker.start()

    def _adv_stats_worker_error(self, msg):
        self._adv_stats_run_btn.setEnabled(True)
        self._adv_stats_status_label.setText(f"Error: {msg}")

    def _adv_stats_worker_finished(self, results_list):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._adv_stats_run_btn.setEnabled(True)
        parts = []
        for r in results_list:
            parts.append(f"{r['name']}: {r['n_trajectories']} trajectories")
        self._adv_stats_status_label.setText("Done. " + ", ".join(parts))
        self._adv_stats_results = results_list
        self._adv_stats_save_btn.setEnabled(True)
        self._render_adv_stats_plots(results_list)

    def _render_adv_stats_plots(self, results_list):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        """Render Advanced Stats plots — TA-EA-SD, 1D Displacement, 1D Ratio + Cauchy fit."""
        # Clear previous plots
        for canvas in self._adv_stats_canvases:
            canvas.setParent(None)
            canvas.deleteLater()
        self._adv_stats_canvases.clear()
        while self._adv_stats_plot_layout.count():
            item = self._adv_stats_plot_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        n_datasets = len(results_list)
        show_legend = self._adv_stats_legend_cb.isChecked()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        ds_palette = sns.color_palette('tab10', n_colors=max(n_datasets, 1))

        dark_style = {
            'figure.facecolor': '#1e1e1e',
            'axes.facecolor': '#1a1a1a',
            'text.color': '#cccccc',
            'axes.labelcolor': '#cccccc',
            'xtick.color': '#cccccc',
            'ytick.color': '#cccccc',
            'axes.edgecolor': '#555555',
            'grid.color': '#333333',
        }

        def _make_canvas(fig):
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(350)
            self._adv_stats_canvases.append(canvas)
            return canvas

        def _make_fig_with_stats():
            fig = Figure(figsize=(10, 4), dpi=100, constrained_layout=True)
            gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.02)
            ax = fig.add_subplot(gs[0])
            ax_stats = fig.add_subplot(gs[1])
            ax_stats.axis('off')
            return fig, ax, ax_stats

        def _fill_stats_panel(ax_stats, stat_lines):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            y = 0.95
            ax_stats.text(0.05, y, 'Statistics', color='#aaaaaa', fontsize=9,
                         fontweight='bold', transform=ax_stats.transAxes, va='top')
            y -= 0.08
            for label, color, loc_val, scale_val in stat_lines:
                txt = f'{label}' if label else ''
                if loc_val is not None:
                    txt += f'  loc={loc_val:.4g}'
                if scale_val is not None:
                    txt += f'  scale={scale_val:.4g}'
                ax_stats.text(0.05, y, txt, color=color, fontsize=8,
                             transform=ax_stats.transAxes, va='top')
                y -= 0.06

        def _ds_label(r, state=None):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            name = r['name']
            if state is not None and len(r['total_states']) > 1:
                return f"{name} S{state}" if n_datasets > 1 else f"State {state}"
            return name

        def _iter_colors(results_list):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            """Match Basic Stats color logic: single dataset → per-state tab10,
            multiple datasets → per-dataset tab10."""
            single = (n_datasets == 1)
            if single:
                r = results_list[0]
                state_pal = sns.color_palette('tab10', n_colors=max(len(r['total_states']), 1))
                for si, st in enumerate(r['total_states']):
                    yield r, st, state_pal[si % len(state_pal)]
            else:
                for ds_idx, r in enumerate(results_list):
                    for st in r['total_states']:
                        yield r, st, ds_palette[ds_idx % len(ds_palette)]

        with plt.style.context(dark_style):
            # ---- 1. TA-EA-SD (Time-Averaged Ensemble-Averaged Squared Displacement) ----
            section1 = CollapsibleSection("TA-EA-SD (Time-Averaged Ensemble-Averaged Squared Displacement)")
            fig = Figure(figsize=(10, 4), dpi=100, constrained_layout=True)
            ax = fig.add_subplot(111)

            for r, st, color in _iter_colors(results_list):
                tamsd = r['tamsd']
                subset = tamsd[tamsd['state'] == st]
                if subset.empty:
                    continue
                label = _ds_label(r, st)
                ax.plot(subset['time'], subset['mean'], color=color, label=label, linewidth=1.5)
                mean_arr = subset['mean'].to_numpy()
                std_arr = subset['std'].to_numpy()
                time_arr = subset['time'].to_numpy()
                ax.fill_between(time_arr, mean_arr - std_arr, mean_arr + std_arr,
                               color=color, alpha=0.2)

            ax.set_xlabel('Time lag (s)')
            ax.set_ylabel('TA-EA-SD (μm²)')
            ax.set_title('TA-EA-SD')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            section1.add_widget(_make_canvas(fig))
            self._adv_stats_plot_layout.addWidget(section1)

            # ---- 1b. TA-EA-SD log-log ----
            section1b = CollapsibleSection("TA-EA-SD — log-log")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            fig_log = Figure(figsize=(10, 4), dpi=100, constrained_layout=True)
            ax_log = fig_log.add_subplot(111)

            for r, st, color in _iter_colors(results_list):
                tamsd = r['tamsd']
                subset = tamsd[tamsd['state'] == st]
                if subset.empty:
                    continue
                label = _ds_label(r, st)
                valid = (subset['mean'] > 0) & (subset['time'] > 0)
                s = subset[valid]
                if s.empty:
                    continue
                ax_log.plot(s['time'], s['mean'], color=color, label=label, linewidth=1.5)

            ax_log.set_xscale('log')
            ax_log.set_yscale('log')
            ax_log.set_xlabel('Time lag (s)')
            ax_log.set_ylabel('TA-EA-SD (μm²)')
            ax_log.set_title('TA-EA-SD — log-log')
            ax_log.legend(fontsize=7, loc='best')
            ax_log.grid(True, alpha=0.3, which='both')
            section1b.add_widget(_make_canvas(fig_log))
            self._adv_stats_plot_layout.addWidget(section1b)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

            # ---- 2. 1D Displacement (Δx, Δy) ----
            section2 = CollapsibleSection("1D Displacement (Δx, Δy) — consecutive frames only, Δt = 1")

            for r, st, color in _iter_colors(results_list):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                disp = r['displacements_1d']
                gaussian_fits = r['gaussian_fits']
                subset = disp[disp['state'] == st]
                if subset.empty:
                    continue

                fig2, ax2, ax2_stats = _make_fig_with_stats()
                label = _ds_label(r, st)
                dx_data = subset['dx'].to_numpy()
                dy_data = subset['dy'].to_numpy()

                bins_range = max(abs(np.percentile(np.concatenate([dx_data, dy_data]), [1, 99]))) * 1.2
                bins = np.linspace(-bins_range, bins_range, 80)

                ax2.hist(dx_data, bins=bins, alpha=0.5, color='#ff6666', label=f'{label} Δx',  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                        density=True, edgecolor='none')
                ax2.hist(dy_data, bins=bins, alpha=0.5, color='#66ff66',
                        label=f'{label} Δy', density=True, edgecolor='none')  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

                # Overlay Gaussian fits
                stat_lines = [
                    (f'Δx (n={len(dx_data)})', '#ff6666', np.mean(dx_data), np.std(dx_data)),  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                    (f'Δy (n={len(dy_data)})', '#66ff66', np.mean(dy_data), np.std(dy_data)),
                ]
                if st in gaussian_fits:
                    gf = gaussian_fits[st]
                    if 'dx' in gf:
                        ax2.plot(gf['dx']['x_fit'], gf['dx']['y_fit'], color='#ff6666',
                                linewidth=2, label='Gaussian fit Δx')
                        stat_lines.append(('--- Gaussian Δx ---', '#ff6666', None, None))
                        stat_lines.append((f'Location = {gf["dx"]["mu"]:.4f}', '#ff6666', None, None))
                        stat_lines.append((f'Scale = {gf["dx"]["sigma"]:.4f}', '#ff6666', None, None))
                        stat_lines.append((f'Amplitude = {gf["dx"]["alpha"]:.4f}', '#ff6666', None, None))
                    if 'dy' in gf:
                        ax2.plot(gf['dy']['x_fit'], gf['dy']['y_fit'], color='#66ff66',
                                linewidth=2, linestyle='--', label='Gaussian fit Δy')
                        stat_lines.append(('--- Gaussian Δy ---', '#66ff66', None, None))
                        stat_lines.append((f'Location = {gf["dy"]["mu"]:.4f}', '#66ff66', None, None))
                        stat_lines.append((f'Scale = {gf["dy"]["sigma"]:.4f}', '#66ff66', None, None))
                        stat_lines.append((f'Amplitude = {gf["dy"]["alpha"]:.4f}', '#66ff66', None, None))

                ax2.set_xlabel('Displacement (μm)')
                ax2.set_ylabel('Density')
                ax2.set_title(f'1D Displacement — Gaussian Fit — {label}')
                ax2.legend(fontsize=7, loc='best')
                ax2.grid(True, alpha=0.3)

                _fill_stats_panel(ax2_stats, stat_lines)
                section2.add_widget(_make_canvas(fig2))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

            self._adv_stats_plot_layout.addWidget(section2)

            # ---- 3. 1D Displacement Ratio — Cauchy Fit ----
            section3 = CollapsibleSection("1D Displacement Ratio — Cauchy Fit")

            for r, st, color in _iter_colors(results_list):
                ratios = r['ratios_1d']
                cauchy_fits = r['cauchy_fits']
                subset = ratios[ratios['state'] == st]
                if subset.empty:
                    continue

                fig3, ax3, ax3_stats = _make_fig_with_stats()
                label = _ds_label(r, st)
                ratio_data = subset['ratio'].to_numpy()
                ratio_clipped = ratio_data[(ratio_data > -10) & (ratio_data < 10)]

                if len(ratio_clipped) > 0:
                    ax3.hist(ratio_clipped, bins=100, density=True, alpha=0.6,
                            color=color, edgecolor='none', label=f'{label} data')

                stat_lines = [(f'Ratio (n={len(ratio_clipped)})', color,
                              np.mean(ratio_clipped) if len(ratio_clipped) > 0 else None,
                              np.std(ratio_clipped) if len(ratio_clipped) > 0 else None)]

                if st in cauchy_fits:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                    cf = cauchy_fits[st]
                    ax3.plot(cf['x_fit'], cf['y_fit'], color='#ff6666', linewidth=2,
                            label=f'Cauchy fit (Ĥ={cf["h_est"]:.3f})')
                    ax3.axvline(cf['location'], color='#ff6666', linestyle='--',
                               alpha=0.6, linewidth=1)
                    stat_lines.append((f'Ĥ = {cf["h_est"]:.4f}', '#ff6666', None, None))
                    stat_lines.append((f'Location param = {cf["location"]:.4f}', '#ff6666', None, None))
                    stat_lines.append((f'Amplitude factor = {cf["alpha_est"]:.4f}', '#ff6666', None, None))

                ax3.set_xlabel('Displacement Ratio')
                ax3.set_ylabel('Density')
                ax3.set_title(f'1D Ratio — Cauchy Fit — {label}')
                ax3.legend(fontsize=7, loc='best')
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(-5, 5)

                _fill_stats_panel(ax3_stats, stat_lines)
                section3.add_widget(_make_canvas(fig3))

            self._adv_stats_plot_layout.addWidget(section3)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

        # Apply legend visibility from checkbox  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        if not show_legend:
            for canvas in self._adv_stats_canvases:
                for ax in canvas.figure.get_axes():
                    leg = ax.get_legend()
                    if leg:
                        leg.set_visible(False)
                canvas.draw_idle()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

    def _on_save_adv_stats_plots(self):
        """Save all Advanced Stats plot canvases as PNG files."""
        if not self._adv_stats_canvases:
            QMessageBox.warning(self, "No plots", "Run Advanced Stats first.")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Select directory to save plots")
        if not save_dir:
            return
        plot_names = ['ta_ea_sd', 'ta_ea_sd_loglog']  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        # After the two TA-EA-SD canvases, remaining alternate: displacement, ratio per state
        pair_idx = 0
        for i in range(2, len(self._adv_stats_canvases)):
            if (i - 2) % 2 == 0:
                plot_names.append(f'displacement_1d_{pair_idx}')
            else:
                plot_names.append(f'ratio_cauchy_{pair_idx}')
                pair_idx += 1  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        saved = 0
        for i, canvas in enumerate(self._adv_stats_canvases):
            name = plot_names[i] if i < len(plot_names) else f'adv_plot_{i}'
            path = os.path.join(save_dir, f'{name}.png')
            canvas.figure.savefig(path, dpi=150, bbox_inches='tight', transparent=True)
            saved += 1
        self._adv_stats_status_label.setText(f"Saved {saved} plots to {save_dir}")

    def _on_adv_stats_legend_toggled(self, state):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        show = bool(state)
        for canvas in self._adv_stats_canvases:
            for ax in canvas.figure.get_axes():
                leg = ax.get_legend()
                if leg:
                    leg.set_visible(show)
            canvas.draw_idle()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

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
    def resizeEvent(self, event):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        super().resizeEvent(event)
        self._rescale_help_image()
        # Debounce: wait 80 ms after the last resize before updating fonts
        self._resize_timer.start(80)

    def _apply_fonts(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        current_scale = self._scale()
        if self._last_applied_scale is not None and self._last_applied_scale == current_scale:
            return
        self._last_applied_scale = current_scale  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
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

    def _on_finished(self, success: bool, message: str):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._reset_buttons()
        if success and message.startswith("BATCH_PARTIAL|"):
            # Batch mode: some files failed, some succeeded
            parts = message.split("|", 2)
            output_dir = parts[1]
            summary = parts[2]  # e.g. "=== Batch complete: 1/2 succeeded ==="
            self._output_dir = output_dir
            self._append_log(f"Done (with errors). Results saved to: {output_dir}")
            self._load_results(output_dir)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            self._tabs.setCurrentIndex(1)
            error_log = os.path.join(output_dir, "error_log.txt")
            QMessageBox.information(
                self, "Batch complete",
                f"{summary.strip('= ')}\n\n"
                f"See error_log.txt for details:\n{error_log}"
            )  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        elif success:
            self._output_dir = message
            self._append_log(f"Done. Results saved to: {message}")
            video_path = self._input_path.text().strip()
            if os.path.isdir(video_path):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                self._load_results(message)  # Batch: show all results
            else:
                video_stem = os.path.splitext(os.path.basename(video_path))[0]
                self._load_results(message, video_stem)  # Single file: filter by stem
            self._tabs.setCurrentIndex(1)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        else:
            self._append_log(f"Failed: {message}")
            QMessageBox.critical(self, "FreeTrace error", message)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

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
        paths = sorted(paths)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
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

            color_paths = {}  # (r,g,b,a) -> QPainterPath  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
            grouped = df.sort_values('frame').groupby('traj_idx', sort=False)
            for tidx, traj_data in grouped:
                if len(traj_data) < 2:
                    continue
                xs, ys = traj_data['x'].values, traj_data['y'].values

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

                color_key = (color.red(), color.green(), color.blue(), color.alpha())
                if color_key not in color_paths:
                    color_paths[color_key] = QPainterPath()
                path = color_paths[color_key]
                path.moveTo(float(xs[0]), float(ys[0]))
                for j in range(1, len(xs)):
                    path.lineTo(float(xs[j]), float(ys[j]))

            for color_key, path in color_paths.items():
                scene.addPath(path, QPen(QColor(*color_key), 0.5))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

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

    def _load_results(self, output_dir: str, video_stem: str = ""):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        # Clear previous dynamic result widgets
        for w in self._result_widgets:
            self._results_layout.removeWidget(w)
            w.deleteLater()
        self._result_widgets.clear()

        image_suffixes = {
            "Trajectory Map": "_traces.png",
            "Localisation Density": "_loc_2d_density.png",
            "H-K Distribution": "_diffusion_distribution.png",
        }

        found = False
        if os.path.isdir(output_dir):
            for title, suffix in image_suffixes.items():
                matches = [
                    f for f in os.listdir(output_dir)
                    if f.endswith(suffix) and (not video_stem or f.startswith(video_stem))
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
            csv_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".csv") and (not video_stem or f.startswith(video_stem)))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
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
def main():  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # PyQt6 kills the app on unhandled exceptions in slots — catch and show them instead
    import traceback as _tb
    _original_excepthook = sys.excepthook
    def _excepthook(exc_type, exc_value, exc_traceback):
        msg = "".join(_tb.format_exception(exc_type, exc_value, exc_traceback))
        sys.stderr.write(msg)
        try:
            QMessageBox.critical(None, "Unexpected Error", msg)
        except Exception:
            pass
    sys.excepthook = _excepthook  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18

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
