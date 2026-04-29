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
import time
import shutil

# Make the FreeTrace_cpp/python helper directory importable so `import cauchy_fit` // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
# resolves to the bundled module + cov tables regardless of cwd.
_PY_HELPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python")
if _PY_HELPERS_DIR not in sys.path:
    sys.path.insert(0, _PY_HELPERS_DIR)

import numpy as np
import pandas as pd
from scipy.optimize import minimize  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
import matplotlib  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QProcess, QPointF, QRectF, QUrl  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
from PyQt6.QtGui import (
    QPixmap, QFont, QColor, QPalette, QIcon, QPainter, QPolygon,
    QPen, QBrush, QPainterPath, QDesktopServices, QTransform, QImage,
)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
    QDoubleSpinBox, QSpinBox, QFileDialog, QTextEdit, QSplitter,
    QTabWidget, QScrollArea, QProgressBar, QMessageBox, QRadioButton,
    QButtonGroup, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsPathItem, QComboBox, QSlider, QSizePolicy,
)

# Base window size — font sizes are defined relative to this
_BASE_W, _BASE_H = 1920, 1080

# Current version — used for update check against GitHub releases  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
_VERSION = "1.6.3.0"
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


# ---------------------------------------------------------------------------  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
# Cross-platform orphan process prevention
# ---------------------------------------------------------------------------
def _make_preexec_fn():
    """Return a preexec_fn that ensures the child dies when the parent exits (Linux/macOS)."""
    if sys.platform == "linux":
        import ctypes
        try:
            _libc = ctypes.CDLL("libc.so.6", use_errno=True)
            def _fn():
                _libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
            return _fn
        except OSError:
            return None
    elif sys.platform == "darwin":
        # macOS: no prctl equivalent. preexec_fn threads are killed by exec(),
        # so child-side watching is not viable. Return None and rely on
        # parent-side cleanup (atexit, closeEvent, SIGINT handler).
        return None  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    return None


def _attach_job_object(proc):
    """Windows: assign process to a Job Object that kills children on parent exit."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # Create a job object
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return

        # Configure: kill all processes in job when last handle closes
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [("_" + str(i), ctypes.c_uint64) for i in range(6)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        kernel32.SetInformationJobObject(
            job, 9,  # JobObjectExtendedLimitInformation
            ctypes.byref(info), ctypes.sizeof(info)
        )

        # Assign process to job
        handle = kernel32.OpenProcess(0x1FFFFF, False, proc.pid)  # PROCESS_ALL_ACCESS
        if handle:
            kernel32.AssignProcessToJobObject(job, handle)
            kernel32.CloseHandle(handle)
        # Keep job handle alive (prevent GC) by attaching to process object
        proc._job_handle = job
    except Exception:
        pass  # Best-effort; atexit is the fallback


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
        self._batch_file_idx = 0   # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._batch_total_files = 1  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
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
            cmd = [self.binary] + self.args  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            # Wrap with stdbuf to force line-buffered stdout (prevents output buffering in pipes)
            if sys.platform != "win32" and shutil.which("stdbuf"):
                cmd = ["stdbuf", "-oL"] + cmd
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
                pfn = _make_preexec_fn()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                if pfn:
                    popen_kwargs["preexec_fn"] = pfn

            self._process = subprocess.Popen(cmd, **popen_kwargs)
            _attach_job_object(self._process)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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
            self._current_stage = "loc"  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            for line in iter(self._process.stdout.readline, ''):
                line = line.rstrip("\n")
                # Show each libtiff warning only once per run
                if line.startswith("TIFFReadDirectory: Warning"):
                    if line not in seen_warnings:
                        seen_warnings.add(line)
                        self.log.emit(line)
                    continue
                # Don't show raw PROGRESS protocol lines in the log  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                if not line.startswith("PROGRESS:") and not line.startswith("PROGRESS_BATCH:"):
                    self.log.emit(line)
                    # Collect error/failure lines for the error dialog
                    line_lower = line.lower()
                    if any(kw in line_lower for kw in ("error", "failed", "cannot", "exception", "err:")):
                        self._error_lines.append(line)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20

                # Parse progress from output  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                # Batch file header: PROGRESS_BATCH:<idx>:<total>:<filename>
                if line.startswith("PROGRESS_BATCH:"):
                    bparts = line.split(":", 3)
                    if len(bparts) == 4:
                        try:
                            self._batch_file_idx = int(bparts[1])
                            self._batch_total_files = max(int(bparts[2]), 1)
                        except ValueError:
                            pass
                        self._current_stage = "loc"
                        fname = bparts[3]
                        batch_base = int(self._batch_file_idx * 100 / self._batch_total_files)
                        self.progress.emit(min(batch_base, 99),
                                           f"[{self._batch_file_idx+1}/{self._batch_total_files}] {fname}")
                    continue

                # Stage banners → set current stage
                if "Localization" in line and "===" in line:
                    self._current_stage = "loc"
                elif "Tracking" in line and "===" in line:
                    self._current_stage = "trk"
                elif "Starting trajectory inference" in line:
                    self._current_stage = "trk"

                # PROGRESS:<pct>:<label> lines from C++ binary
                if line.startswith("PROGRESS:"):
                    parts = line.split(":", 2)
                    if len(parts) == 3:
                        try:
                            sub_pct = int(parts[1])
                            label = parts[2]
                        except ValueError:
                            sub_pct, label = 0, ""
                        # Map sub_pct into per-file stage range (0–100%)
                        if "Localizing" in label:
                            file_pct = int(sub_pct * 0.40)        # 0–40%
                        elif "Tracking frame" in label:
                            file_pct = 40 + int(sub_pct * 0.30)   # 40–70%
                        elif "Estimating H" in label:
                            file_pct = 70 + int(sub_pct * 0.20)   # 70–90%
                        elif "Estimating K" in label:
                            file_pct = 92
                        else:
                            file_pct = sub_pct
                        # Map per-file pct into global batch range
                        n = self._batch_total_files
                        idx = self._batch_file_idx
                        overall = int((idx * 100 + file_pct) / n)
                        batch_prefix = f"[{idx+1}/{n}] " if n > 1 else ""
                        self.progress.emit(min(overall, 99), f"{batch_prefix}{label}")
                elif "Batch complete" in line:
                    self._batch_summary = line
                    self.progress.emit(95, "Finishing")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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
            else:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                pfn = _make_preexec_fn()
                if pfn:
                    popen_kw["preexec_fn"] = pfn
            proc = subprocess.Popen(cmd, **popen_kw)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            _attach_job_object(proc)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
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


# Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
def _read_metadata_from_nd2(nd2_path):
    """Extract acquisition metadata from an ND2 file using the `nd2` library.

    Returns a dict matching _read_metadata_from_tif's contract:
      pixel_size_um, finterval_s, exposure_s, R, message.
    Frame interval is derived from per-frame time stamps (median Δ over first 50
    frames) since NETimeLoop periodMs is unreliable for non-equidistant captures.
    Exposure parsed from text_info['capturing'] or ['description'] via regex.
    """
    out = {'pixel_size_um': None, 'finterval_s': None, 'exposure_s': None,
           'R': None, 'message': ''}
    try:
        import nd2
    except Exception as e:
        out['message'] = f"nd2 library not available: {e}"
        return out
    try:
        with nd2.ND2File(nd2_path) as n:
            # Pixel size (μm/px)
            try:
                vs = n.voxel_size()
                if vs.x and vs.x > 0:
                    out['pixel_size_um'] = float(vs.x)
            except Exception:
                pass

            # Frame interval from time stamps
            try:
                n_frames = int(getattr(n.attributes, 'sequenceCount', 0))
                if n_frames >= 2:
                    sample = min(n_frames, 50)
                    times_ms = []
                    for i in range(sample):
                        fm = n.frame_metadata(i)
                        ch = fm.channels[0] if fm.channels else None
                        if ch is None:
                            continue
                        t = float(ch.time.relativeTimeMs)
                        if np.isfinite(t):
                            times_ms.append(t)
                    if len(times_ms) >= 2:
                        diffs = np.diff(np.asarray(times_ms))
                        diffs = diffs[diffs > 0]
                        if diffs.size:
                            out['finterval_s'] = float(np.median(diffs)) / 1000.0
            except Exception:
                pass

            # Exposure from text_info — Andor / NIS-Elements format "Exposure: 30 ms"
            try:
                ti = n.text_info or {}
                blob = '\n'.join(str(v) for v in ti.values() if isinstance(v, str))
                import re
                m = re.search(r'Exposure[^\n]*?:\s*([\d.]+)\s*(ms|s)\b', blob, re.IGNORECASE)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2).lower()
                    out['exposure_s'] = val / 1000.0 if unit == 'ms' else val
            except Exception:
                pass
    except Exception as e:
        out['message'] = f"Could not read ND2: {e}"
        return out

    # Compute R
    if out['finterval_s'] and out['exposure_s']:
        R = out['exposure_s'] / out['finterval_s']
        if 0.0 < R <= 1.01:
            out['R'] = min(R, 1.0)

    parts = []
    if out['pixel_size_um'] is not None:
        parts.append(f"pixel size = {out['pixel_size_um']:.4f} μm/px")
    if out['finterval_s'] is not None:
        parts.append(f"Δt = {out['finterval_s']*1000:.3f} ms")
    if out['exposure_s'] is not None:
        parts.append(f"τ_exp = {out['exposure_s']*1000:.3f} ms")
    if out['R'] is not None:
        parts.append(f"R = {out['R']:.4f}")
    out['message'] = ", ".join(parts) if parts else "No usable metadata found in ND2."
    return out


# Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
def _read_metadata_from_video(path):
    """Dispatch to TIFF or ND2 reader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.nd2':
        return _read_metadata_from_nd2(path)
    return _read_metadata_from_tif(path)


# Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
def _read_metadata_from_tif(tif_path):
    """Extract acquisition metadata from a TIFF file (NIS-Elements-style ImageJ Info).

    Returns a dict with keys (any may be None if unavailable):
      - pixel_size_um:  pixel size in μm (NIS-Elements 'dCalibration' field)
      - finterval_s:    frame interval in seconds (ImageJ 'finterval' tag)
      - exposure_s:     exposure time in seconds (NIS-Elements 'Exposure time (text)')
      - R:              τ_exp / Δt ∈ [0,1] (computed if both above are present)
      - message:        human-readable status string
    """
    out = {'pixel_size_um': None, 'finterval_s': None, 'exposure_s': None,
           'R': None, 'message': ''}
    try:
        import tifffile
    except Exception as e:
        out['message'] = f"tifffile not available: {e}"
        return out
    try:
        with tifffile.TiffFile(tif_path) as tif:
            ij = tif.imagej_metadata or {}
            # Frame interval (ImageJ tag, seconds)
            try:
                if ij.get('finterval') is not None:
                    val = float(ij['finterval'])
                    if val > 0:
                        out['finterval_s'] = val
            except Exception:
                pass
            info_text = ij.get('Info', '')
            if not isinstance(info_text, str):
                info_text = ''
            # NIS-Elements: 'Exposure time (text) = 0.0XXXX' (seconds)
            for line in info_text.split('\n'):
                if 'Exposure time (text)' in line:
                    try:
                        out['exposure_s'] = float(line.split('=', 1)[1].strip())
                        break
                    except Exception:
                        pass
            # Fallback: 'Exposure = X' (ms)
            if out['exposure_s'] is None:
                for line in info_text.split('\n'):
                    s = line.strip()
                    if s.startswith('Exposure ='):
                        try:
                            out['exposure_s'] = float(s.split('=', 1)[1].strip()) / 1000.0
                            break
                        except Exception:
                            pass
            # NIS-Elements pixel-size: 'dCalibration = 0.16' (μm/px)
            for line in info_text.split('\n'):
                s = line.strip()
                if s.startswith('dCalibration ='):
                    try:
                        v = float(s.split('=', 1)[1].strip())
                        if v > 0:
                            out['pixel_size_um'] = v
                            break
                    except Exception:
                        pass
    except Exception as e:
        out['message'] = f"Could not read TIFF: {e}"
        return out

    # Compute R
    if out['finterval_s'] and out['exposure_s']:
        R = out['exposure_s'] / out['finterval_s']
        if 0.0 < R <= 1.01:
            out['R'] = min(R, 1.0)

    # Human-readable summary
    parts = []
    if out['pixel_size_um'] is not None:
        parts.append(f"pixel size = {out['pixel_size_um']:.4f} μm/px")
    if out['finterval_s'] is not None:
        parts.append(f"Δt = {out['finterval_s']*1000:.3f} ms")
    if out['exposure_s'] is not None:
        parts.append(f"τ_exp = {out['exposure_s']*1000:.3f} ms")
    if out['R'] is not None:
        parts.append(f"R = {out['R']:.4f}")
    if parts:
        out['message'] = ", ".join(parts)
    else:
        out['message'] = "No usable metadata found in TIFF."
    return out


# Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
def _trajs_from_traces_df(traces_df, state, min_length=3):
    """Convert a per-state traces DataFrame into a list of (T, 3) arrays
    [frame, x, y] expected by cauchy_fit.extract_ratios.

    Uses the trajectory's reconstructed state field if present (defaults to 0).
    Trajectories with length < min_length are dropped.
    """
    sub = traces_df[traces_df['state'] == state] if 'state' in traces_df.columns else traces_df
    out = []
    for _, group in sub.groupby('traj_idx'):
        g = group.sort_values('frame')
        if len(g) < min_length:
            continue
        # cauchy_fit.extract_ratios expects (T, >=3) with cols (1,2) = x, y.
        out.append(g[['frame', 'x', 'y']].to_numpy(dtype=np.float64))
    return out


def _estimate_K_corrected_msd(trajs, sigma_loc_px, R, max_lag=10):
    """Fit K (and a side H) from the corrected ensemble-averaged TAMSD over τ=1..max_lag.

    Model: MSD(τ) = 2*K*J_var(H, R, τ) + 2*sigma_loc_px².
    Bounded curve_fit in (K, H) over τ=1..max_lag with SEM weighting (matches thesis
    fit_K_from_msd in PhD_thesis/.../ch1_h2b_cauchy_multidelta_crlb.py).

    Returns (K_est, H_est) or (None, None) if the fit fails / data insufficient.
    """  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
    from scipy.optimize import curve_fit
    try:
        from cauchy_fit import J_var
    except Exception:
        return None, None
    msd_vals, msd_sem = {}, {}
    for tau in range(1, max_lag + 1):
        sds = []
        for tr in trajs:
            for coord in (1, 2):
                p = tr[:, coord]
                if len(p) < tau + 1:
                    continue
                d = p[tau:] - p[:-tau]
                sds.extend(d * d)
        if sds:
            arr = np.asarray(sds, dtype=np.float64)
            msd_vals[tau] = float(arr.mean())
            sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else float(arr.mean())
            msd_sem[tau] = max(float(sem), 1e-12)
    if len(msd_vals) < 2:
        return None, None
    taus = np.array(sorted(msd_vals.keys()), dtype=np.float64)
    msd = np.array([msd_vals[int(t)] for t in taus])
    sem = np.array([msd_sem[int(t)] for t in taus])
    s2 = float(sigma_loc_px) ** 2 if sigma_loc_px is not None and np.isfinite(sigma_loc_px) else 0.0

    def model(t, K, H):
        return np.array([2.0 * K * J_var(float(H), float(R), float(tt)) + 2.0 * s2 for tt in t])

    p0 = [max(msd[0] / 2.0 - s2, 1e-3), 0.4]
    try:
        (K_est, H_est), _ = curve_fit(
            model, taus, msd, p0=p0, sigma=sem, absolute_sigma=True,
            maxfev=20000, bounds=([1e-4, 0.01], [10.0, 0.99]),
        )
        return float(K_est), float(H_est)
    except Exception:
        return None, None


def _invert_H_from_rho(rho_target, Delta, R, K, sigma_loc,  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                        H_bounds=(0.02, 0.98)):
    """Bisection-solve rho_corrected(H, Δ, R, K, σ_loc) = rho_target.

    Used by the K- and σ_loc-sensitivity panels: keep the empirical Cauchy location
    fixed and ask "what H would the corrected model report under a perturbed K or
    σ_loc?". Returns NaN if the target is outside the achievable range on H_bounds.
    """
    try:
        from cauchy_fit import rho_corrected
    except Exception:
        return np.nan
    if not np.isfinite(rho_target):
        return np.nan
    lo, hi = H_bounds
    f_lo = rho_corrected(lo, Delta, R, K, sigma_loc) - rho_target
    f_hi = rho_corrected(hi, Delta, R, K, sigma_loc) - rho_target
    if f_lo * f_hi > 0:
        # Target outside achievable range — return the closer endpoint.
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = rho_corrected(mid, Delta, R, K, sigma_loc) - rho_target
        if abs(f_mid) < 1e-9 or (hi - lo) < 1e-7:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(0.5 * (lo + hi))


def _run_multi_delta_scan(trajs, sigma_loc_rms_px, K, R, delta_max=None,
                          delta_max_cap=75, min_n_ratios=30):
    """Multi-Δ corrected-Cauchy scan over Δ = 1 .. Δ_max.

    Auto-selects Δ_max as the largest Δ ≤ delta_max_cap with n_ratios ≥ min_n_ratios,
    unless delta_max is explicitly supplied. Returns a dict with arrays:
        deltas:    array of Δ values
        H_est:     Ĥ(Δ)
        n_ratios:  ratio counts per Δ
        converged: bool array
        n_eff:     theoretical n_eff(Δ) at fitted Ĥ(Δ)
        sigma_H:   CRLB σ_H(Δ) = 1/sqrt(n_eff·I_1_H), evaluated at fitted Ĥ
                   (95% CI band half-width = 1.96 * sigma_H)
    """  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
    try:
        from cauchy_fit import (extract_ratios, count_ratios, fit_cauchy,
                                 n_eff_theory, sigma_H_crlb)
    except Exception:
        return None
    if not trajs:
        return None
    # Pick Δ_max
    if delta_max is None:
        delta_max = 1
        for d in range(1, delta_max_cap + 1):
            if count_ratios(trajs, d) >= min_n_ratios:
                delta_max = d
            else:
                break
        if delta_max < 1:
            return None
    deltas = np.arange(1, int(delta_max) + 1)
    H_est = np.full(len(deltas), np.nan)
    rho_arr = np.full(len(deltas), np.nan)  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
    n_ratios_arr = np.zeros(len(deltas), dtype=np.int64)
    converged = np.zeros(len(deltas), dtype=bool)
    n_eff_arr = np.full(len(deltas), np.nan)
    sigma_H_arr = np.full(len(deltas), np.nan)
    s_loc = float(sigma_loc_rms_px) if sigma_loc_rms_px is not None else 0.0
    traj_lengths = [len(tr) for tr in trajs]
    for i, d in enumerate(deltas):
        ratios = extract_ratios(trajs, int(d))
        n_ratios_arr[i] = ratios.size
        if ratios.size < min_n_ratios:
            continue
        try:
            res = fit_cauchy(ratios, Delta=int(d), R=float(R), K=float(K),
                             sigma_loc=s_loc)
            H_est[i] = res.get('H', np.nan)
            rho_arr[i] = res.get('rho', np.nan)
            converged[i] = bool(res.get('converged', False))
        except Exception:
            continue
        # CRLB band — evaluate at the fitted Ĥ. n_eff_theory snaps H to {0.25,0.5,0.75}
        # internally via _load_cov_table.
        H_hat = float(H_est[i])
        if not np.isfinite(H_hat):
            continue
        try:
            n_eff = n_eff_theory(H_hat, int(d), traj_lengths,
                                 R=float(R), K=float(K), sigma_loc=s_loc)
            if np.isfinite(n_eff) and n_eff > 0:
                n_eff_arr[i] = float(n_eff)
                sigma_H_arr[i] = sigma_H_crlb(H_hat, int(d), float(R), float(K),
                                              s_loc, n_eff)
        except Exception:
            pass
    return {
        'deltas': deltas,
        'H_est': H_est,
        'rho_arr': rho_arr,
        'n_ratios': n_ratios_arr,
        'converged': converged,
        'n_eff': n_eff_arr,
        'sigma_H': sigma_H_arr,
        'delta_max': int(delta_max),
        'traj_lengths': traj_lengths,
    }


# Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
def _recompute_bg_var_and_flux_from_tif(loc_df, tif_path, R_BG=6, R_SIGNAL=3):
    """Recompute per-spot bg_var, bg_median, and integrated_flux from the raw TIFF.

    Mirrors PhD_thesis/Figures/tmp/ch1_h2b_crlb.py exactly:
      - bg pixels: 13×13 patch around the spot (R_BG=6) with a central R_SIGNAL=3
        disk masked out → annulus.
      - bg_median = np.median(annulus); bg_var = np.var(annulus).
      - integrated_flux = sum(window) − window.size · bg_median   (N_direct).

    Returns dict of arrays aligned with loc_df rows, NaN where the patch can't be
    cropped or fewer than 10 annulus pixels remain. Used to bypass the localiser's
    own bg_var / integrated_flux columns when a raw TIFF is alongside the loc.csv,
    so σ_loc matches the thesis convention (otherwise the localiser-saved values
    underestimate bg_var → σ_loc is biased low → K is biased high).
    """
    import tifffile
    imgs = tifffile.imread(tif_path)
    if imgs.ndim == 2:
        imgs = imgs[np.newaxis, ...]
    T_im, H_im, W_im = imgs.shape
    n = len(loc_df)
    bg_var_arr = np.full(n, np.nan)
    bg_med_arr = np.full(n, np.nan)
    flux_arr = np.full(n, np.nan)
    frames = loc_df['frame'].to_numpy(dtype=int)
    xc = loc_df['x'].to_numpy(dtype=np.float64)
    yc = loc_df['y'].to_numpy(dtype=np.float64)
    ws = loc_df['window_size'].to_numpy(dtype=int)
    for i in range(n):
        f = frames[i] - 1
        if not (0 <= f < T_im):
            continue
        cx = int(round(xc[i])); cy = int(round(yc[i]))
        x0 = max(0, cx - R_BG); x1 = min(W_im, cx + R_BG + 1)
        y0 = max(0, cy - R_BG); y1 = min(H_im, cy + R_BG + 1)
        patch = imgs[f, y0:y1, x0:x1].astype(np.float64)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        bg_pix = patch[r2 > R_SIGNAL ** 2]
        if bg_pix.size < 10:
            continue
        bg_med = float(np.median(bg_pix))
        bg_med_arr[i] = bg_med
        bg_var_arr[i] = float(np.var(bg_pix))
        w = int(ws[i]); r = w // 2
        x0w = max(0, cx - r); x1w = min(W_im, cx + r + 1)
        y0w = max(0, cy - r); y1w = min(H_im, cy + r + 1)
        wp = imgs[f, y0w:y1w, x0w:x1w].astype(np.float64)
        flux_arr[i] = float(wp.sum() - wp.size * bg_med)
    return dict(bg_var=bg_var_arr, bg_median=bg_med_arr, integrated_flux=flux_arr)


def _find_sibling_tif(loc_path, traces_path, video_name):
    """Search for a TIFF matching <video_name> next to loc.csv or traces.csv."""
    candidates = []
    for ext in ('.tif', '.tiff'):
        for d in {os.path.dirname(loc_path), os.path.dirname(traces_path)}:
            candidates.append(os.path.join(d, video_name + ext))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None
# End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29


def _compute_sigma_loc_per_spot(loc_df):
    """Per-spot localisation precision (sigma_loc, in pixels) from a _loc.csv DataFrame.

    Mirrors the thesis script /home/junwoo/claude/PhD_thesis/Figures/tmp/ch1_h2b_crlb.py:
      1. Capture-corrected total photon count: I_tot = integrated_flux / capture
         where capture = erf(r/(σx√2)) * erf(r/(σy√2)), r = window_size/2, clipped to [0.5,1].
      2. Per-spot 2x2 positional Fisher info via analytic ∂μ/∂x, ∂μ/∂y on the fit window,
         using bg_var as the per-pixel noise variance.
      3. Inversion → var_x, var_y; σ_loc = 0.5*(σx + σy) (isotropic average).

    Requires columns: x, y, xvar, yvar, rho, window_size, bg_var, integrated_flux.
    Returns dict with per-spot arrays:  // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
        sigma_loc_px : σ_loc (pixels) — NaN where the fit is invalid
        I_tot        : capture-corrected total photons (ADU)
        bg_var       : per-pixel background variance (raw ADU²)
        psf_sigma_px : 0.5*(σx + σy) of the fitted PSF in pixels
    For backward compatibility callers can wrap with .get('sigma_loc_px') or pass the
    dict to np.asarray (which yields the dict, not the array — callers must extract).
    """
    try:
        from scipy.special import erf
    except Exception:
        n0 = len(loc_df)
        nan_arr = np.full(n0, np.nan)
        return dict(sigma_loc_px=nan_arr, I_tot=nan_arr, bg_var=nan_arr, psf_sigma_px=nan_arr)

    xc = loc_df['x'].to_numpy(dtype=np.float64)
    yc = loc_df['y'].to_numpy(dtype=np.float64)
    xv = loc_df['xvar'].to_numpy(dtype=np.float64)
    yv = loc_df['yvar'].to_numpy(dtype=np.float64)
    rho = loc_df['rho'].to_numpy(dtype=np.float64)
    ws_arr = loc_df['window_size'].to_numpy(dtype=np.int32)
    bg_var = loc_df['bg_var'].to_numpy(dtype=np.float64)
    flux = loc_df['integrated_flux'].to_numpy(dtype=np.float64)

    n = len(loc_df)
    sig_loc_px = np.full(n, np.nan)
    I_tot_arr = np.full(n, np.nan)
    psf_sigma_px = np.full(n, np.nan)
    for i in range(n):
        u = xv[i]; v = yv[i]; r_ = rho[i]
        if not (u > 0 and v > 0):
            continue
        if not (np.isfinite(bg_var[i]) and bg_var[i] > 0):
            continue
        if not np.isfinite(flux[i]):
            continue
        sx = np.sqrt(u); sy = np.sqrt(v)
        psf_sigma_px[i] = 0.5 * (sx + sy)
        # Capture-fraction correction (window may truncate Gaussian tails)
        w = int(ws_arr[i]); r_half = w / 2.0
        cap = float(erf(r_half / (sx * np.sqrt(2.0))) * erf(r_half / (sy * np.sqrt(2.0))))
        cap = max(0.5, min(1.0, cap))
        I_tot = flux[i] / cap
        if not (I_tot > 0):
            continue
        I_tot_arr[i] = I_tot

        k = 1.0 - r_ * r_
        if k <= 0:
            continue

        cx_int = int(round(xc[i]))
        cy_int = int(round(yc[i]))
        nn = np.arange(cx_int - w // 2, cx_int + w // 2 + 1) - xc[i]   # column offsets
        mm = np.arange(cy_int - w // 2, cy_int + w // 2 + 1) - yc[i]   # row offsets
        N_grid, M_grid = np.meshgrid(nn, mm, indexing='xy')
        Q = 0.5 / k * (N_grid ** 2 / u + M_grid ** 2 / v
                       - 2.0 * r_ * N_grid * M_grid / (sx * sy))
        mu_star = (I_tot / (2.0 * np.pi * sx * sy * np.sqrt(k))) * np.exp(-Q)
        dmu_dx = mu_star / k * (N_grid / u - r_ * M_grid / (sx * sy))
        dmu_dy = mu_star / k * (M_grid / v - r_ * N_grid / (sx * sy))
        inv_sn2 = 1.0 / bg_var[i]
        Ixx = inv_sn2 * np.sum(dmu_dx * dmu_dx)
        Iyy = inv_sn2 * np.sum(dmu_dy * dmu_dy)
        Ixy = inv_sn2 * np.sum(dmu_dx * dmu_dy)
        det = Ixx * Iyy - Ixy * Ixy
        if det > 0:
            var_x = Iyy / det
            var_y = Ixx / det
            sig_loc_px[i] = 0.5 * (np.sqrt(var_x) + np.sqrt(var_y))
    return dict(sigma_loc_px=sig_loc_px, I_tot=I_tot_arr,
                bg_var=bg_var, psf_sigma_px=psf_sigma_px)


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
                # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                # Suppress divide-by-zero / invalid-value warnings for the rare zero-denominator
                # cases — the np.isfinite filter below drops inf/nan rows.
                with np.errstate(divide='ignore', invalid='ignore'):
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
                # Also compute noise-free advanced stats (TA-EA-SD, ratios, Cauchy fit at Δ=1) // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                # — these were previously in the Advanced Stats tab; moved to Basic since they
                # are also noise-free / empirical. Advanced Stats now keeps only 1D Displacement+Gaussian.
                tamsd_df = ratios_1d_df = cauchy_fits = None
                try:
                    adv_result = _preprocess_for_adv_stats(
                        data, self._pixelmicrons, self._framerate,
                        cutoff_min=self._cutoff_min,
                    )
                    tamsd_df, _disp_df, ratios_1d_df, cauchy_fits, _gauss_fits, _ts = adv_result
                except Exception:
                    pass  # adv preprocessing failure is non-fatal for basic stats
                if ad1 is not None and len(ad1) > 0:
                    all_results.append({
                        'name': name,
                        'analysis_data1': ad1,
                        'analysis_data2': ad2,
                        'analysis_data3': ad3,
                        'msd': msd,
                        'total_states': total_states,
                        'has_diffusion': has_diff,
                        # noise-free adv data (may be None if preprocessing failed)
                        'tamsd': tamsd_df,
                        'ratios_1d': ratios_1d_df,
                        'cauchy_fits': cauchy_fits,
                    })
            if not all_results:
                self.error.emit("No data remaining after filtering.")
                return
            self.progress.emit(95, "Building results...")
            self.finished.emit(all_results)
        except Exception as e:
            self.error.emit(str(e))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19


class AdvStatsWorker(QThread):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
    """Background worker for Advanced Stats — runs TAMSD + 1D displacement + Cauchy fit
    + (when σ_loc and R are available) corrected-Cauchy multi-Δ scan per state.

    Multi-Δ scan operates on raw pixel-coordinate trajectories (cauchy_fit's K is in
    px²/frame^(2H), σ_loc in pixels, R unitless). Skipped per-dataset when σ_loc_rms_px
    is None (i.e. the sibling _loc.csv lacks bg_median/bg_var/integrated_flux columns).
    """  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, datasets, pixelmicrons, framerate, cutoff_min,
                 R=0.0, sigma_loc_rms_per_dataset=None):  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
        """datasets: list of (name, DataFrame) tuples.
        R: motion-blur fraction (toolbar spinbox value); when 0 the multi-Δ scan still
           runs but assumes no motion blur.
        sigma_loc_rms_per_dataset: list aligned with datasets of σ_loc_rms in pixels
           (or None when CRLB columns are absent). When entry is None, multi-Δ scan
           is skipped for that dataset.
        """
        super().__init__()
        self._datasets = datasets
        self._pixelmicrons = pixelmicrons
        self._framerate = framerate
        self._cutoff_min = cutoff_min
        self._R = float(R)
        if sigma_loc_rms_per_dataset is None:
            sigma_loc_rms_per_dataset = [None] * len(datasets)
        self._sigma_loc_rms_per_dataset = sigma_loc_rms_per_dataset

    def run(self):
        try:
            all_results = []
            n = len(self._datasets)
            for idx, (name, data) in enumerate(self._datasets):
                pct = int(10 + 70 * idx / max(n, 1))
                self.progress.emit(pct, f"Processing {name}...")
                result = _preprocess_for_adv_stats(
                    data, self._pixelmicrons, self._framerate,
                    cutoff_min=self._cutoff_min,
                )
                tamsd_df, disp_df, ratios_df, cauchy_fits, gaussian_fits, total_states = result  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                n_trajs = int(data['traj_idx'].nunique())
                if tamsd_df is None or len(tamsd_df) == 0:
                    continue

                # Multi-Δ corrected-Cauchy scan per state (in pixel coords). // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                # Skip the scan when raw MSD(τ=1) is at or below the localisation noise
                # floor 2σ_loc² (signal/noise ≤ 1) — the corrected Cauchy fit cannot
                # recover Ĥ when there is no diffusion signal above noise. We surface a
                # diagnostic to the user instead of plotting H values that hit the
                # boundary {0.02, 0.98} as artefacts.
                multi_delta_per_state = {}
                K_est_per_state = {}
                noise_floor_per_state = {}  # (msd1, 2σ², s/n)
                sigma_loc_rms_px = self._sigma_loc_rms_per_dataset[idx] if idx < len(self._sigma_loc_rms_per_dataset) else None
                if sigma_loc_rms_px is not None and np.isfinite(sigma_loc_rms_px):
                    self.progress.emit(pct + 5, f"Multi-Δ scan {name}...")
                    for st in total_states:
                        trajs = _trajs_from_traces_df(data, st, min_length=max(self._cutoff_min, 3))
                        if len(trajs) < 5:
                            continue
                        # Per-state signal-to-noise check.
                        sds = []
                        for tr in trajs:
                            for c in (1, 2):
                                p = tr[:, c]
                                if len(p) >= 2:
                                    sds.append(((p[1:] - p[:-1]) ** 2))
                        if not sds:
                            continue
                        msd1 = float(np.concatenate(sds).mean())
                        noise_floor = 2.0 * float(sigma_loc_rms_px) ** 2
                        snr = msd1 / noise_floor if noise_floor > 0 else np.inf
                        noise_floor_per_state[st] = (msd1, noise_floor, snr)
                        # Run K_est unconditionally; rely on its own lower-bound rejection // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                        # rather than a fixed S/N threshold (testsample4 region subsets sit
                        # at S/N≈1.4 but still have a clear MSD growth signal).
                        K_est, _H_init = _estimate_K_corrected_msd(
                            trajs, sigma_loc_rms_px, self._R, max_lag=10,
                        )
                        if K_est is None or not np.isfinite(K_est) or K_est <= 0:
                            continue
                        # Reject K_est pinned to the curve_fit lower bound (no diffusion signal).
                        if K_est <= 1.01e-4:
                            continue
                        K_est_per_state[st] = float(K_est)
                        scan = _run_multi_delta_scan(
                            trajs, sigma_loc_rms_px, K_est, self._R,
                            delta_max=None, delta_max_cap=75, min_n_ratios=30,
                        )
                        if scan is not None:
                            multi_delta_per_state[st] = scan

                all_results.append({
                    'name': name,
                    'tamsd': tamsd_df,
                    'displacements_1d': disp_df,
                    'ratios_1d': ratios_df,
                    'cauchy_fits': cauchy_fits,
                    'gaussian_fits': gaussian_fits,
                    'total_states': total_states,
                    'n_trajectories': n_trajs,
                    'multi_delta_per_state': multi_delta_per_state,  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                    'K_est_per_state': K_est_per_state,
                    'noise_floor_per_state': noise_floor_per_state,  # state -> (msd1, 2σ², snr)
                    'sigma_loc_rms_px': sigma_loc_rms_px,
                    'R_used': self._R,
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

    _MARGIN_LEFT = 120  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    _MARGIN_BOTTOM = 100
    _MARGIN_TOP = 60
    _MARGIN_RIGHT = 60
    _PLOT_W = 1000
    _PLOT_H = 800  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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
        scene_font = QFont()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        scene_font.setPointSize(18)  # scaled for 1000x800 scene

        self._scene.addRect(
            QRectF(self._MARGIN_LEFT, self._MARGIN_TOP, self._PLOT_W, self._PLOT_H),
            QPen(Qt.PenStyle.NoPen), QBrush(QColor(30, 30, 30))
        )

        for h_val in np.arange(0.0, 1.01, 0.1):
            x = self._h_to_x(h_val)
            self._scene.addLine(x, self._MARGIN_TOP, x, self._MARGIN_TOP + self._PLOT_H, pen_grid)
            txt = self._scene.addSimpleText(f"{h_val:.1f}", scene_font)
            txt.setBrush(pen_text)
            txt.setPos(x - 18, self._MARGIN_TOP + self._PLOT_H + 8)

        for logk_val in range(int(self._logk_min), int(self._logk_max) + 1):
            y = self._logk_to_y(logk_val)
            self._scene.addLine(self._MARGIN_LEFT, y, self._MARGIN_LEFT + self._PLOT_W, y, pen_grid)
            txt = self._scene.addSimpleText(f"1e{logk_val}", scene_font)
            txt.setBrush(pen_text)
            txt.setPos(self._MARGIN_LEFT - 80, y - 12)

        self._scene.addLine(
            self._MARGIN_LEFT, self._MARGIN_TOP + self._PLOT_H,
            self._MARGIN_LEFT + self._PLOT_W, self._MARGIN_TOP + self._PLOT_H, pen_axis
        )
        self._scene.addLine(
            self._MARGIN_LEFT, self._MARGIN_TOP,
            self._MARGIN_LEFT, self._MARGIN_TOP + self._PLOT_H, pen_axis
        )

        x_label = self._scene.addSimpleText("H (Hurst exponent)", scene_font)
        x_label.setBrush(pen_text)
        x_label.setPos(self._MARGIN_LEFT + self._PLOT_W / 2 - 100, self._MARGIN_TOP + self._PLOT_H + 45)

        y_label = self._scene.addSimpleText("K", scene_font)
        y_label.setBrush(pen_text)
        y_label.setPos(8, self._MARGIN_TOP + self._PLOT_H / 2 - 12)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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
        dot_r = 5.0  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
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
        # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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


# ---------------------------------------------------------------------------  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
# ImageJ/Fiji ROI file parser
# ---------------------------------------------------------------------------
def _parse_imagej_roi(data):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    """Parse a single ImageJ .roi binary blob into a shape dict.

    Returns a dict with 'type' and coordinates in image pixel space,
    or None if the ROI type is unsupported.
    Supported: rect, oval, line, rotated rect, rotated ellipse.
    """
    import struct

    if len(data) < 64 or data[0:4] != b'Iout':
        return None

    def gi16(off):
        return struct.unpack_from('>h', data, off)[0]

    def gu16(off):
        return struct.unpack_from('>H', data, off)[0]

    def gi32(off):
        return struct.unpack_from('>i', data, off)[0]

    def gf32(off):
        return struct.unpack_from('>f', data, off)[0]

    version = gu16(4)
    roi_type = data[6]
    top = gi16(8)
    left = gi16(10)
    bottom = gi16(12)
    right = gi16(14)
    subtype = gu16(48) if len(data) > 50 else 0
    options = gu16(50) if len(data) > 52 else 0

    # Sub-pixel bounding box (version >= 223 and SUB_PIXEL_RESOLUTION flag set)
    sub_pixel = bool(options & 128) and version >= 223

    # Read name from header2 if available
    name = None
    hdr2_off = gi32(60) if len(data) > 63 else 0
    if hdr2_off > 0 and hdr2_off + 24 <= len(data):
        name_off = gi32(hdr2_off + 16)
        name_len = gi32(hdr2_off + 20)
        if name_off > 0 and name_len > 0 and name_off + name_len * 2 <= len(data):
            name = data[name_off:name_off + name_len * 2].decode('utf-16-be', errors='replace')

    result = {'name': name}

    if roi_type == 1:  # rect
        if sub_pixel:
            xd, yd = gf32(18), gf32(22)
            wd, hd = gf32(26), gf32(30)
            result.update({'type': 'rect', 'x1': xd, 'y1': yd,
                           'x2': xd + wd, 'y2': yd + hd, 'angle': 0.0})
        else:
            result.update({'type': 'rect', 'x1': float(left), 'y1': float(top),
                           'x2': float(right), 'y2': float(bottom), 'angle': 0.0})
        return result

    elif roi_type == 2:  # oval
        if sub_pixel:
            xd, yd = gf32(18), gf32(22)
            wd, hd = gf32(26), gf32(30)
            result.update({'type': 'ellipse',
                           'cx': xd + wd / 2, 'cy': yd + hd / 2,
                           'rx': wd / 2, 'ry': hd / 2, 'angle': 0.0})
        else:
            cx = (left + right) / 2.0
            cy = (top + bottom) / 2.0
            rx = (right - left) / 2.0
            ry = (bottom - top) / 2.0
            result.update({'type': 'ellipse', 'cx': cx, 'cy': cy,
                           'rx': rx, 'ry': ry, 'angle': 0.0})
        return result

    elif roi_type == 3:  # line
        x1, y1 = gf32(18), gf32(22)
        x2, y2 = gf32(26), gf32(30)
        result.update({'type': 'line', 'p1': [x1, y1], 'p2': [x2, y2]})
        return result

    elif roi_type == 7:  # freehand — check subtype
        if subtype == 3:  # ELLIPSE
            # X1,Y1,X2,Y2 = major axis endpoints; FLOAT_PARAM = aspect ratio
            x1, y1 = gf32(18), gf32(22)
            x2, y2 = gf32(26), gf32(30)
            aspect = gf32(52) if len(data) > 55 else 1.0
            # Major axis length and center
            dx, dy = x2 - x1, y2 - y1
            major = math.sqrt(dx * dx + dy * dy)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx = major / 2  # semi-major
            ry = rx * aspect  # semi-minor
            angle = math.degrees(math.atan2(dy, dx))
            result.update({'type': 'ellipse', 'cx': cx, 'cy': cy,
                           'rx': rx, 'ry': ry, 'angle': angle})
            return result

        elif subtype == 5:  # ROTATED_RECT
            # X1,Y1,X2,Y2 = one side; FLOAT_PARAM = width (perpendicular)
            x1, y1 = gf32(18), gf32(22)
            x2, y2 = gf32(26), gf32(30)
            width = gf32(52) if len(data) > 55 else 10.0
            dx, dy = x2 - x1, y2 - y1
            side_len = math.sqrt(dx * dx + dy * dy)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            angle = math.degrees(math.atan2(dy, dx))
            # side_len = length along the defined side, width = perpendicular
            result.update({'type': 'rect',
                           'x1': cx - side_len / 2, 'y1': cy - width / 2,
                           'x2': cx + side_len / 2, 'y2': cy + width / 2,
                           'angle': angle})
            return result

    # Unsupported type (polygon, polyline, freehand, etc.)
    return None


def _load_imagej_rois(path):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    """Load ROI shapes from an ImageJ .roi or .zip file.

    Returns (shapes_list, skipped_count) where shapes_list contains
    dicts compatible with ROICanvas.set_shapes_data().
    """
    import zipfile

    shapes = []
    skipped = 0

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, 'r') as zf:
            for entry in zf.namelist():
                if not entry.lower().endswith('.roi'):
                    continue
                roi_data = zf.read(entry)
                parsed = _parse_imagej_roi(roi_data)
                if parsed is not None:
                    shapes.append(parsed)
                else:
                    skipped += 1
    else:
        with open(path, 'rb') as f:
            roi_data = f.read()
        parsed = _parse_imagej_roi(roi_data)
        if parsed is not None:
            shapes.append(parsed)
        else:
            skipped += 1

    return shapes, skipped


# ---------------------------------------------------------------------------
# ROI Canvas — spatial trajectory plot with rectangle/ellipse/line ROIs
# ---------------------------------------------------------------------------
class ROICanvas(QGraphicsView):
    """Interactive spatial trajectory plot with structured ROI drawing.

    Supports four modes: Rectangle, Ellipse, Line (draw shapes),
    and Select (move/resize existing shapes).
    Line mode auto-extends endpoints to the plot edges.
    Each shape boundary divides space via flood fill → roi0, roi1, ...
    Right-click removes the last drawn shape.
    Two classification modes: Mean Position (by trajectory centroid)
    and Strict Containment (all points must be in same ROI).
    """
    roi_changed = pyqtSignal()
    mode_requested = pyqtSignal(str)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    _MARGIN_LEFT = 120  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    _MARGIN_BOTTOM = 100
    _MARGIN_TOP = 60
    _MARGIN_RIGHT = 60
    _PLOT_W = 1000
    _PLOT_H = 1000  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27

    _ROI_COLORS = [
        QColor(100, 180, 255, 200),   # blue
        QColor(255, 120, 80, 200),    # orange
        QColor(100, 220, 100, 200),   # green
        QColor(200, 100, 255, 200),   # purple
        QColor(255, 220, 60, 200),    # yellow
        QColor(255, 100, 200, 200),   # pink
        QColor(100, 220, 220, 200),   # cyan
        QColor(220, 180, 100, 200),   # tan
    ]

    MODE_RECT = "Rectangle"
    MODE_ELLIPSE = "Ellipse"
    MODE_LINE = "Line"
    MODE_SELECT = "Select"

    CLASSIFY_MEAN = "Mean Position"
    CLASSIFY_STRICT = "Strict Containment"

    _HANDLE_SIZE = 14  # pixels, half-width of resize handles  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("background:#1a1a1a; border:none;")

        # Trajectory data: per-trajectory point lists and mean positions
        self._traj_indices = np.array([])
        self._traj_points = []  # list of (xs, ys) arrays per trajectory
        self._X = np.array([])  # mean x per trajectory
        self._Y = np.array([])  # mean y per trajectory

        self._x_min, self._x_max = 0.0, 1.0
        self._y_min, self._y_max = 0.0, 1.0

        # Drawing state
        self._draw_mode = self.MODE_RECT
        self._classify_mode = self.CLASSIFY_MEAN
        self._drawing = False
        self._drag_start = None
        self._current_preview = None
        self._shapes = []
        self._traj_pixmap_item = None
        self._roi_labels = None
        self._handle_items = []  # graphics items for resize handles

        # Select mode state
        self._selected_shape_idx = -1
        self._dragging_shape = False
        self._dragging_handle = None  # (shape_idx, handle_id) or None
        self._select_offset = QPointF(0, 0)

        self._color_default = QColor(180, 180, 180, 100)
        self._color_excluded = QColor(80, 80, 80, 60)

    def set_draw_mode(self, mode: str):
        self._draw_mode = mode
        self._selected_shape_idx = -1
        self._remove_handles()

    def set_classify_mode(self, mode: str):
        self._classify_mode = mode
        if self._shapes:
            self._classify_points()
            self._render_traj_pixmap()
            self.roi_changed.emit()

    def set_data(self, traj_indices, traj_points, mean_x, mean_y):
        """Set trajectory data for plotting.

        Args:
            traj_indices: array of trajectory IDs
            traj_points: list of (xs, ys) tuples per trajectory
            mean_x: mean x per trajectory
            mean_y: mean y per trajectory
        """
        self._traj_indices = np.array(traj_indices)
        self._traj_points = traj_points
        self._X = np.array(mean_x, dtype=float)
        self._Y = np.array(mean_y, dtype=float)

        if len(self._X) > 0:
            # Compute bounds from all trajectory points
            all_x = np.concatenate([pts[0] for pts in traj_points]) if traj_points else self._X
            all_y = np.concatenate([pts[1] for pts in traj_points]) if traj_points else self._Y
            pad_x = max((all_x.max() - all_x.min()) * 0.05, 1.0)
            pad_y = max((all_y.max() - all_y.min()) * 0.05, 1.0)
            self._x_min = float(all_x.min() - pad_x)
            self._x_max = float(all_x.max() + pad_x)
            self._y_min = float(all_y.min() - pad_y)
            self._y_max = float(all_y.max() + pad_y)
            # Enforce equal range on both axes, centered on each axis's own midpoint
            x_center = (self._x_min + self._x_max) / 2
            y_center = (self._y_min + self._y_max) / 2
            half_range = max(self._x_max - self._x_min, self._y_max - self._y_min) / 2
            self._x_min, self._x_max = x_center - half_range, x_center + half_range
            self._y_min, self._y_max = y_center - half_range, y_center + half_range
        self._roi_labels = None
        self._clear_shapes()
        self._draw_plot()

    # --- Coordinate conversions (data ↔ scene) ---
    def _x_to_sx(self, xval):
        return self._MARGIN_LEFT + (xval - self._x_min) / (self._x_max - self._x_min) * self._PLOT_W

    def _y_to_sy(self, yval):
        return self._MARGIN_TOP + (yval - self._y_min) / (self._y_max - self._y_min) * self._PLOT_H

    def _sx_to_x(self, sx):
        return self._x_min + (sx - self._MARGIN_LEFT) / self._PLOT_W * (self._x_max - self._x_min)

    def _sy_to_y(self, sy):
        return self._y_min + (sy - self._MARGIN_TOP) / self._PLOT_H * (self._y_max - self._y_min)

    # --- Plot rendering ---
    def _draw_plot(self):
        self._scene.clear()
        self._traj_pixmap_item = None
        self._handle_items = []

        total_w = self._MARGIN_LEFT + self._PLOT_W + self._MARGIN_RIGHT
        total_h = self._MARGIN_TOP + self._PLOT_H + self._MARGIN_BOTTOM
        self._scene.setSceneRect(0, 0, total_w, total_h)

        pen_axis = QPen(QColor(150, 150, 150), 1.5)
        pen_grid = QPen(QColor(60, 60, 60), 0.5, Qt.PenStyle.DashLine)
        pen_text = QColor(180, 180, 180)
        scene_font = QFont()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        scene_font.setPointSize(18)  # scaled for 1000x800 scene

        self._scene.addRect(
            QRectF(self._MARGIN_LEFT, self._MARGIN_TOP, self._PLOT_W, self._PLOT_H),
            QPen(Qt.PenStyle.NoPen), QBrush(QColor(30, 30, 30))
        )

        # Use a single grid step for both axes to keep square cells
        x_range = self._x_max - self._x_min
        y_range = self._y_max - self._y_min
        grid_step = self._nice_step(max(x_range, y_range), 8)

        # Grid lines — X axis
        xv = math.ceil(self._x_min / grid_step) * grid_step
        while xv <= self._x_max:
            sx = self._x_to_sx(xv)
            if self._MARGIN_LEFT <= sx <= self._MARGIN_LEFT + self._PLOT_W:
                self._scene.addLine(sx, self._MARGIN_TOP, sx, self._MARGIN_TOP + self._PLOT_H, pen_grid)
                txt = self._scene.addSimpleText(f"{xv:.0f}", scene_font)
                txt.setBrush(pen_text)
                txt.setPos(sx - 20, self._MARGIN_TOP + self._PLOT_H + 8)
            xv += grid_step

        # Grid lines — Y axis
        yv = math.ceil(self._y_min / grid_step) * grid_step
        while yv <= self._y_max:
            sy = self._y_to_sy(yv)
            if self._MARGIN_TOP <= sy <= self._MARGIN_TOP + self._PLOT_H:
                self._scene.addLine(self._MARGIN_LEFT, sy, self._MARGIN_LEFT + self._PLOT_W, sy, pen_grid)
                txt = self._scene.addSimpleText(f"{yv:.0f}", scene_font)
                txt.setBrush(pen_text)
                txt.setPos(self._MARGIN_LEFT - 80, sy - 12)
            yv += grid_step

        self._scene.addLine(
            self._MARGIN_LEFT, self._MARGIN_TOP + self._PLOT_H,
            self._MARGIN_LEFT + self._PLOT_W, self._MARGIN_TOP + self._PLOT_H, pen_axis)
        self._scene.addLine(
            self._MARGIN_LEFT, self._MARGIN_TOP,
            self._MARGIN_LEFT, self._MARGIN_TOP + self._PLOT_H, pen_axis)

        x_label = self._scene.addSimpleText("X (pixels)", scene_font)
        x_label.setBrush(pen_text)
        x_label.setPos(self._MARGIN_LEFT + self._PLOT_W / 2 - 55, self._MARGIN_TOP + self._PLOT_H + 45)
        y_label = self._scene.addSimpleText("Y (pixels)", scene_font)
        y_label.setBrush(pen_text)
        y_label.setRotation(-90)
        y_label.setPos(20, self._MARGIN_TOP + self._PLOT_H / 2 + 40)

        self._render_traj_pixmap()

        for shape in self._shapes:
            shape['path_item'] = self._draw_shape_on_scene(shape)

        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    @staticmethod
    def _nice_step(data_range, max_ticks):
        raw = data_range / max(max_ticks, 1)
        magnitude = 10 ** math.floor(math.log10(max(raw, 1e-12)))
        residual = raw / magnitude
        if residual <= 1.5:
            return magnitude
        elif residual <= 3.0:
            return 2 * magnitude
        elif residual <= 7.0:
            return 5 * magnitude
        else:
            return 10 * magnitude

    @staticmethod  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    def _rotate_point(px, py, cx, cy, angle_deg):
        """Rotate point (px,py) around (cx,cy) by angle_deg degrees."""
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        dx, dy = px - cx, py - cy
        return cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy

    def _render_traj_pixmap(self):
        """Render trajectories as connected lines onto a single QPixmap."""
        if self._traj_pixmap_item and self._traj_pixmap_item.scene():
            self._scene.removeItem(self._traj_pixmap_item)
            self._traj_pixmap_item = None
        rect = self._scene.sceneRect()
        w, h = int(rect.width()), int(rect.height())
        if w <= 0 or h <= 0:
            return
        pix = QPixmap(w, h)
        pix.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for i, (xs, ys) in enumerate(self._traj_points):
            if self._roi_labels is not None:
                lbl = int(self._roi_labels[i])
                if lbl < 0:
                    color = self._color_excluded
                else:
                    color = self._ROI_COLORS[lbl % len(self._ROI_COLORS)]
            else:
                color = self._color_default
            pen = QPen(color, 1.0)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Convert to scene coords and draw connected line
            sxs = self._MARGIN_LEFT + (xs - self._x_min) / (self._x_max - self._x_min) * self._PLOT_W
            sys_ = self._MARGIN_TOP + (ys - self._y_min) / (self._y_max - self._y_min) * self._PLOT_H
            for j in range(len(sxs) - 1):
                painter.drawLine(QPointF(sxs[j], sys_[j]), QPointF(sxs[j + 1], sys_[j + 1]))

        painter.end()
        self._traj_pixmap_item = self._scene.addPixmap(pix)
        self._traj_pixmap_item.setZValue(1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._scene.sceneRect().width() > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # --- Mouse interaction ---
    def _clamp_to_plot(self, pos):
        x = max(self._MARGIN_LEFT, min(pos.x(), self._MARGIN_LEFT + self._PLOT_W))
        y = max(self._MARGIN_TOP, min(pos.y(), self._MARGIN_TOP + self._PLOT_H))
        return QPointF(x, y)

    def keyPressEvent(self, event):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        key = event.key()
        # Delete selected shape
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if 0 <= self._selected_shape_idx < len(self._shapes):
                s = self._shapes.pop(self._selected_shape_idx)
                if s['path_item'] and s['path_item'].scene():
                    self._scene.removeItem(s['path_item'])
                self._remove_handles()
                self._selected_shape_idx = -1
                self._classify_points()
                self._render_traj_pixmap()
                self.roi_changed.emit()
                return
        # Mode shortcuts
        mode_map = {Qt.Key.Key_R: self.MODE_RECT, Qt.Key.Key_E: self.MODE_ELLIPSE,
                    Qt.Key.Key_L: self.MODE_LINE, Qt.Key.Key_S: self.MODE_SELECT}
        if key in mode_map:
            self.mode_requested.emit(mode_map[key])
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        if event.button() == Qt.MouseButton.RightButton and len(self._shapes) > 0:
            shape = self._shapes.pop()
            if shape['path_item'] and shape['path_item'].scene():
                self._scene.removeItem(shape['path_item'])
            self._remove_handles()
            self._selected_shape_idx = -1
            self._classify_points()
            self._render_traj_pixmap()
            self.roi_changed.emit()
            super().mousePressEvent(event)
            return

        if event.button() != Qt.MouseButton.LeftButton or len(self._X) == 0:
            super().mousePressEvent(event)
            return

        pos = self._clamp_to_plot(self.mapToScene(event.pos()))

        # Always check handles and shapes first, regardless of draw mode
        handle_hit = self._hit_test_handle(pos)
        if handle_hit is not None:
            self._dragging_handle = handle_hit
            self._drag_start = pos
            super().mousePressEvent(event)
            return

        shape_idx = self._hit_test_shape(pos)
        if shape_idx >= 0:
            self._selected_shape_idx = shape_idx
            self._dragging_shape = True
            self._drag_start = pos
            self._show_handles(shape_idx)
            super().mousePressEvent(event)
            return

        # Clicked on empty space — deselect and start drawing (unless Select mode)
        self._selected_shape_idx = -1
        self._dragging_shape = False
        self._remove_handles()
        if self._draw_mode != self.MODE_SELECT:
            self._drawing = True
            self._drag_start = pos
        super().mousePressEvent(event)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def mouseMoveEvent(self, event):
        pos = self._clamp_to_plot(self.mapToScene(event.pos()))

        if self._dragging_handle is not None and self._drag_start is not None:
            self._resize_shape(pos)
        elif self._dragging_shape and self._drag_start is not None:
            self._move_shape(pos)
        elif self._drawing and self._drag_start is not None:
            self._draw_preview(self._drag_start, pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._dragging_shape or self._dragging_handle is not None:
            self._dragging_shape = False
            self._dragging_handle = None
            self._drag_start = None
            self._sync_shape_data_coords()
            self._classify_points()
            self._render_traj_pixmap()
            if self._selected_shape_idx >= 0:
                self._show_handles(self._selected_shape_idx)
            self.roi_changed.emit()
        elif self._drawing:
            self._drawing = False
            if self._drag_start is not None:
                end = self._clamp_to_plot(self.mapToScene(event.pos()))
                if self._current_preview and self._current_preview.scene():
                    self._scene.removeItem(self._current_preview)
                    self._current_preview = None
                dx = abs(end.x() - self._drag_start.x())
                dy = abs(end.y() - self._drag_start.y())
                if dx > 2 or dy > 2:
                    shape = self._create_shape(self._drag_start, end)
                    shape['path_item'] = self._draw_shape_on_scene(shape)
                    self._shapes.append(shape)
                    self._classify_points()
                    self._render_traj_pixmap()
                    self.roi_changed.emit()
                self._drag_start = None
        super().mouseReleaseEvent(event)

    # --- Select mode: hit testing, move, resize ---
    def _hit_test_shape(self, pos):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Return index of shape under pos, or -1."""
        threshold = 8.0
        rp = self._rotate_point
        for i in range(len(self._shapes) - 1, -1, -1):
            s = self._shapes[i]
            if s['type'] == 'rect':
                # Unrotate pos into rect's local frame
                cx = (s['x1'] + s['x2']) / 2
                cy = (s['y1'] + s['y2']) / 2
                lx, ly = rp(pos.x(), pos.y(), cx, cy, -s.get('angle', 0.0))
                r = QRectF(s['x1'] - threshold, s['y1'] - threshold,
                           s['x2'] - s['x1'] + 2 * threshold, s['y2'] - s['y1'] + 2 * threshold)
                if r.contains(QPointF(lx, ly)):
                    return i
            elif s['type'] == 'ellipse':
                # Unrotate pos into ellipse's local frame
                lx, ly = rp(pos.x(), pos.y(), s['cx'], s['cy'], -s.get('angle', 0.0))
                dx = (lx - s['cx']) / max(s['rx'] + threshold, 1)
                dy = (ly - s['cy']) / max(s['ry'] + threshold, 1)
                if dx * dx + dy * dy <= 1.0:
                    return i
            elif s['type'] == 'line':
                # Distance from point to line segment
                lx, ly = s['p2'].x() - s['p1'].x(), s['p2'].y() - s['p1'].y()
                l2 = lx * lx + ly * ly
                if l2 < 1e-6:
                    continue
                t = max(0, min(1, ((pos.x() - s['p1'].x()) * lx + (pos.y() - s['p1'].y()) * ly) / l2))
                proj_x = s['p1'].x() + t * lx
                proj_y = s['p1'].y() + t * ly
                dist = math.sqrt((pos.x() - proj_x) ** 2 + (pos.y() - proj_y) ** 2)
                if dist <= threshold:
                    return i
        return -1

    _ROT_HANDLE_OFFSET = 45  # scene pixels from shape edge to rotation handle  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _get_handles(self, shape_idx):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Return list of (handle_id, QPointF) for a shape."""
        s = self._shapes[shape_idx]
        rp = self._rotate_point
        if s['type'] == 'rect':
            cx = (s['x1'] + s['x2']) / 2
            cy = (s['y1'] + s['y2']) / 2
            hw = (s['x2'] - s['x1']) / 2
            hh = (s['y2'] - s['y1']) / 2
            angle = s.get('angle', 0.0)
            corners = [('tl', -hw, -hh), ('tr', hw, -hh),
                       ('bl', -hw, hh), ('br', hw, hh)]
            handles = [(hid, QPointF(*rp(cx + lx, cy + ly, cx, cy, angle)))
                       for hid, lx, ly in corners]
            # Rotation handle above top-center
            rot_x, rot_y = rp(cx, cy - hh - self._ROT_HANDLE_OFFSET, cx, cy, angle)
            handles.append(('rot', QPointF(rot_x, rot_y)))
            return handles
        elif s['type'] == 'ellipse':
            cx, cy = s['cx'], s['cy']
            rx, ry = s['rx'], s['ry']
            angle = s.get('angle', 0.0)
            corners = [('tl', -rx, -ry), ('tr', rx, -ry),
                       ('bl', -rx, ry), ('br', rx, ry)]
            handles = [(hid, QPointF(*rp(cx + lx, cy + ly, cx, cy, angle)))
                       for hid, lx, ly in corners]
            rot_x, rot_y = rp(cx, cy - ry - self._ROT_HANDLE_OFFSET, cx, cy, angle)
            handles.append(('rot', QPointF(rot_x, rot_y)))
            return handles
        elif s['type'] == 'line':
            return [('p1', s['p1']), ('p2', s['p2'])]
        return []  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _hit_test_handle(self, pos):
        """Return (shape_idx, handle_id) if pos is on a handle, else None."""
        if self._selected_shape_idx < 0:
            return None
        hs = self._HANDLE_SIZE + 2
        for hid, hpos in self._get_handles(self._selected_shape_idx):
            if abs(pos.x() - hpos.x()) <= hs and abs(pos.y() - hpos.y()) <= hs:
                return (self._selected_shape_idx, hid)
        return None

    def _show_handles(self, shape_idx):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Draw resize handles and rotation handle for the selected shape."""
        self._remove_handles()
        hs = self._HANDLE_SIZE
        s = self._shapes[shape_idx]
        # Compute top-center (rotated) for the connecting line
        top_center = None
        if s['type'] == 'rect':
            cx = (s['x1'] + s['x2']) / 2
            cy = (s['y1'] + s['y2']) / 2
            hh = (s['y2'] - s['y1']) / 2
            top_center = QPointF(*self._rotate_point(cx, cy - hh, cx, cy, s.get('angle', 0.0)))
        elif s['type'] == 'ellipse':
            cx, cy, ry = s['cx'], s['cy'], s['ry']
            top_center = QPointF(*self._rotate_point(cx, cy - ry, cx, cy, s.get('angle', 0.0)))

        for hid, hpos in self._get_handles(shape_idx):
            if hid == 'rot':
                # Connecting line from top-center to rotation handle
                if top_center is not None:
                    line_item = self._scene.addLine(
                        top_center.x(), top_center.y(), hpos.x(), hpos.y(),
                        QPen(QColor(0, 220, 220, 140), 1.0, Qt.PenStyle.DashLine))
                    line_item.setZValue(19)
                    self._handle_items.append(line_item)
                # Cyan circle for rotation handle
                item = self._scene.addEllipse(
                    QRectF(hpos.x() - hs / 2, hpos.y() - hs / 2, hs, hs),
                    QPen(QColor(255, 255, 255, 220), 1.0),
                    QBrush(QColor(0, 220, 220, 200)))
            else:
                # Yellow square for resize handles
                item = self._scene.addRect(
                    QRectF(hpos.x() - hs / 2, hpos.y() - hs / 2, hs, hs),
                    QPen(QColor(255, 255, 255, 220), 1.0),
                    QBrush(QColor(255, 255, 0, 180)))
            item.setZValue(20)
            self._handle_items.append(item)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _remove_handles(self):
        for item in self._handle_items:
            if item.scene():
                self._scene.removeItem(item)
        self._handle_items = []

    def _move_shape(self, pos):
        """Translate the selected shape by drag delta."""
        if self._selected_shape_idx < 0 or self._drag_start is None:
            return
        dx = pos.x() - self._drag_start.x()
        dy = pos.y() - self._drag_start.y()
        self._drag_start = pos
        s = self._shapes[self._selected_shape_idx]

        if s['type'] == 'rect':
            s['x1'] += dx; s['y1'] += dy; s['x2'] += dx; s['y2'] += dy
        elif s['type'] == 'ellipse':
            s['cx'] += dx; s['cy'] += dy
        elif s['type'] == 'line':
            s['p1'] = QPointF(s['p1'].x() + dx, s['p1'].y() + dy)
            s['p2'] = QPointF(s['p2'].x() + dx, s['p2'].y() + dy)

        # Redraw this shape
        if s['path_item'] and s['path_item'].scene():
            self._scene.removeItem(s['path_item'])
        s['path_item'] = self._draw_shape_on_scene(s)
        self._show_handles(self._selected_shape_idx)

    def _resize_shape(self, pos):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Resize or rotate the selected shape by dragging a handle."""
        if self._dragging_handle is None:
            return
        shape_idx, hid = self._dragging_handle
        s = self._shapes[shape_idx]

        if hid == 'rot':
            # Rotation: compute angle from center to mouse
            if s['type'] == 'rect':
                cx = (s['x1'] + s['x2']) / 2
                cy = (s['y1'] + s['y2']) / 2
            else:  # ellipse
                cx, cy = s['cx'], s['cy']
            # atan2 with "up" as reference direction (0 degrees)
            angle = math.degrees(math.atan2(pos.x() - cx, -(pos.y() - cy)))
            # Shift key: snap to 15° increments  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                angle = round(angle / 15.0) * 15.0
            s['angle'] = angle
        else:
            pos = self._clamp_to_plot(pos)
            if s['type'] == 'rect':
                # Symmetric resize around center in local (unrotated) frame
                cx = (s['x1'] + s['x2']) / 2
                cy = (s['y1'] + s['y2']) / 2
                lx, ly = self._rotate_point(pos.x(), pos.y(), cx, cy, -s.get('angle', 0.0))
                hw = max(abs(lx - cx), 2.0)
                hh = max(abs(ly - cy), 2.0)
                s['x1'] = cx - hw; s['x2'] = cx + hw
                s['y1'] = cy - hh; s['y2'] = cy + hh
            elif s['type'] == 'ellipse':
                # Symmetric resize around center in local frame
                lx, ly = self._rotate_point(pos.x(), pos.y(), s['cx'], s['cy'],
                                            -s.get('angle', 0.0))
                s['rx'] = max(abs(lx - s['cx']), 2.0)
                s['ry'] = max(abs(ly - s['cy']), 2.0)
            elif s['type'] == 'line':
                if hid == 'p1': s['p1'] = pos
                elif hid == 'p2': s['p2'] = pos

        if s['path_item'] and s['path_item'].scene():
            self._scene.removeItem(s['path_item'])
        s['path_item'] = self._draw_shape_on_scene(s)
        self._show_handles(shape_idx)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _sync_shape_data_coords(self):
        """Update data coordinate fields for all shapes from current scene coords."""
        for s in self._shapes:
            if s['type'] == 'line':
                s['p1_data'] = [self._sx_to_x(s['p1'].x()), self._sy_to_y(s['p1'].y())]
                s['p2_data'] = [self._sx_to_x(s['p2'].x()), self._sy_to_y(s['p2'].y())]
            elif s['type'] == 'ellipse':
                s['cx_data'] = self._sx_to_x(s['cx'])
                s['cy_data'] = self._sy_to_y(s['cy'])
                s['rx_data'] = s['rx'] / self._PLOT_W * (self._x_max - self._x_min)
                s['ry_data'] = s['ry'] / self._PLOT_H * (self._y_max - self._y_min)
            else:  # rect
                s['x1_data'] = self._sx_to_x(s['x1'])
                s['y1_data'] = self._sy_to_y(s['y1'])
                s['x2_data'] = self._sx_to_x(s['x2'])
                s['y2_data'] = self._sy_to_y(s['y2'])

    # --- Shape creation and rendering ---
    def _create_shape(self, start, end):
        if self._draw_mode == self.MODE_LINE:
            p1, p2 = self._extend_line_to_edges(start, end)
            return {'type': 'line', 'p1': p1, 'p2': p2, 'path_item': None,
                    'p1_data': [self._sx_to_x(p1.x()), self._sy_to_y(p1.y())],
                    'p2_data': [self._sx_to_x(p2.x()), self._sy_to_y(p2.y())]}
        elif self._draw_mode == self.MODE_ELLIPSE:
            cx = (start.x() + end.x()) / 2
            cy = (start.y() + end.y()) / 2
            rx = abs(end.x() - start.x()) / 2
            ry = abs(end.y() - start.y()) / 2
            return {'type': 'ellipse', 'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry,
                    'angle': 0.0, 'path_item': None,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                    'cx_data': self._sx_to_x(cx), 'cy_data': self._sy_to_y(cy),
                    'rx_data': rx / self._PLOT_W * (self._x_max - self._x_min),
                    'ry_data': ry / self._PLOT_H * (self._y_max - self._y_min)}
        else:  # Rectangle
            x1, x2 = min(start.x(), end.x()), max(start.x(), end.x())
            y1, y2 = min(start.y(), end.y()), max(start.y(), end.y())
            return {'type': 'rect', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'angle': 0.0, 'path_item': None,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                    'x1_data': self._sx_to_x(x1), 'y1_data': self._sy_to_y(y1),
                    'x2_data': self._sx_to_x(x2), 'y2_data': self._sy_to_y(y2)}

    def _extend_line_to_edges(self, start, end):
        pl = self._MARGIN_LEFT
        pr = self._MARGIN_LEFT + self._PLOT_W
        pt = self._MARGIN_TOP
        pb = self._MARGIN_TOP + self._PLOT_H
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return start, end
        dx /= length
        dy /= length

        def _intersect_edge(px, py, dirx, diry):
            candidates = []
            if abs(dirx) > 1e-9:
                for edge_x in (pl, pr):
                    t = (edge_x - px) / dirx
                    if t > 1e-6:
                        yy = py + t * diry
                        if pt - 0.5 <= yy <= pb + 0.5:
                            candidates.append((t, QPointF(edge_x, max(pt, min(yy, pb)))))
            if abs(diry) > 1e-9:
                for edge_y in (pt, pb):
                    t = (edge_y - py) / diry
                    if t > 1e-6:
                        xx = px + t * dirx
                        if pl - 0.5 <= xx <= pr + 0.5:
                            candidates.append((t, QPointF(max(pl, min(xx, pr)), edge_y)))
            if candidates:
                candidates.sort(key=lambda c: c[0])
                return candidates[0][1]
            return QPointF(px, py)

        p1 = _intersect_edge(start.x(), start.y(), -dx, -dy)
        p2 = _intersect_edge(end.x(), end.y(), dx, dy)
        return p1, p2

    def _draw_preview(self, start, end):
        if self._current_preview and self._current_preview.scene():
            self._scene.removeItem(self._current_preview)
            self._current_preview = None
        pen = QPen(QColor(255, 255, 0, 180), 2.0, Qt.PenStyle.DashLine)
        if self._draw_mode == self.MODE_LINE:
            p1, p2 = self._extend_line_to_edges(start, end)
            self._current_preview = self._scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen)
        elif self._draw_mode == self.MODE_ELLIPSE:
            rx = abs(end.x() - start.x()) / 2
            ry = abs(end.y() - start.y()) / 2
            cx = (start.x() + end.x()) / 2
            cy = (start.y() + end.y()) / 2
            path = QPainterPath()
            path.addEllipse(QPointF(cx, cy), rx, ry)
            self._current_preview = self._scene.addPath(path, pen)
        else:
            x1, x2 = min(start.x(), end.x()), max(start.x(), end.x())
            y1, y2 = min(start.y(), end.y()), max(start.y(), end.y())
            self._current_preview = self._scene.addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)
        if self._current_preview:
            self._current_preview.setZValue(10)

    def _draw_shape_on_scene(self, shape):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        pen = QPen(QColor(255, 255, 0, 220), 2.0)
        if shape['type'] == 'line':
            item = self._scene.addLine(
                shape['p1'].x(), shape['p1'].y(), shape['p2'].x(), shape['p2'].y(), pen)
        elif shape['type'] == 'ellipse':
            angle = shape.get('angle', 0.0)
            path = QPainterPath()
            path.addEllipse(QPointF(0, 0), shape['rx'], shape['ry'])
            t = QTransform()
            t.translate(shape['cx'], shape['cy'])
            t.rotate(angle)
            item = self._scene.addPath(t.map(path), pen)
        else:  # rect
            angle = shape.get('angle', 0.0)
            cx = (shape['x1'] + shape['x2']) / 2
            cy = (shape['y1'] + shape['y2']) / 2
            hw = (shape['x2'] - shape['x1']) / 2
            hh = (shape['y2'] - shape['y1']) / 2
            path = QPainterPath()
            path.addRect(QRectF(-hw, -hh, 2 * hw, 2 * hh))
            t = QTransform()
            t.translate(cx, cy)
            t.rotate(angle)
            item = self._scene.addPath(t.map(path), pen)
        item.setZValue(5)
        return item  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    # --- Classification (flood fill) ---
    def _classify_points(self):
        """Classify trajectories into ROIs using rasterized flood fill."""
        from scipy.ndimage import label as ndimage_label, distance_transform_edt

        if not self._shapes:
            self._roi_labels = None
            return

        grid_w = int(self._PLOT_W)
        grid_h = int(self._PLOT_H)
        wall = np.zeros((grid_h, grid_w), dtype=bool)

        for shape in self._shapes:
            if shape['type'] == 'line':
                self._rasterize_line(wall, shape['p1'], shape['p2'])
            elif shape['type'] == 'ellipse':
                self._rasterize_ellipse(wall, shape['cx'], shape['cy'],
                                        shape['rx'], shape['ry'],
                                        shape.get('angle', 0.0))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            else:
                self._rasterize_rect(wall, shape['x1'], shape['y1'],
                                     shape['x2'], shape['y2'],
                                     shape.get('angle', 0.0))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

        grid, n_regions = ndimage_label(~wall)

        # Precompute distance transform for wall-pixel fix
        _, nearest_indices = distance_transform_edt(wall, return_distances=True, return_indices=True)
        nr, nc = nearest_indices[0], nearest_indices[1]

        def _label_points(xs, ys):
            """Get grid labels for an array of data-space points."""
            px = (xs - self._x_min) / (self._x_max - self._x_min) * self._PLOT_W
            py = (ys - self._y_min) / (self._y_max - self._y_min) * self._PLOT_H
            gx = np.clip(np.round(px).astype(int), 0, grid_w - 1)
            gy = np.clip(np.round(py).astype(int), 0, grid_h - 1)
            lbls = grid[gy, gx]
            on_wall = lbls == 0
            if np.any(on_wall):
                lbls[on_wall] = grid[nr[gy[on_wall], gx[on_wall]], nc[gy[on_wall], gx[on_wall]]]
            return np.maximum(0, lbls - 1)

        if self._classify_mode == self.CLASSIFY_STRICT:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            # Strict: all points of trajectory must be in same region
            self._roi_labels = np.full(len(self._X), -1, dtype=int)
            for i, (xs, ys) in enumerate(self._traj_points):
                if len(xs) == 0:
                    continue
                pt_labels = _label_points(np.array(xs), np.array(ys))
                # Early-exit: check min==max instead of np.unique (faster)
                lmin, lmax = int(pt_labels.min()), int(pt_labels.max())
                if lmin == lmax:
                    self._roi_labels[i] = lmin
                # else: stays -1 (excluded)
        else:
            # Mean position mode
            self._roi_labels = _label_points(self._X, self._Y).astype(int)

        # Reorder by ascending mean X (only for assigned labels >= 0)
        valid_mask = self._roi_labels >= 0
        if np.any(valid_mask):
            unique_labels = np.unique(self._roi_labels[valid_mask])
            if len(unique_labels) > 1:
                means = np.array([np.mean(self._X[(self._roi_labels == lbl) & valid_mask])
                                  for lbl in unique_labels])
                order = np.argsort(means)
                remap = np.zeros(int(unique_labels.max()) + 1, dtype=int)
                for new_lbl, idx in enumerate(order):
                    remap[unique_labels[idx]] = new_lbl
                assigned = self._roi_labels >= 0
                self._roi_labels[assigned] = remap[self._roi_labels[assigned]]

    def _rasterize_line(self, wall, p1, p2):
        x0 = p1.x() - self._MARGIN_LEFT
        y0 = p1.y() - self._MARGIN_TOP
        x1 = p2.x() - self._MARGIN_LEFT
        y1 = p2.y() - self._MARGIN_TOP
        dx, dy = x1 - x0, y1 - y0
        steps = max(int(max(abs(dx), abs(dy))), 1)
        xs = np.round(np.linspace(x0, x1, steps + 1)).astype(int)
        ys = np.round(np.linspace(y0, y1, steps + 1)).astype(int)
        gh, gw = wall.shape
        for di, dj in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            rs = np.clip(ys + di, 0, gh - 1)
            cs = np.clip(xs + dj, 0, gw - 1)
            wall[rs, cs] = True

    def _rasterize_ellipse(self, wall, cx, cy, rx, ry, angle=0.0):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        cx0 = cx - self._MARGIN_LEFT
        cy0 = cy - self._MARGIN_TOP
        gh, gw = wall.shape
        n_pts = max(int(2 * math.pi * max(rx, ry)), 200)
        angles = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
        local_xs = rx * np.cos(angles)
        local_ys = ry * np.sin(angles)
        # Apply rotation
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        xs = np.round(cx0 + local_xs * cos_a - local_ys * sin_a).astype(int)
        ys = np.round(cy0 + local_xs * sin_a + local_ys * cos_a).astype(int)
        for di, dj in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            rs = np.clip(ys + di, 0, gh - 1)
            cs = np.clip(xs + dj, 0, gw - 1)
            wall[rs, cs] = True  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _rasterize_rect(self, wall, x1, y1, x2, y2, angle=0.0):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        hw = (x2 - x1) / 2
        hh = (y2 - y1) / 2
        rp = self._rotate_point
        # Compute rotated corners
        c0 = QPointF(*rp(cx - hw, cy - hh, cx, cy, angle))
        c1 = QPointF(*rp(cx + hw, cy - hh, cx, cy, angle))
        c2 = QPointF(*rp(cx + hw, cy + hh, cx, cy, angle))
        c3 = QPointF(*rp(cx - hw, cy + hh, cx, cy, angle))
        for p1, p2 in [(c0, c1), (c1, c2), (c2, c3), (c3, c0)]:
            self._rasterize_line(wall, p1, p2)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    # --- Public API ---
    def _clear_shapes(self):
        for shape in self._shapes:
            if shape['path_item'] and shape['path_item'].scene():
                self._scene.removeItem(shape['path_item'])
        self._shapes = []
        self._remove_handles()
        self._selected_shape_idx = -1
        if self._current_preview and self._current_preview.scene():
            self._scene.removeItem(self._current_preview)
            self._current_preview = None
        self._roi_labels = None

    def clear_roi(self):
        self._clear_shapes()
        self._render_traj_pixmap()
        self.roi_changed.emit()

    def get_roi_data(self):
        result = {
            'X': self._X,
            'Y': self._Y,
            'traj_indices': self._traj_indices,
            'labels': self._roi_labels,
        }
        if self._roi_labels is not None:
            valid = self._roi_labels >= 0
            unique_labels = sorted(set(self._roi_labels[valid].tolist())) if np.any(valid) else []
            result['n_rois'] = len(unique_labels)
            result['rois'] = {
                r: np.where(self._roi_labels == r)[0] for r in unique_labels
            }
            n_excluded = int(np.sum(self._roi_labels < 0))
            result['n_excluded'] = n_excluded
        else:
            result['n_rois'] = 1
            result['rois'] = {0: np.arange(len(self._X))}
            result['n_excluded'] = 0
        return result

    def get_shapes_data(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        result = []
        for shape in self._shapes:
            if shape['type'] == 'line':
                result.append({'type': 'line',
                               'p1': shape['p1_data'], 'p2': shape['p2_data']})
            elif shape['type'] == 'ellipse':
                result.append({'type': 'ellipse',
                               'cx': shape['cx_data'], 'cy': shape['cy_data'],
                               'rx': shape['rx_data'], 'ry': shape['ry_data'],
                               'angle': shape.get('angle', 0.0)})
            else:
                result.append({'type': 'rect',
                               'x1': shape['x1_data'], 'y1': shape['y1_data'],
                               'x2': shape['x2_data'], 'y2': shape['y2_data'],
                               'angle': shape.get('angle', 0.0)})
        return result  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def set_shapes_data(self, shapes_data):
        self._clear_shapes()
        for sd in shapes_data:
            if sd['type'] == 'line':
                p1 = QPointF(self._x_to_sx(sd['p1'][0]), self._y_to_sy(sd['p1'][1]))
                p2 = QPointF(self._x_to_sx(sd['p2'][0]), self._y_to_sy(sd['p2'][1]))
                shape = {'type': 'line', 'p1': p1, 'p2': p2, 'path_item': None,
                         'p1_data': sd['p1'], 'p2_data': sd['p2']}
            elif sd['type'] == 'ellipse':
                cx = self._x_to_sx(sd['cx'])
                cy = self._y_to_sy(sd['cy'])
                rx = sd['rx'] / (self._x_max - self._x_min) * self._PLOT_W
                ry = sd['ry'] / (self._y_max - self._y_min) * self._PLOT_H
                shape = {'type': 'ellipse', 'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry,
                         'angle': sd.get('angle', 0.0), 'path_item': None,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                         'cx_data': sd['cx'], 'cy_data': sd['cy'],
                         'rx_data': sd['rx'], 'ry_data': sd['ry']}
            else:
                x1 = self._x_to_sx(sd['x1'])
                y1 = self._y_to_sy(sd['y1'])
                x2 = self._x_to_sx(sd['x2'])
                y2 = self._y_to_sy(sd['y2'])
                shape = {'type': 'rect', 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                         'angle': sd.get('angle', 0.0), 'path_item': None,  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                         'x1_data': sd['x1'], 'y1_data': sd['y1'],
                         'x2_data': sd['x2'], 'y2_data': sd['y2']}
            shape['path_item'] = self._draw_shape_on_scene(shape)
            self._shapes.append(shape)
        if self._shapes:
            self._classify_points()
            self._render_traj_pixmap()
            self.roi_changed.emit()
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26


# ---------------------------------------------------------------------------
def _open_url(url: str):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-22
    """Open a URL in the default browser, with WSL2 and native Windows support."""
    import subprocess, webbrowser, platform
    if platform.system() == 'Windows':
        os.startfile(url)
    elif 'microsoft' in os.uname().release.lower():
        subprocess.Popen(['cmd.exe', '/c', 'start', url.replace('&', '^&')])
    elif not QDesktopServices.openUrl(QUrl(url)):
        webbrowser.open(url)


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
# LaTeX-to-Unicode converter for chat display
# ---------------------------------------------------------------------------
def _latex_to_unicode(text: str) -> str:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-21
    """Convert common LaTeX math expressions to Unicode characters."""
    import re

    _GREEK = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\varepsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η',
        r'\theta': 'θ', r'\vartheta': 'ϑ', r'\iota': 'ι', r'\kappa': 'κ',
        r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ',
        r'\pi': 'π', r'\rho': 'ρ', r'\sigma': 'σ', r'\tau': 'τ',
        r'\upsilon': 'υ', r'\phi': 'φ', r'\varphi': 'φ', r'\chi': 'χ',
        r'\psi': 'ψ', r'\omega': 'ω',
        r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
        r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Phi': 'Φ',
        r'\Psi': 'Ψ', r'\Omega': 'Ω',
    }

    _SUP = str.maketrans('0123456789+-=()nixy', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣʸ')
    _SUB = str.maketrans('0123456789+-=()aeioruvx', '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵣᵤᵥₓ')

    _SYMBOLS = {
        r'\infty': '∞', r'\pm': '±', r'\mp': '∓', r'\times': '×',
        r'\cdot': '·', r'\div': '÷', r'\neq': '≠', r'\approx': '≈',
        r'\leq': '≤', r'\geq': '≥', r'\ll': '≪', r'\gg': '≫',
        r'\propto': '∝', r'\sim': '∼', r'\simeq': '≃',
        r'\partial': '∂', r'\nabla': '∇', r'\sqrt': '√',
        r'\sum': 'Σ', r'\prod': '∏', r'\int': '∫',
        r'\rightarrow': '→', r'\leftarrow': '←', r'\leftrightarrow': '↔',
        r'\Rightarrow': '⇒', r'\Leftarrow': '⇐',
        r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\supset': '⊃',
        r'\forall': '∀', r'\exists': '∃',
        r'\langle': '⟨', r'\rangle': '⟩',
        r'\hat': '', r'\bar': '', r'\vec': '', r'\dot': '',
        r'\mathrm': '', r'\text': '', r'\mathbf': '', r'\textbf': '',
        r'\left': '', r'\right': '',
    }

    def _convert_expr(expr: str) -> str:
        s = expr.strip()
        for latex, uni in sorted(_GREEK.items(), key=lambda x: -len(x[0])):
            s = s.replace(latex, uni)
        for latex, uni in sorted(_SYMBOLS.items(), key=lambda x: -len(x[0])):
            s = s.replace(latex, uni)
        s = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', s)
        def _sup_repl(m):
            base = m.group(1) if m.group(1) else ''
            return base + m.group(2).translate(_SUP)
        s = re.sub(r'(\w?)\^\{([^}]*)\}', _sup_repl, s)
        s = re.sub(r'(\w?)\^(\w)', lambda m: m.group(1) + m.group(2).translate(_SUP), s)
        def _sub_repl(m):
            base = m.group(1) if m.group(1) else ''
            return base + m.group(2).translate(_SUB)
        s = re.sub(r'(\w?)_\{([^}]*)\}', _sub_repl, s)
        s = re.sub(r'(\w?)_(\w)', lambda m: m.group(1) + m.group(2).translate(_SUB), s)
        s = s.replace('{', '').replace('}', '')
        return s

    text = re.sub(r'\$\$(.+?)\$\$', lambda m: _convert_expr(m.group(1)), text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$', lambda m: _convert_expr(m.group(1)), text)
    return text
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-21


# ---------------------------------------------------------------------------
# Gemini AI chat worker — sends queries to Gemini API in background
# ---------------------------------------------------------------------------
class GeminiChatWorker(QThread):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20 17:00
    reply_chunk = pyqtSignal(str)   # partial text (streamed)
    reply_done = pyqtSignal()       # generation finished
    error = pyqtSignal(str)         # error message

    _SYSTEM_PROMPT = (
        "You are the FreeTrace AI Assistant, embedded in the FreeTrace GUI — "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-23
        "a single-molecule tracking software that uses fractional Brownian motion (fBm) inference. "
        "FreeTrace performs: (1) particle localization via Gaussian fitting, "
        "(2) trajectory linking via graph-based optimization with a Cauchy-distribution cost function — "
        "for fBm, the ratio of consecutive displacement increments follows a Cauchy distribution "
        "parametrized by the Hurst exponent H, which FreeTrace uses to score trajectory links, "
        "(3) anomalous diffusion classification via a neural network that estimates H.\n\n"  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-23
        "Key parameters users can set in the GUI:\n"
        "- Window size: size of the sub-image for localization (default 7)\n"
        "- Threshold: intensity threshold for spot detection (default 50)\n"
        "- Min trajectory length: minimum number of points to keep a trajectory (default 7)\n"
        "- Graph depth: search depth for trajectory linking (default 3)\n"
        "- Jump threshold: maximum allowed jump distance between frames (default 2.8)\n"
        "- FBM mode: enable/disable fBm classification\n"
        "- GPU: enable/disable GPU acceleration for the neural network\n"
        "- Save video: export a video of the tracking result\n\n"
        "Answer questions about FreeTrace clearly and concisely. "
        "You can also answer general science questions — microscopy, biophysics, "
        "diffusion, image processing, statistics, and related topics. "
        "Keep answers very short and concise — 2-4 sentences maximum. "
        "Avoid bullet points and lists unless explicitly asked. "
        "Do not repeat the question. Go straight to the answer. "
        "Note: AI responses may not always be accurate — users should verify important results independently."
    )

    def __init__(self, api_key: str, message: str, history: list):
        super().__init__()
        self.api_key = api_key
        self.message = message
        self.history = history  # list of {"role": ..., "parts": ...}

    def run(self):
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)

            contents = list(self.history)
            contents.append({"role": "user", "parts": [{"text": self.message}]})

            response = client.models.generate_content_stream(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=self._SYSTEM_PROMPT,
                    max_output_tokens=512,
                ),
            )
            for chunk in response:
                if chunk.text:
                    self.reply_chunk.emit(chunk.text)
            self.reply_done.emit()
        except ImportError:
            self.error.emit(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai"
            )
        except Exception as e:
            self.error.emit(self._classify_error(str(e)))

    @staticmethod
    def _classify_error(msg: str) -> str:
        msg_lower = msg.lower()
        if "api_key_invalid" in msg_lower or "api key not valid" in msg_lower:
            return (
                "Invalid API key. Please check your Gemini API key and try again.\n"
                "You can get a free key at: https://aistudio.google.com/apikey"
            )
        if "permission_denied" in msg_lower or "403" in msg:
            return (
                "Access denied by Google. This may indicate a change in Google's "
                "Gemini API policy. The free tier for Gemini in FreeTrace may no "
                "longer be available. Please check: https://ai.google.dev/gemini-api/docs/pricing"
            )
        if "resource_exhausted" in msg_lower or "429" in msg or "quota" in msg_lower:
            if "per minute" in msg_lower or "rpm" in msg_lower:
                return (
                    "Too many requests per minute (limit: 15 RPM). "
                    "Please wait a moment and try again."
                )
            return (
                "Rate limit reached. The free tier allows 15 requests/minute "
                "and 1,000 requests/day. Please try again later."
            )
        if "not_found" in msg_lower or ("404" in msg and "model" in msg_lower):
            return (
                "The Gemini model used by FreeTrace is no longer available. "
                "This may be due to a change in Google's API. "
                "Please check for a FreeTrace update."
            )
        if any(k in msg_lower for k in ("timeout", "connection", "network", "dns", "resolve")):
            return "Network error. Please check your internet connection and try again."
        return f"Unexpected error: {msg}"
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20 17:00


# ---------------------------------------------------------------------------  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
# Vertical range slider — dual-handle slider for min/max selection
# ---------------------------------------------------------------------------
class VRangeSlider(QWidget):
    """Vertical slider with two draggable handles (low and high)."""
    range_changed = pyqtSignal(float, float)  # emits (low_frac, high_frac) in [0, 1]

    _HANDLE_H = 8   # handle half-height in pixels
    _TRACK_W = 6    # track width
    _MARGIN = 12    # top/bottom margin for handles

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(30)
        self.setMinimumHeight(100)
        self._low = 0.0   # fraction [0, 1]
        self._high = 1.0
        self._dragging = None  # 'low', 'high', or None

    def set_range(self, low, high):
        self._low = max(0.0, min(1.0, low))
        self._high = max(self._low, min(1.0, high))
        self.update()

    def low(self):
        return self._low

    def high(self):
        return self._high

    def _y_to_frac(self, y):
        """Convert pixel y to fraction (top=1, bottom=0)."""
        usable = self.height() - 2 * self._MARGIN
        if usable <= 0:
            return 0.5
        return max(0.0, min(1.0, 1.0 - (y - self._MARGIN) / usable))

    def _frac_to_y(self, frac):
        """Convert fraction to pixel y."""
        usable = self.height() - 2 * self._MARGIN
        return self._MARGIN + (1.0 - frac) * usable

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        cx = w // 2

        # Track background
        track_x = cx - self._TRACK_W // 2
        y_top = self._MARGIN
        y_bot = self.height() - self._MARGIN
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(60, 60, 60))
        painter.drawRoundedRect(QRectF(track_x, y_top, self._TRACK_W, y_bot - y_top), 3, 3)

        # Active range highlight
        y_high = self._frac_to_y(self._high)
        y_low = self._frac_to_y(self._low)
        painter.setBrush(QColor(80, 160, 255, 160))
        painter.drawRect(QRectF(track_x, y_high, self._TRACK_W, y_low - y_high))

        # Handles
        for frac, color in [(self._high, QColor(100, 200, 255)),
                            (self._low, QColor(100, 200, 255))]:
            y = self._frac_to_y(frac)
            painter.setBrush(color)
            painter.setPen(QPen(QColor(200, 200, 200), 1.0))
            painter.drawRoundedRect(QRectF(2, y - self._HANDLE_H / 2,
                                           w - 4, self._HANDLE_H), 3, 3)

        painter.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        y = event.pos().y()
        y_low = self._frac_to_y(self._low)
        y_high = self._frac_to_y(self._high)
        # Pick closest handle
        d_low = abs(y - y_low)
        d_high = abs(y - y_high)
        if d_low < d_high:
            self._dragging = 'low'
        else:
            self._dragging = 'high'

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        frac = self._y_to_frac(event.pos().y())
        if self._dragging == 'low':
            self._low = min(frac, self._high)
        else:
            self._high = max(frac, self._low)
        self.update()
        self.range_changed.emit(self._low, self._high)

    def mouseReleaseEvent(self, event):
        if self._dragging is not None:
            self._dragging = None
            self.range_changed.emit(self._low, self._high)
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26


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
        self._chat_worker = None  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20 17:00
        self._chat_history = []   # Gemini conversation history
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
        download_btn.clicked.connect(lambda: _open_url(url))  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
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

    def _build_freetrace_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20 17:00
        tab = QWidget()
        tab_layout = QHBoxLayout(tab)
        tab_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        tab_layout.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.addWidget(self._build_chat_tab())
        splitter.setSizes([330, 550, 350])
        return tab

    # ---- AI Chat tab ---------------------------------------------------
    # Predefined Q&A database (offline, no API needed)
    _PREDEFINED_QA = [  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-21
        {
            "keywords": ["window size", "window_size", "sub-image", "subimage",
                         "crop size", "roi size", "patch size", "box size", "spot size",
                         "psf", "point spread function"],
            "question": "What is the window size parameter?",
            "answer": (
                "Window size defines the side length (in pixels) of the square sub-image "
                "extracted around each detected spot for Gaussian fitting during localization. "
                "Default is 7. Increase it for larger PSFs (point spread functions); too small "
                "a window size increases the estimation bias. Decrease it for dense samples "
                "to avoid overlapping spots."
            ),
        },
        {
            "keywords": ["threshold", "intensity", "detection", "sensitivity",
                         "brightness", "signal", "noise", "snr", "dim", "bright"],
            "question": "What does the threshold parameter do?",
            "answer": (
                "Threshold is a multiplier applied to the frame-specific detection threshold "
                "computed from the background statistics (mean and standard deviation). "
                "Default is 1.0. Lower values (e.g. 0.5) increase detection sensitivity "
                "and detect dimmer particles but may increase false positives; higher values "
                "(e.g. 2.0) suppress weak detections and keep only bright spots."
            ),
        },
        {
            "keywords": ["min trajectory", "minimum trajectory", "min_traj", "trajectory length",
                         "short trajectory", "minimum length", "min length", "too short",
                         "minimum points", "few points", "cutoff"],
            "question": "What is the minimum trajectory length?",
            "answer": (
                "Minimum trajectory length (cutoff) sets the minimum number of frames "
                "a particle must be tracked to be kept as a valid trajectory. Default is 3. "
                "Increase this for cleaner results; decrease it if particles are short-lived."
            ),
        },
        {
            "keywords": ["graph depth", "search depth", "linking depth", "frame gap",
                         "blinking", "disappear", "reappear", "gap closing",
                         "missing frames", "skip frames"],
            "question": "What does graph depth control?",
            "answer": (
                "Graph depth controls how many frames ahead the tracking algorithm "
                "searches when linking particle positions into trajectories. Default is 3 "
                "(maximum 5). Higher values can recover trajectories through brief "
                "disappearances (blinking) but the computation cost grows factorially "
                "with graph depth in the worst case."
            ),
        },
        {
            "keywords": ["jump threshold", "jump_threshold", "maximum jump", "max jump",
                         "displacement", "max displacement", "how far", "move between frames",
                         "step size", "max distance", "maximum distance", "speed limit"],
            "question": "What is the jump threshold?",
            "answer": (
                "Jump threshold sets the maximum allowed displacement (in pixels) "
                "between consecutive frames for a particle to be linked into a trajectory. "
                "Default is 0 (auto), which lets FreeTrace infer the value automatically "
                "using a Gaussian Mixture Model (minimum 5 pixels). Set a fixed value for "
                "fast-moving particles; lower it to avoid linking distinct particles that "
                "happen to be close."
            ),
        },
        {
            "keywords": ["fbm", "fractional brownian", "hurst", "anomalous diffusion",
                         "classification", "diffusion type", "subdiffusion", "superdiffusion",
                         "confined", "directed", "normal diffusion", "exponent",
                         "motion type", "diffusion mode"],
            "question": "What is FBM mode?",
            "answer": (
                "FBM (fractional Brownian motion) mode enables trajectory reconstruction "
                "that accounts for anomalous diffusion. When active, a neural network "
                "estimates the Hurst exponent H for each trajectory: H = 0.5 means normal "
                "diffusion, H < 0.5 indicates subdiffusion (anti-persistent motion), and "
                "H > 0.5 indicates superdiffusion (persistent motion). Without FBM mode, "
                "trajectories are reconstructed assuming classical Brownian motion."
            ),
        },
        {
            "keywords": ["onnx", "neural network", "model", "inference",
                         "speed up", "faster", "slow", "performance"],
            "question": "How does neural network inference work?",
            "answer": (
                "FreeTrace C++ uses ONNX Runtime for neural network inference when FBM "
                "mode is enabled. The pre-converted ONNX models estimate the Hurst exponent "
                "H and generalised diffusion coefficient K for each trajectory. No TensorFlow "
                "or GPU is required — inference runs on CPU via ONNX Runtime."
            ),
        },
        {
            "keywords": ["localization", "gaussian", "spot detection", "fitting",
                         "find particles", "detect spots", "detect particles",
                         "position", "sub-pixel", "precision"],
            "question": "How does localization work?",
            "answer": (
                "FreeTrace detects bright spots in each frame using intensity thresholding, "
                "then fits a 2D Gaussian function to each spot to determine its sub-pixel "
                "position."
            ),
        },
        {
            "keywords": ["tracking", "linking", "trajectory", "graph",
                         "connect", "assignment", "matching", "link particles",
                         "connect spots", "build trajectories"],
            "question": "How does trajectory linking work?",
            "answer": (
                "FreeTrace uses a graph-based algorithm to link localized positions across "
                "frames into trajectories. It builds a directed graph where edges connect "
                "candidate particle positions within the jump threshold, then enumerates "
                "paths and selects the lowest-cost trajectory using a Cauchy-distribution-based "
                "cost function. This handles appearing/disappearing particles and temporary gaps."
            ),
        },
        {
            "keywords": ["input", "file format", "tiff", "tif", "video format",
                         "nd2", "nikon", "open file", "load", "supported format",
                         "image stack", "what files"],
            "question": "What input formats are supported?",
            "answer": (
                "FreeTrace accepts TIFF image stacks (.tiff, .tif) and Nikon ND2 files (.nd2). "
                "Each frame in the stack is processed sequentially for localization and tracking."
            ),
        },
        {
            "keywords": ["output", "results", "csv", "export", "save results",
                         "output files", "what output", "data format",
                         "download results", "where are results"],
            "question": "What are the output files?",
            "answer": (
                "FreeTrace outputs CSV files containing trajectory data: particle positions "
                "(x, y) per frame, trajectory IDs, and — if FBM mode is enabled — the "
                "estimated Hurst exponent H and generalised diffusion coefficient K for "
                "each trajectory."
            ),
        },
        {  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-21
            "keywords": ["pixel size", "pixel_size", "frame rate", "framerate",
                         "micron", "micrometer", "physical units", "conversion",
                         "camera", "um/pixel", "seconds per frame"],
            "question": "What are the pixel size and frame rate settings?",
            "answer": (
                "Pixel size (μm/pixel) and frame rate (s/frame) are set in the Basic and "
                "Advanced Stats tabs to convert raw pixel and frame data into physical units "
                "(μm and seconds). Default is 1.0 for both. FreeTrace performs all localization "
                "and tracking in pixel coordinates — unit conversion is applied only during "
                "statistical analysis. These values are camera-dependent: verify your pixel "
                "size and frame rate before analysing data."
            ),
        },
        {
            "keywords": ["jump auto", "auto infer", "jump 0", "automatic jump",
                         "gmm", "gaussian mixture", "infer jump", "default jump",
                         "jump threshold 0"],
            "question": "What happens when jump threshold is set to 0?",
            "answer": (
                "When jump threshold is 0 (the default), FreeTrace automatically infers "
                "the maximum jump distance using a Gaussian Mixture Model (GMM) fitted to "
                "the inter-frame displacement distribution. The minimum auto-inferred value "
                "is clamped to 5 pixels, to avoid overfitting on the high number of "
                "membrane-bound or DNA-bound jump distances that tend to dominate the "
                "short-displacement peak. Set a fixed value (in pixels) only if you want "
                "to override the automatic estimate."
            ),
        },
        {
            "keywords": ["preview", "test", "quick check", "try settings",
                         "before running", "check parameters", "sample frames"],
            "question": "What does the Preview button do?",
            "answer": (
                "Preview runs localization on 50 frames from the middle of your video "
                "using CPU only. It shows detected spots overlaid on the frames so you "
                "can verify that your window size and threshold settings are appropriate "
                "before running the full analysis."
            ),
        },
        {
            "keywords": ["gating", "scatter plot", "class tab", "classify",
                         "h-k scatter", "hk scatter", "select trajectories",
                         "boundary", "region", "export classification",
                         "diffusion state", "population"],
            "question": "What is the H-K scatter plot and gating?",
            "answer": (
                "The Class tab displays a scatter plot of Hurst exponent (H) vs. diffusion "
                "coefficient (K) for every trajectory. You can draw freehand boundaries on "
                "this plot to classify trajectories into different diffusion states (e.g., "
                "confined, Brownian, directed). The Export Classification button saves, for "
                "each gated region and each video, a _diffusion.csv (H and K per trajectory) "
                "and a _traces.csv (full trajectory coordinates), plus a "
                "classification_boundaries.json file to reload the gating later. Requires "
                "both _traces.csv and _diffusion.csv."
            ),
        },
        {
            "keywords": ["diffusion coefficient", "K value", "generalised diffusion",
                         "generalized diffusion", "magnitude", "how fast",
                         "diffusion constant", "diffusivity"],
            "question": "What is the diffusion coefficient K?",
            "answer": (
                "K is the generalised diffusion coefficient estimated by a neural network "
                "for each trajectory. It quantifies the magnitude of motion — larger K means "
                "faster diffusion. K is reported in pixel²/frame^(2H) for fractional Brownian "
                "motion. For classical Brownian motion (H = 0.5), K reduces to D, the standard "
                "diffusion coefficient, in pixel²/frame. The estimated K is not converted to "
                "μm²/s during the analysis. K estimation is only available when FBM mode "
                "is enabled."
            ),
        },
        {
            "keywords": ["hurst exponent", "hurst", "H value", "H exponent",
                         "anomalous exponent", "alpha", "diffusion type",
                         "subdiffusion", "superdiffusion", "brownian",
                         "confined", "directed", "anti-persistent", "persistent"],
            "question": "What is the Hurst exponent H?",
            "answer": (
                "The Hurst exponent H characterises the type of diffusion for each trajectory. "
                "H = 0.5 indicates normal (Brownian) diffusion, H < 0.5 indicates subdiffusion "
                "(confined or anti-persistent motion), and H > 0.5 indicates superdiffusion "
                "(persistent motion). When FBM mode is enabled, the estimated H "
                "for each trajectory is written to _diffusion.csv. Alternatively, you can "
                "estimate the ensemble H for a homogeneous population using the Cauchy "
                "fitting method in the Advanced Stats tab."
            ),
        },
        {
            "keywords": ["angle", "polar angle", "deflection", "turning angle",
                         "direction", "isotropy", "anisotropy", "orientation"],
            "question": "What are the angle and polar angle plots?",
            "answer": (
                "The angle plot shows the deflection angle (0°–180°) between consecutive "
                "step pairs (both steps must be Δt = 1). The polar angle plot shows the "
                "turning angle (0°–360°). For isotropic Brownian motion, both distributions "
                "should be uniform. Deviations reveal directed motion or confinement."
            ),
        },
        {
            "keywords": ["ea-sd", "easd", "ensemble average", "squared displacement",
                         "spreading", "msd basic", "displacement from origin"],
            "question": "What is the EA-SD plot?",
            "answer": (
                "EA-SD (Ensemble-Averaged Squared Displacement) computes, at each time "
                "point, the squared displacement from the trajectory origin, averaged over "
                "all trajectories. It characterises how particles spread over time. A linear "
                "EA-SD indicates Brownian diffusion; sub-linear indicates confinement; "
                "super-linear indicates directed motion. Note: SD here stands for Squared "
                "Displacement, not standard deviation."
            ),
        },
        {
            "keywords": ["ta-ea-sd", "taeasd", "tamsd", "msd", "time average",
                         "mean squared displacement", "advanced msd",
                         "time-averaged", "log-log"],
            "question": "What is TA-EA-SD?",
            "answer": (
                "TA-EA-SD (Time-Averaged Ensemble-Averaged Squared Displacement) is also "
                "known as the MSD (Mean Squared Displacement), where 'mean' refers to the "
                "average over both time and trajectories. For each trajectory, the squared "
                "displacement at lag τ is averaged over all valid time windows of that size "
                "(time-average), then averaged across all trajectories (ensemble-average). "
                "Unlike EA-SD, it exploits all overlapping windows rather than only the "
                "displacement from the origin. Only windows where the actual frame gap "
                "equals τ are included — gaps are never interpolated."
            ),
        },
        {
            "keywords": ["common normalisation", "common normalization", "normalize",
                         "normalise", "compare datasets", "shared bins",
                         "bin edges", "population ratio"],
            "question": "What does common normalisation do?",
            "answer": (
                "When enabled, all loaded datasets share the same histogram bin edges and "
                "the y-axis is normalised to the dataset with the most data points. This "
                "preserves the relative population sizes between datasets, making it "
                "meaningful to compare distributions side by side. When disabled, each "
                "dataset uses its own independent binning."
            ),
        },
        {  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-23
            "keywords": ["cauchy", "cauchy distribution", "cauchy fitting", "cauchy cost",
                         "ratio distribution", "displacement ratio", "cost function",
                         "linking cost", "trajectory cost"],
            "question": "How does FreeTrace use the Cauchy distribution?",
            "answer": (
                "FreeTrace uses the Cauchy distribution in two ways. First, in the core tracking "
                "algorithm: for fractional Brownian motion, the ratio of consecutive displacement "
                "increments follows a Cauchy distribution parametrised by the Hurst exponent H. "
                "FreeTrace evaluates this Cauchy log-likelihood as the cost function when scoring "
                "candidate trajectory links in the graph-based optimisation. Second, in the "
                "Advanced Stats tab: a Cauchy distribution is fitted to the 1D displacement ratio "
                "histogram to estimate H per diffusion state as a post-analysis diagnostic."
            ),
        },
        {  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            "keywords": ["roi", "region of interest", "spatial", "roi tab",
                         "rectangle", "ellipse", "line", "draw shape",
                         "classification", "strict containment", "mean position"],
            "question": "What is the ROI tab and how does spatial ROI selection work?",
            "answer": (
                "The ROI tab lets you select trajectories by spatial region. Load a _traces.csv "
                "(optionally with _diffusion.csv) and draw shapes (rectangle, ellipse, or line) "
                "on the trajectory plot. Each shape boundary divides the space via flood fill into "
                "regions labelled roi0, roi1, etc. Two classification modes: 'Mean Position' classifies "
                "by trajectory centroid, 'Strict Containment' requires all points in the same region "
                "(otherwise excluded). Shapes can be moved, resized, and rotated. Press R/E/L/S to "
                "switch modes, Delete to remove a selected shape, hold Shift while rotating for 15° "
                "snapping. If diffusion data is loaded, the stats table shows per-ROI mean H and K. "
                "Export saves per-ROI _traces.csv and _diffusion.csv files."
            ),
        },
        {
            "keywords": ["imagej", "fiji", "roi file", ".roi", ".zip",
                         "load roi", "import roi"],
            "question": "Can I load ImageJ Fiji ROI files?",
            "answer": (
                "Yes. The 'Load ROI' button in the ROI tab accepts ImageJ/Fiji .roi files (single ROI) "
                "and .zip files containing multiple ROIs (as exported by ImageJ's ROI Manager). "
                "Supported ROI types: rectangle, oval, line, rotated ellipse (subtype ELLIPSE), and "
                "rotated rectangle (subtype ROTATED_RECT). Unsupported types like polygon, freehand, "
                "or polyline are skipped with a count shown in the info label. The coordinates are "
                "in image pixel space, matching FreeTrace's trajectory coordinate system."
            ),
        },
        {  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            "keywords": ["viz", "visualization", "visualisation", "colour", "color",
                         "color map", "colormap", "colourmap", "trajectory color",
                         "trajectory colour", "hurst color", "K color", "jet", "viridis"],
            "question": "What does the Viz tab do?",
            "answer": (
                "The Viz tab renders all trajectories on a spatial plot coloured by a diffusion "
                "property — either H (Hurst exponent) or log K (log₁₀ diffusion coefficient). "
                "Load a _traces.csv + _diffusion.csv pair, then choose 'Color by' (H or log K) "
                "and a colormap (Jet or Viridis). The min/max range defaults to the 2.5th–97.5th "
                "percentile and can be adjusted with the spinboxes or the vertical range slider "
                "on the left of the canvas. The colour bar on the right shows the value mapping."
            ),
        },
        {
            "keywords": ["viz save", "save viz", "save trajectory", "export viz",
                         "save image", "save png", "high resolution", "hi-res",
                         "transparent background", "transparent"],
            "question": "How do I save the Viz trajectory map?",
            "answer": (
                "Click the 'Save' button in the Viz tab controls row. The image is exported as "
                "a PNG with transparent background at high resolution (at least 2048 px on the "
                "shorter side). The saved image contains only the coloured trajectories and the "
                "colour bar with tick labels — no grid, axis lines, or axis labels are included, "
                "so it can be overlaid on other images or used in presentations."
            ),
        },
        {  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            "keywords": ["progress", "progress bar", "elapsed", "eta", "time remaining",
                         "time left", "how long", "estimated time"],
            "question": "What does the progress bar show during execution?",
            "answer": (
                "During execution, the progress bar shows the current stage and completion "
                "percentage. The stages are: Localization (0–40%), Tracking (40–70%), "
                "H estimation (70–90%), and K estimation (90–95%). An elapsed timer is shown "
                "to the right of the bar, along with an estimated time remaining (ETA) based "
                "on the recent progress rate. The ETA adapts to each stage's speed — it will "
                "increase when entering slow stages like H estimation."
            ),
        },
        {
            "keywords": ["batch", "batch mode", "folder", "multiple files", "batch progress"],
            "question": "How does batch mode work?",
            "answer": (
                "Select 'Batch (folder)' mode and choose a folder containing TIFF/ND2 files. "
                "FreeTrace processes each file sequentially. The progress bar shows overall "
                "progress across all files — e.g., with 5 files, each file contributes 20% "
                "to the total. The current file name and index (e.g., [3/5]) are shown in the "
                "progress label and window title. If some files fail, a summary and error log "
                "are shown at the end."
            ),
        },
        {
            "keywords": ["cancel", "stop", "abort", "kill", "terminate", "running"],
            "question": "Can I stop FreeTrace while it is running?",
            "answer": (
                "Yes. Click the 'Stop' button during execution to cancel. The C++ process "
                "is terminated gracefully (SIGTERM). If you close the GUI window while "
                "FreeTrace is running, the child process is also terminated automatically. "
                "On Linux and macOS, orphan prevention ensures the C++ process is killed "
                "even if the GUI crashes unexpectedly."
            ),
        },
        {
            "keywords": ["auto run", "auto statistics", "automatic", "load data",
                         "run statistics", "preprocessing"],
            "question": "Do I need to manually click 'Run Statistics' after loading data?",
            "answer": (
                "No. Both the Basic Stats and Advanced Stats tabs automatically run their "
                "preprocessing as soon as data is loaded. You do not need to click 'Run "
                "Statistics' or 'Run Advanced Stats' — the results appear automatically "
                "after loading. You can still re-run manually if you change parameters "
                "like pixel size or frame rate."
            ),
        },
    ]  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _build_chat_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20 18:00
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Mode selector row
        mode_row = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("color:#ccc; font-size:12px;")
        mode_row.addWidget(mode_label)

        self._chat_mode = QComboBox()
        self._chat_mode.addItems(["FreeTrace Q&A (offline)", "Gemini AI (online)"])
        self._chat_mode.setStyleSheet(
            "background:#2a2a2a; color:#ccc; border:1px solid #555; "
            "border-radius:4px; padding:4px 8px; font-size:12px;"
        )
        self._chat_mode.currentIndexChanged.connect(self._on_chat_mode_changed)
        mode_row.addWidget(self._chat_mode, 1)
        layout.addLayout(mode_row)

        # API key row (Gemini mode only)
        self._gemini_key_widget = QWidget()
        key_layout = QVBoxLayout(self._gemini_key_widget)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(2)

        key_row = QHBoxLayout()
        key_label = QLabel("Gemini API Key:")
        key_label.setStyleSheet("color:#ccc; font-size:12px;")
        key_row.addWidget(key_label)

        self._gemini_key_input = QLineEdit()
        self._gemini_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._gemini_key_input.setPlaceholderText("Paste your Gemini API key here")
        self._gemini_key_input.setStyleSheet(
            "background:#2a2a2a; color:#ccc; border:1px solid #555; "
            "border-radius:4px; padding:4px 8px; font-size:12px;"
        )
        key_row.addWidget(self._gemini_key_input, 1)

        get_key_btn = QPushButton("Get free key")
        get_key_btn.setStyleSheet(
            "background:#2e7d32; color:#fff; border:none; border-radius:4px; "
            "padding:4px 12px; font-size:12px;"
        )
        get_key_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        get_key_btn.setToolTip(
            "Free — no payment required. Usage beyond the daily limit is automatically stopped, never charged."
        )
        get_key_btn.clicked.connect(
            lambda: _open_url("https://aistudio.google.com/apikey")
        )
        key_row.addWidget(get_key_btn)

        del_key_btn = QPushButton("Delete key")
        del_key_btn.setStyleSheet(
            "background:#8b0000; color:#fff; border:none; border-radius:4px; "
            "padding:4px 12px; font-size:12px;"
        )
        del_key_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        del_key_btn.setToolTip(
            "Opens Google AI Studio. To delete your key, click the three-dot menu next to it and select Delete."
        )
        del_key_btn.clicked.connect(
            lambda: _open_url("https://aistudio.google.com/apikey")
        )
        key_row.addWidget(del_key_btn)
        key_layout.addLayout(key_row)

        privacy_label = QLabel(
            "Free tier: your prompts may be used by Google to improve their products."
        )
        privacy_label.setStyleSheet("color:#888; font-size:10px; font-style:italic;")
        key_layout.addWidget(privacy_label)

        self._gemini_key_widget.hide()  # hidden by default (Q&A mode)
        layout.addWidget(self._gemini_key_widget)

        # Chat display area
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setStyleSheet(
            "background:#1a1a1a; color:#ccc; border:1px solid #333; "
            "border-radius:4px; padding:8px; font-size:13px; "
            "font-family:'Courier New', monospace;"
        )
        self._chat_display.setHtml(self._qa_welcome_html())
        layout.addWidget(self._chat_display, 1)

        # Input row
        input_row = QHBoxLayout()
        self._chat_input = QLineEdit()
        self._chat_input.setPlaceholderText("Type your question...")
        self._chat_input.setStyleSheet(
            "background:#2a2a2a; color:#ccc; border:1px solid #555; "
            "border-radius:4px; padding:6px 10px; font-size:13px;"
        )
        self._chat_input.returnPressed.connect(self._send_chat)
        input_row.addWidget(self._chat_input, 1)

        self._chat_send_btn = QPushButton("Send")
        self._chat_send_btn.setStyleSheet(
            "background:#3a3a3a; color:#ddd; border:1px solid #555; "
            "border-radius:4px; padding:6px 16px; font-size:13px;"
        )
        self._chat_send_btn.clicked.connect(self._send_chat)
        input_row.addWidget(self._chat_send_btn)

        self._chat_clear_btn = QPushButton("Clear")
        self._chat_clear_btn.setStyleSheet(
            "background:#3a3a3a; color:#ddd; border:1px solid #555; "
            "border-radius:4px; padding:6px 12px; font-size:13px;"
        )
        self._chat_clear_btn.clicked.connect(self._clear_chat)
        input_row.addWidget(self._chat_clear_btn)

        chat_help_btn = QPushButton("?")
        chat_help_btn.setFixedWidth(32)
        chat_help_btn.setStyleSheet(
            "background:#3a3a3a; color:#7ec8e3; border:1px solid #555; "
            "border-radius:4px; padding:6px; font-size:14px; font-weight:bold;"
        )
        chat_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        chat_help_btn.clicked.connect(self._show_chat_help)
        input_row.addWidget(chat_help_btn)

        layout.addLayout(input_row)
        return widget

    def _qa_welcome_html(self):
        """Generate welcome message listing available Q&A topics."""
        lines = [
            "<p style='color:#7ec8e3; font-size:13px;'><b>FreeTrace Q&A</b></p>",
            "<p style='color:#999; font-size:12px;'>Type a question or keyword. "
            "Available topics:</p>",
            "<ul style='color:#aaa; font-size:12px;'>",
        ]
        for qa in self._PREDEFINED_QA:
            lines.append(f"<li>{qa['question']}</li>")
        lines.append("</ul>")
        lines.append(
            "<p style='color:#888; font-size:11px; font-style:italic;'>"
            "Switch to Gemini AI mode for open-ended questions.</p>"
        )
        return "".join(lines)

    def _on_chat_mode_changed(self, index):
        """Toggle between Q&A and Gemini mode."""
        if index == 0:  # Q&A mode
            self._gemini_key_widget.hide()
            self._chat_display.setHtml(self._qa_welcome_html())
        else:  # Gemini mode
            self._gemini_key_widget.show()
            self._chat_display.setHtml(
                "<p style='color:#666; text-align:center; margin-top:40px;'>"
                "Ask questions about FreeTrace, microscopy, biophysics, "
                "or general science.</p>"
            )
        self._chat_history.clear()

    def _search_qa(self, query: str):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-21
        """Search predefined Q&A by keyword + fuzzy matching. Returns (question, answer) or None."""
        from difflib import SequenceMatcher
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 2]
        best_match = None
        best_score = 0.0
        for qa in self._PREDEFINED_QA:
            score = 0.0
            # Exact keyword match (strongest signal)
            for kw in qa["keywords"]:
                if kw in query_lower:
                    score += 2.0
            # Fuzzy keyword match (handles typos)
            for word in query_words:
                for kw in qa["keywords"]:
                    for kw_part in kw.split():
                        ratio = SequenceMatcher(None, word, kw_part).ratio()
                        if ratio >= 0.75:
                            score += ratio
            # Word overlap with question text
            q_words = qa["question"].lower().split()
            for word in query_words:
                if word in q_words:
                    score += 0.5
                else:
                    for qw in q_words:
                        if SequenceMatcher(None, word, qw).ratio() >= 0.8:
                            score += 0.3
                            break
            if score > best_score:
                best_score = score
                best_match = qa
        if best_score >= 0.75:
            return best_match["question"], best_match["answer"]
        return None

    def _show_chat_help(self):
        """Show information about the AI Chat feature and billing."""
        if self._chat_mode.currentIndex() == 0:
            QMessageBox.information(self, "FreeTrace Q&A — Info", (
                "<b>FreeTrace Q&A (offline)</b><br><br>"
                "This mode answers common questions about FreeTrace parameters "
                "and features using predefined answers.<br><br>"
                "No internet connection or API key is required. "
                "No data is sent anywhere.<br><br>"
                "For open-ended questions, switch to Gemini AI mode."
            ))
        else:
            QMessageBox.information(self, "Gemini AI — Info", (
                "<b>About Gemini AI</b><br><br>"
                "This mode uses Google's Gemini API (Flash-Lite model) to answer "
                "questions about FreeTrace, microscopy, biophysics, "
                "and general science. Requires an internet connection.<br><br>"
                "<b>Is it free?</b><br>"
                "Yes. The Gemini API free tier requires no credit card and no billing "
                "account. You will never be charged — if you exceed the daily limit "
                "(1,000 requests/day), the API simply returns an error.<br><br>"
                "<b>Can Google change this?</b><br>"
                "Even if Google removes the free tier in the future, you cannot be "
                "charged. Free-tier API keys have no payment method attached — "
                "Google has no way to bill you. The worst case is that the chat "
                "feature stops working.<br><br>"
                "<b>Privacy</b><br>"
                "Your prompts and responses are sent directly to Google's servers "
                "via the Gemini API. Your microscopy images, videos, and tracking "
                "results are never transferred — only the text you type in the chat. "
                "On the free tier, Google may use your prompts and responses to "
                "improve their products.<br><br>"
                "<b>Accuracy</b><br>"
                "AI responses may not always be accurate. Please verify important "
                "results independently."
            ))

    def _send_chat(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20 18:00
        """Send user message — routes to Q&A or Gemini based on mode."""
        message = self._chat_input.text().strip()
        if not message:
            return

        if self._chat_mode.currentIndex() == 0:
            # Predefined Q&A mode
            self._chat_display.append(
                f"<p style='color:#7ec8e3;'><b>You:</b> {message}</p>"
            )
            self._chat_input.clear()
            result = self._search_qa(message)
            if result:
                _, answer = result
                html_answer = answer.replace("\n", "<br>")
                self._chat_display.append(
                    f"<p style='color:#ccc;'><b>AI:</b> {html_answer}</p>"
                )
            else:
                self._chat_display.append(
                    "<p style='color:#888;'><b>AI:</b> Sorry, I don't have a predefined "
                    "answer for that question. Try different keywords, or switch to "
                    "Gemini AI mode for open-ended questions.</p>"
                )
            self._chat_display.verticalScrollBar().setValue(
                self._chat_display.verticalScrollBar().maximum()
            )
            return

        # Gemini AI mode
        api_key = self._gemini_key_input.text().strip()
        if not api_key:
            self._chat_display.append(
                "<p style='color:#f0a500;'>Please enter your Gemini API key above.</p>"
            )
            return
        if self._chat_worker is not None and self._chat_worker.isRunning():
            return

        self._chat_display.append(
            f"<p style='color:#7ec8e3;'><b>You:</b> {message}</p>"
        )
        self._chat_input.clear()
        self._chat_send_btn.setEnabled(False)

        self._chat_display.append(
            "<p style='color:#888;'><i>Thinking...</i></p>"
        )
        self._chat_ai_buffer = ""

        self._chat_worker = GeminiChatWorker(api_key, message, self._chat_history)
        self._chat_worker.reply_chunk.connect(self._on_chat_chunk)
        self._chat_worker.reply_done.connect(self._on_chat_done)
        self._chat_worker.error.connect(self._on_chat_error)
        self._chat_worker.start()

    def _on_chat_chunk(self, text: str):
        """Accumulate streamed text chunks."""
        self._chat_ai_buffer += text

    def _on_chat_done(self):
        """Display the complete AI response and update history."""
        self._chat_send_btn.setEnabled(True)
        cursor = self._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        import re
        converted = _latex_to_unicode(self._chat_ai_buffer)
        html_text = converted.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html_text)  # **bold**
        html_text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', html_text)      # *italic*
        html_text = html_text.replace("\n", "<br>")
        self._chat_display.append(
            f"<p style='color:#ccc;'><b>AI:</b> {html_text}</p>"
        )
        self._chat_display.verticalScrollBar().setValue(
            self._chat_display.verticalScrollBar().maximum()
        )
        self._chat_history.append(
            {"role": "user", "parts": [{"text": self._chat_worker.message}]}
        )
        self._chat_history.append(
            {"role": "model", "parts": [{"text": self._chat_ai_buffer}]}
        )
        if len(self._chat_history) > 40:
            self._chat_history = self._chat_history[-40:]
        self._chat_ai_buffer = ""

    def _on_chat_error(self, error_msg: str):
        """Display error in chat."""
        self._chat_send_btn.setEnabled(True)
        cursor = self._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.deletePreviousChar()
        self._chat_display.append(
            f"<p style='color:#e57373;'>Error: {error_msg}</p>"
        )
        self._chat_ai_buffer = ""

    def _clear_chat(self):
        """Clear chat display and history."""
        if self._chat_mode.currentIndex() == 0:
            self._chat_display.setHtml(self._qa_welcome_html())
        else:
            self._chat_display.setHtml(
                "<p style='color:#666; text-align:center; margin-top:40px;'>"
                "Ask questions about FreeTrace, microscopy, biophysics, "
                "or general science.</p>"
            )
        self._chat_history.clear()
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20 18:00

    # ---- Analysis tab (sub-tabs: Class | ROI | Basic Stats | Adv Stats) ----
    def _build_analysis_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sub-tab widget inside Analysis
        self._analysis_tabs = QTabWidget()
        self._analysis_tabs.setObjectName("analysisTabs")
        self._analysis_tabs.addTab(self._build_help_tab(), "Help")
        self._analysis_tabs.addTab(self._build_class_tab(), "Class")
        self._analysis_tabs.addTab(self._build_roi_tab(), "ROI")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._analysis_tabs.addTab(self._build_basic_stats_tab(), "Basic Stats")
        self._analysis_tabs.addTab(self._build_adv_stats_tab(), "Adv Stats")
        self._analysis_tabs.addTab(self._build_viz_tab(), "Viz")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._analysis_tabs.currentChanged.connect(self._on_analysis_tab_changed)
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
            "<p><b>ROI tab</b> — Requires <code>_traces.csv</code>; "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            "<code>_diffusion.csv</code> is optional. Draw rectangles, ellipses, or lines "
            "on the spatial trajectory plot to define regions of interest (ROIs). "
            "Shapes can be moved, resized, and rotated by clicking on them "
            "(in any drawing mode or Select mode). Two classification modes: "
            "<i>Mean Position</i> (classify by trajectory centroid) and "
            "<i>Strict Containment</i> (all points must lie in the same ROI). "
            "If diffusion data is available, the H-K scatter is colored by ROI. "
            "Export filtered <code>_traces.csv</code> and <code>_diffusion.csv</code> per ROI.</p>"
            "<p><b>Basic Stats tab</b> — Requires <code>_traces.csv</code>; "
            "<code>_diffusion.csv</code> is optional. If only traces are available, "
            "all trajectory-based plots (jump distance, duration, EA-SD, angles) work normally. "
            "H and K distributions require <code>_diffusion.csv</code>.</p>"
            "<p><b>Advanced Stats tab</b> — Requires <code>_traces.csv</code>; "  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
            "auto-pairs a sibling <code>_loc.csv</code> for the noise-aware panels "
            "(skipped silently when <code>bg_median</code>/<code>bg_var</code>/"
            "<code>integrated_flux</code> are missing — re-run localisation with the "
            "current FreeTrace to enable). A sibling TIFF, when present, recomputes "
            "<code>bg_var</code>/<code>integrated_flux</code> via the thesis-style "
            "annulus convention. <code>_diffusion.csv</code> is not used.</p>"
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
            "<h3 style='color:#66ccff;'>ROI Tab</h3>"  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            "<p>The ROI tab allows spatial selection of trajectories by region of interest. "
            "Trajectories are rendered as connected lines on the plot. Draw shapes to divide "
            "the spatial domain into ROIs (roi0, roi1, ...). Four modes:</p>"
            "<ul>"
            "<li><b>Rectangle</b> — Click and drag to draw a rectangular ROI.</li>"
            "<li><b>Ellipse</b> — Click and drag to draw an elliptical ROI.</li>"
            "<li><b>Line</b> — Click and drag to draw a dividing line; endpoints auto-extend "
            "to the plot edges, splitting the space in two.</li>"
            "<li><b>Select</b> — Click on a shape to select it without drawing new shapes.</li>"
            "</ul>"
            "<p><b>Keyboard shortcuts:</b> Press <b>R</b>, <b>E</b>, <b>L</b>, or <b>S</b> "
            "to switch to Rectangle, Ellipse, Line, or Select mode. "
            "Press <b>Delete</b> or <b>Backspace</b> to remove the currently selected shape.</p>"
            "<p><b>Selecting &amp; editing shapes:</b> Click on any existing shape (in any mode) "
            "to select it. Yellow square handles appear at the corners for resizing, "
            "and a cyan circle handle above the shape for rotating. Drag the shape body "
            "to move it. Resize is symmetric around the shape centre.</p>"
            "<p><b>Rotation:</b> Drag the cyan rotation handle to rotate rectangles and ellipses "
            "around their centre. Hold <b>Shift</b> while rotating to snap to 15° increments. "
            "The rotation angle is preserved when saving/loading ROI "
            "boundaries. Lines do not support rotation (use their endpoint handles instead).</p>"
            "<p>Right-click removes the last drawn shape. Multiple shapes further subdivide "
            "the space.</p>"
            "<p><b>Classification modes:</b></p>"
            "<ul>"
            "<li><b>Mean Position</b> — Each trajectory is classified by its centroid "
            "(mean x, y). Fast and simple.</li>"
            "<li><b>Strict Containment</b> — A trajectory is assigned to a ROI only if "
            "<i>all</i> its points lie within the same region. Trajectories crossing "
            "ROI boundaries are excluded (shown dimmed, labelled 'excluded' in stats).</li>"
            "</ul>"
            "<p>If <code>_diffusion.csv</code> is loaded, the right panel shows an "
            "H-K scatter plot coloured by ROI assignment, and the stats table includes "
            "per-ROI mean H and mean K values. Excluded trajectories appear dimmed.</p>"
            "<p><b>Export ROI</b> — Saves per-ROI <code>_traces.csv</code> and "
            "<code>_diffusion.csv</code> (if available), plus <code>roi_boundaries.json</code> "
            "(including shape type, coordinates, and rotation angle).</p>"
            "<p><b>Load ROI</b> — Import shapes from a FreeTrace <code>roi_boundaries.json</code>, "
            "an ImageJ/Fiji <code>.roi</code> file (single ROI), or a <code>.zip</code> file "
            "containing multiple ImageJ ROIs. Supported ImageJ ROI types: rectangle, oval, line, "
            "rotated ellipse, and rotated rectangle. Unsupported types (polygon, freehand, etc.) "
            "are skipped with a warning.</p>"
            "<h3 style='color:#66ccff;'>Basic Stats Tab</h3>"
            "<p><b>H &amp; K distributions</b> — Per-trajectory Hurst exponent and diffusion "
            "coefficient (requires <code>_diffusion.csv</code>). "
            "K is computed in pixel &amp; frame scale, not converted to μm &amp; s.</p>"
            "<p><b>Jump Distance</b> — Per-step Euclidean displacement √(Δx² + Δy²), Δt = 1 only. "
            "Assumes isotropic motion.</p>"
            "<p><b>Mean Jump Distance</b> — Average jump distance per trajectory (one value per trajectory).</p>"
            "<p><b>Duration</b> — Total observation time in frames: the sum of frame differences "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "across the trajectory (last frame − first frame).</p>"
            "<p><b>EA-SD</b> — Ensemble-Averaged Squared Displacement: at each time point, "  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-20
            "the squared displacement from the trajectory origin is averaged over all "
            "trajectories. Here, SD stands for Squared Displacement, not standard deviation.</p>"
            "<p><b>Angle / Polar Angle</b> — Deflection angle (0°–180°) and signed turning "
            "angle (0°–360°) between consecutive step pairs, both Δt = 1. "
            "Uniform if isotropic &amp; Brownian.</p>"
            "<p><b>TA-EA-SD (Time-Averaged Ensemble-Averaged Squared Displacement)</b> — "  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
            "For each trajectory, the squared displacement at lag τ is averaged over all "
            "valid time windows of size τ (the <i>time-average</i>). These per-trajectory "
            "means are then averaged across all trajectories in the ensemble (the "
            "<i>ensemble-average</i>). This two-stage averaging is more robust than EA-SD "
            "above, especially for short trajectories with frame gaps: EA-SD uses only "
            "the displacement from the trajectory origin at each time point, while "
            "TA-EA-SD exploits all overlapping windows of size τ. "
            "Only windows where the actual frame gap equals the lag τ are included "
            "(gaps are skipped, not interpolated). The shaded region shows ± 1 std "
            "across the ensemble.</p>"
            "<p>The <b>log-log TA-EA-SD</b> is shown below. On a log-log scale, the "
            "slope directly gives the anomalous diffusion exponent α: slope = 1 for "
            "Brownian motion, &lt; 1 for subdiffusion, &gt; 1 for superdiffusion. "
            "The log-log plot omits the std fill to avoid y-axis distortion.</p>"
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
            "per homogeneous population are excluded from fitting. This noise-free "
            "single-Ĥ Cauchy fit is distinct from the per-Δ corrected Cauchy scan in "
            "the Advanced Stats tab.</p>"
            "<h3 style='color:#66ccff;'>Advanced Stats Tab</h3>"  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
            "<p>The Advanced Stats tab adds noise-aware diffusion analyses driven by the "
            "per-spot σ_loc CRLB and a motion-blur correction R. The headline output is "
            "the corrected Cauchy multi-Δ scan Ĥ(Δ) with a 95% CRLB band, complemented "
            "by a pairwise Z-significance matrix, an adjacent-Δ drift trace, a λ_noise "
            "sensitivity sweep, a noise-floor diagnostic, and a precision map. The "
            "panel-by-panel guide further down this Help tab covers each in detail.</p>"
            "<p><b>1D Displacement (Δx, Δy)</b> — Projection of each step onto the x and y "
            "axes separately, using only consecutive-frame steps (Δt = 1). For a "
            "homogeneous population of Brownian or fBm molecules, each projection is "
            "Gaussian with zero mean. A non-Gaussian shape (heavy tails, multiple peaks) "
            "indicates a heterogeneous population with mixed diffusion states. Δx and Δy "
            "are shown overlaid; for isotropic motion they should be identical. "
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
            "<td style='padding:4px 12px;'>Per-population — one Ĥ per homogeneous population</td></tr>"
            "<tr><td style='padding:4px 12px;'><b>Requires</b></td>"
            "<td style='padding:4px 12px;'><code>_diffusion.csv</code> (Basic Stats / Class tab)</td>"
            "<td style='padding:4px 12px;'><code>_traces.csv</code> (Basic Stats tab)</td></tr>"  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
            "<tr><td style='padding:4px 12px;'><b>Best for</b></td>"
            "<td style='padding:4px 12px;'>Per-trajectory H distribution, H-K scatter classification</td>"
            "<td style='padding:4px 12px;'>Independent validation of the ensemble diffusion regime</td></tr>"
            "</table>"
            "<p>Comparing the two: the NN-estimated H distribution (Basic Stats) "  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
            "shows the spread of H across individual trajectories, while the "
            "Cauchy-fitted Ĥ (Basic Stats) gives an ensemble-level estimate. "
            "If the Cauchy Ĥ falls near the peak of the NN H distribution, "
            "both methods agree. A significant discrepancy may indicate that the "
            "NN model and the Cauchy assumption capture different aspects of the "
            "motion, or that the population is heterogeneous.</p>"
            "<p><b>Bias warning:</b> The NN-estimated H can be biased for short "
            "trajectories. The neural network requires a minimum number of "
            "data points to produce a reliable estimate; when the trajectory is "
            "too short, the network tends to regress toward H ≈ 0.5 (Brownian) "
            "regardless of the true dynamics. The Cauchy fit, by pooling all "
            "displacement ratios across an entire population, is more robust "
            "against short trajectory lengths. When the NN H distribution "
            "shows a strong peak at H = 0.5 while the Cauchy Ĥ deviates from "
            "0.5, short-trajectory bias in the NN estimate is a likely cause.</p>"
            "<h3 style='color:#66ccff;'>Running FreeTrace</h3>"  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            "<p><b>Progress bar</b> — Shows real-time progress during execution. "
            "The four stages are mapped to percentage ranges: Localization (0–40%), "
            "Tracking (40–70%), H estimation (70–90%), and K estimation (90–95%). "
            "The progress bar also displays the current stage label. "
            "An elapsed timer and estimated time remaining (ETA) are shown to the right.</p>"
            "<p><b>ETA</b> — The estimated time remaining adapts to the current processing "
            "speed. It uses the rate of recent progress updates rather than a simple average, "
            "so it adjusts when entering slower stages like H estimation.</p>"
            "<p><b>Batch mode</b> — In batch mode, the progress bar maps each file's "
            "internal progress into a global percentage. The window title shows the "
            "current file index (e.g., [3/5]). If some files fail, an error log is "
            "written and a summary dialog is shown at the end.</p>"
            "<p><b>Stop / Cancel</b> — Click 'Stop' to cancel execution. The child "
            "process is terminated. Closing the GUI window also terminates any running "
            "process. On Linux and macOS, orphan prevention ensures the C++ process "
            "is killed even if the GUI exits unexpectedly.</p>"
            "<h3 style='color:#66ccff;'>Viz Tab</h3>"
            "<p>The Viz tab provides trajectory visualisation coloured by diffusion "
            "properties. Load a FreeTrace output pair (<code>_traces.csv</code> + "
            "<code>_diffusion.csv</code>) to render all trajectories on a spatial plot.</p>"
            "<p><b>Color by</b> — Choose between <b>H</b> (Hurst exponent) or "
            "<b>log K</b> (log₁₀ of the diffusion coefficient). Each trajectory "
            "is drawn in a colour corresponding to its value.</p>"
            "<p><b>Colormap</b> — Select from <b>Jet</b> (red → blue) or "
            "<b>Viridis</b> (purple → yellow). The colour bar on the right of the "
            "canvas shows the mapping.</p>"
            "<p><b>Min / Max range</b> — Controls which value range is mapped to the "
            "full colour scale. The vertical range slider on the left of the canvas "
            "and the Min/Max spinboxes are synchronised. Default range: 2.5th–97.5th "
            "percentile of the data. Values outside the range are clamped to the "
            "endpoint colours.</p>"
            "<p><b>Save</b> — Export the current view as a high-resolution PNG with "
            "transparent background (≥ 2048 px). The saved image contains only the "
            "coloured trajectories and the colour bar with tick labels — no grid, "
            "axes, or axis labels.</p>"
            "<h3 style='color:#66ccff;'>Common Normalisation</h3>"
            "<p>When enabled, all datasets share the same bin edges and the y-axis is "
            "normalised to the dataset with the most data points. The largest dataset "
            "sums to 100%, smaller datasets sum proportionally less. This preserves "
            "population size information when comparing datasets.</p>"
        )
        container_layout.addWidget(help_text)

        # Adv Stats panel-by-panel guide (moved from former popup) // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
        adv_stats_panel_guide = QLabel()
        adv_stats_panel_guide.setWordWrap(True)
        adv_stats_panel_guide.setTextFormat(Qt.TextFormat.RichText)
        adv_stats_panel_guide.setStyleSheet("color:#cccccc; font-size:13px; padding:0 20px 12px 20px;")
        adv_stats_panel_guide.setText(self._adv_stats_help_html())
        container_layout.addWidget(adv_stats_panel_guide)
        # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29

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
        layout.addLayout(toolbar)

        self._analysis_info_label = QLabel("Draw a boundary on the H-K plot to classify trajectories.")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._analysis_info_label.setStyleSheet("color:#888; padding-left:4px;")
        self._analysis_info_label.setWordWrap(True)
        layout.addWidget(self._analysis_info_label)

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

    # ---- ROI sub-tab (spatial region of interest) -------------------------  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    def _build_roi_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Toolbar row
        toolbar = QHBoxLayout()

        self._roi_load_btn = QPushButton("Load Data")
        self._roi_load_btn.clicked.connect(self._on_roi_load_data)
        toolbar.addWidget(self._roi_load_btn)

        # Shape mode selector
        mode_label = QLabel("Shape:")
        mode_label.setStyleSheet("color:#aaa;")
        toolbar.addWidget(mode_label)
        self._roi_mode_combo = QComboBox()
        self._roi_mode_combo.addItems([ROICanvas.MODE_RECT, ROICanvas.MODE_ELLIPSE,
                                       ROICanvas.MODE_LINE, ROICanvas.MODE_SELECT])  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._roi_mode_combo.currentTextChanged.connect(self._on_roi_mode_changed)
        self._roi_mode_combo.setFixedWidth(110)
        toolbar.addWidget(self._roi_mode_combo)

        # Classification mode selector
        classify_label = QLabel("Classify:")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        classify_label.setStyleSheet("color:#aaa;")
        toolbar.addWidget(classify_label)
        self._roi_classify_combo = QComboBox()
        self._roi_classify_combo.addItems([ROICanvas.CLASSIFY_MEAN, ROICanvas.CLASSIFY_STRICT])
        self._roi_classify_combo.currentTextChanged.connect(self._on_roi_classify_mode_changed)
        self._roi_classify_combo.setFixedWidth(160)
        toolbar.addWidget(self._roi_classify_combo)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

        self._roi_clear_btn = QPushButton("Clear ROI")
        self._roi_clear_btn.clicked.connect(self._on_roi_clear)
        toolbar.addWidget(self._roi_clear_btn)

        self._roi_load_boundary_btn = QPushButton("Load ROI")
        self._roi_load_boundary_btn.clicked.connect(self._on_roi_load_boundary)
        toolbar.addWidget(self._roi_load_boundary_btn)

        self._roi_export_btn = QPushButton("Export ROI")
        self._roi_export_btn.clicked.connect(self._on_roi_export)
        toolbar.addWidget(self._roi_export_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._roi_info_label = QLabel("Draw shapes on the trajectory plot to define spatial ROIs.")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._roi_info_label.setStyleSheet("color:#888; padding-left:4px;")
        self._roi_info_label.setWordWrap(True)
        layout.addWidget(self._roi_info_label)

        # Vertical splitter: top = canvases, bottom = stats
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: two canvases side by side (ROI scatter + H-K colored by ROI)
        canvas_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ROI canvas (left) with title
        roi_container = QWidget()
        roi_layout = QVBoxLayout(roi_container)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.setSpacing(2)
        roi_title = QLabel("Trajectory Positions — Draw Shapes to Define ROIs")
        roi_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        roi_title.setStyleSheet("color:#ccc; font-size:12px; font-weight:bold; padding:4px;")
        roi_layout.addWidget(roi_title)
        self._roi_canvas = ROICanvas()
        self._roi_canvas.roi_changed.connect(self._on_roi_changed)
        self._roi_canvas.mode_requested.connect(self._roi_mode_combo.setCurrentText)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._roi_canvas.setMinimumSize(300, 250)
        roi_layout.addWidget(self._roi_canvas)
        canvas_splitter.addWidget(roi_container)

        # H-K scatter colored by ROI (right) — only visible when _diffusion.csv loaded
        hk_roi_container = QWidget()
        hk_roi_layout = QVBoxLayout(hk_roi_container)
        hk_roi_layout.setContentsMargins(0, 0, 0, 0)
        hk_roi_layout.setSpacing(2)
        self._roi_hk_title = QLabel("H-K Scatter Colored by ROI")
        self._roi_hk_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._roi_hk_title.setStyleSheet("color:#ccc; font-size:12px; font-weight:bold; padding:4px;")
        hk_roi_layout.addWidget(self._roi_hk_title)
        self._roi_hk_canvas = QGraphicsView()
        self._roi_hk_canvas.setStyleSheet("background:#1a1a1a; border:none;")
        self._roi_hk_canvas.setMinimumSize(300, 250)
        self._roi_hk_scene = QGraphicsScene()
        self._roi_hk_canvas.setScene(self._roi_hk_scene)
        self._roi_hk_canvas.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._roi_hk_canvas.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._roi_hk_canvas.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hk_roi_layout.addWidget(self._roi_hk_canvas)
        canvas_splitter.addWidget(hk_roi_container)

        canvas_splitter.setSizes([500, 500])
        main_splitter.addWidget(canvas_splitter)

        # Bottom: stats panel
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_widget = QWidget()
        self._roi_stats_layout = QVBoxLayout(stats_widget)
        self._roi_stats_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._roi_stats_label = QLabel(
            "No data loaded.\n\nClick 'Load Data' to load trajectory data, "
            "then draw shapes to define ROIs.")
        self._roi_stats_label.setWordWrap(True)
        self._roi_stats_label.setStyleSheet("color:#aaa; font-size:13px; padding:8px;")
        self._roi_stats_layout.addWidget(self._roi_stats_label)

        stats_scroll.setWidget(stats_widget)
        stats_scroll.setMinimumHeight(80)
        main_splitter.addWidget(stats_scroll)

        main_splitter.setSizes([500, 150])
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter)

        # ROI-specific data storage
        self._roi_datasets = []  # list of dicts: video_name, traces_df, diffusion_df (optional)

        return widget
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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

        self._stats_info_label = QLabel("Click 'Load Data' to load trajectory data.")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._stats_info_label.setStyleSheet("color:#888; padding-left:4px;")
        self._stats_info_label.setWordWrap(True)
        layout.addWidget(self._stats_info_label)

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
        if n > 0:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            has_diff = sum(1 for ds in self._stats_datasets if ds['diffusion_df'] is not None)
            total_traj = sum(ds['traces_df']['traj_idx'].nunique() for ds in self._stats_datasets)
            if n == 1:
                fname = self._stats_datasets[0]['video_name']
                self._stats_info_label.setText(
                    f"Loaded {total_traj} trajectories from '{fname}'"
                    f" ({has_diff} with diffusion data).")
            else:
                self._stats_info_label.setText(
                    f"Loaded {total_traj} trajectories from {n} videos"
                    f" ({has_diff} with diffusion data).")
            self._on_run_preprocessing()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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
            """Fill the stats panel with lines of (label, color, mean, std).
            If mean/std are both None, only the bold label line is drawn (informational)."""
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
                if mean_val is not None and std_val is not None:  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
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

        # --- TA-EA-SD (Time-Averaged Ensemble-Averaged Squared Displacement) --- // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
        # Moved here from Advanced Stats since these are noise-free / empirical analyses.
        tamsd_section = CollapsibleSection("TA-EA-SD (Time-Averaged Ensemble-Averaged Squared Displacement)")
        with plt.style.context(dark_style):
            fig = Figure(figsize=(10, 4), dpi=100, constrained_layout=True)
            ax = fig.add_subplot(111)
            plotted = False
            for di, r, st, color in _iter_colors(results_list):
                tamsd = r.get('tamsd')
                if tamsd is None or len(tamsd) == 0:
                    continue
                subset = tamsd[tamsd['state'] == st]
                if subset.empty:
                    continue
                label = _ds_label(r, st) or 'data'
                ax.plot(subset['time'], subset['mean'], color=color, label=label, linewidth=1.5)
                mean_arr = subset['mean'].to_numpy()
                std_arr = subset['std'].to_numpy()
                time_arr = subset['time'].to_numpy()
                ax.fill_between(time_arr, mean_arr - std_arr, mean_arr + std_arr,
                                color=color, alpha=0.2)
                plotted = True
            ax.set_xlabel('Time lag (s)')
            ax.set_ylabel('TA-EA-SD (μm²)')
            ax.set_title('TA-EA-SD')
            if need_legend and plotted:
                ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
        tamsd_section.add_widget(_make_canvas(fig))
        self._stats_plot_layout.addWidget(tamsd_section)

        # --- TA-EA-SD log-log ---
        tamsd_log_section = CollapsibleSection("TA-EA-SD — log-log")
        with plt.style.context(dark_style):
            fig_log = Figure(figsize=(10, 4), dpi=100, constrained_layout=True)
            ax_log = fig_log.add_subplot(111)
            plotted = False
            for di, r, st, color in _iter_colors(results_list):
                tamsd = r.get('tamsd')
                if tamsd is None or len(tamsd) == 0:
                    continue
                subset = tamsd[tamsd['state'] == st]
                if subset.empty:
                    continue
                valid = (subset['mean'] > 0) & (subset['time'] > 0)
                s = subset[valid]
                if s.empty:
                    continue
                label = _ds_label(r, st) or 'data'
                ax_log.plot(s['time'], s['mean'], color=color, label=label, linewidth=1.5)
                plotted = True
            ax_log.set_xscale('log')
            ax_log.set_yscale('log')
            ax_log.set_xlabel('Time lag (s)')
            ax_log.set_ylabel('TA-EA-SD (μm²)')
            ax_log.set_title('TA-EA-SD — log-log')
            if need_legend and plotted:
                ax_log.legend(fontsize=7, loc='best')
            ax_log.grid(True, alpha=0.3, which='both')
        tamsd_log_section.add_widget(_make_canvas(fig_log))
        self._stats_plot_layout.addWidget(tamsd_log_section)

        # --- 1D Displacement Ratio — Cauchy fit (noise-free, Δ=1) ---
        ratio_section = CollapsibleSection("1D Displacement Ratio — Cauchy Fit (noise-free, Δ = 1)")
        with plt.style.context(dark_style):
            for di, r, st, color in _iter_colors(results_list):
                ratios_df = r.get('ratios_1d')
                cauchy_fits = r.get('cauchy_fits') or {}
                if ratios_df is None or len(ratios_df) == 0:
                    continue
                subset = ratios_df[ratios_df['state'] == st]
                if subset.empty:
                    continue
                fig3, ax3, ax3_stats = _make_fig_with_stats()
                label = _ds_label(r, st) or 'data'
                ratio_data = subset['ratio'].to_numpy()
                ratio_clipped = ratio_data[(ratio_data > -10) & (ratio_data < 10)]
                if len(ratio_clipped) > 0:
                    ax3.hist(ratio_clipped, bins=100, density=True, alpha=0.6,
                             color=color, edgecolor='none', label=f'{label} data')
                stat_lines = [(f'Ratio (n={len(ratio_clipped)})', color,
                               np.mean(ratio_clipped) if len(ratio_clipped) > 0 else None,
                               np.std(ratio_clipped) if len(ratio_clipped) > 0 else None)]
                if st in cauchy_fits:
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
                if need_legend:
                    ax3.legend(fontsize=7, loc='best')
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(-5, 5)
                _fill_stats_panel(ax3_stats, stat_lines)
                ratio_section.add_widget(_make_canvas(fig3))
        self._stats_plot_layout.addWidget(ratio_section)

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

        # Toolbar — three rows (config / actions / messages). // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
        # Row 1: Load Data | σ_loc | Pixel size | Frame rate | Min traj length | R | Read metadata | Legend
        row1 = QHBoxLayout()
        self._adv_stats_load_btn = QPushButton("Load Data")
        self._adv_stats_load_btn.clicked.connect(self._on_adv_stats_load_data)
        row1.addWidget(self._adv_stats_load_btn)

        # σ_loc spinbox — auto-filled on data load from loc.csv / TIFF, user-editable.
        row1.addWidget(QLabel("σ_loc:"))
        self._adv_stats_sigma_loc = QDoubleSpinBox()
        self._adv_stats_sigma_loc.setRange(0.0, 5.0)
        self._adv_stats_sigma_loc.setDecimals(4)
        self._adv_stats_sigma_loc.setSingleStep(0.001)
        self._adv_stats_sigma_loc.setValue(0.0)
        self._adv_stats_sigma_loc.setSuffix(" px")
        self._adv_stats_sigma_loc.setToolTip(
            "Localisation precision σ_loc (pixels) used as plug-in for the corrected\n"
            "Cauchy multi-Δ scan and K_est. Auto-filled on data load from loc.csv\n"
            "(or sibling TIFF, thesis-style annulus). Edit to override."
        )
        row1.addWidget(self._adv_stats_sigma_loc)

        row1.addWidget(QLabel("Pixel size (μm):"))
        self._adv_stats_pixelsize = QDoubleSpinBox()
        self._adv_stats_pixelsize.setRange(0.001, 10.0)
        self._adv_stats_pixelsize.setValue(1.0)
        self._adv_stats_pixelsize.setDecimals(4)
        self._adv_stats_pixelsize.setSingleStep(0.01)
        row1.addWidget(self._adv_stats_pixelsize)

        row1.addWidget(QLabel("Frame rate (s):"))
        self._adv_stats_framerate = QDoubleSpinBox()
        self._adv_stats_framerate.setRange(0.0001, 10.0)
        self._adv_stats_framerate.setValue(1.0)
        self._adv_stats_framerate.setDecimals(4)
        self._adv_stats_framerate.setSingleStep(0.001)
        row1.addWidget(self._adv_stats_framerate)

        row1.addWidget(QLabel("Min traj length:"))
        self._adv_stats_cutoff = QSpinBox()
        self._adv_stats_cutoff.setRange(1, 9999)
        self._adv_stats_cutoff.setValue(3)
        row1.addWidget(self._adv_stats_cutoff)

        row1.addWidget(QLabel("R (τ_exp/Δt):"))
        self._adv_stats_R = QDoubleSpinBox()
        self._adv_stats_R.setRange(0.0, 1.0)
        self._adv_stats_R.setDecimals(4)
        self._adv_stats_R.setSingleStep(0.01)
        self._adv_stats_R.setValue(0.0)
        self._adv_stats_R.setToolTip(
            "Exposure ratio R = exposure time / frame interval ∈ [0, 1].\n"
            "0 = instantaneous exposure (no motion blur). 1 = continuous exposure.\n"
            "Use 'Read metadata from video' to extract from TIFF metadata, or set manually."
        )
        row1.addWidget(self._adv_stats_R)

        self._adv_stats_R_btn = QPushButton("Read metadata from video")
        self._adv_stats_R_btn.setToolTip(
            "Pick a TIFF/ND2 video and read its ImageJ/NIS-Elements metadata.\n"
            "Auto-fills pixel size (μm), frame interval (s), and R = τ_exp/Δt when available."
        )
        self._adv_stats_R_btn.clicked.connect(self._on_adv_stats_read_R)
        row1.addWidget(self._adv_stats_R_btn)

        self._adv_stats_legend_cb = QCheckBox("Legend")
        self._adv_stats_legend_cb.setChecked(True)
        self._adv_stats_legend_cb.stateChanged.connect(self._on_adv_stats_legend_toggled)
        row1.addWidget(self._adv_stats_legend_cb)

        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: Run | Save Plots | Save Data
        row2 = QHBoxLayout()
        self._adv_stats_run_btn = QPushButton("▶ Run Advanced Stats")
        self._adv_stats_run_btn.clicked.connect(self._on_run_adv_stats)
        row2.addWidget(self._adv_stats_run_btn)

        self._adv_stats_save_btn = QPushButton("Save Plots")
        self._adv_stats_save_btn.clicked.connect(self._on_save_adv_stats_plots)
        self._adv_stats_save_btn.setEnabled(False)
        row2.addWidget(self._adv_stats_save_btn)

        self._adv_stats_save_data_btn = QPushButton("Save Data")
        self._adv_stats_save_data_btn.setToolTip(
            "Export multi-Δ scan results as CSV (one file per dataset×state)."
        )
        self._adv_stats_save_data_btn.clicked.connect(self._on_save_adv_stats_data)
        self._adv_stats_save_data_btn.setEnabled(False)
        row2.addWidget(self._adv_stats_save_data_btn)

        row2.addStretch()
        layout.addLayout(row2)

        # Row 3: status messages
        row3 = QHBoxLayout()
        self._adv_stats_status_label = QLabel("")
        self._adv_stats_status_label.setStyleSheet("color:#888;")
        row3.addWidget(self._adv_stats_status_label)
        row3.addStretch()
        layout.addLayout(row3)
        # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29

        self._adv_stats_info_label = QLabel("Click 'Load Data' to load trajectory data.")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._adv_stats_info_label.setStyleSheet("color:#888; padding-left:4px;")
        self._adv_stats_info_label.setWordWrap(True)
        layout.addWidget(self._adv_stats_info_label)

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

    # --- Viz Tab ---------------------------------------------------------------
    _VIZ_COLORMAPS = {  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        "Jet": [  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27
            (0.00, (128, 0, 0)),
            (0.13, (255, 0, 0)),
            (0.37, (255, 255, 0)),
            (0.50, (0, 255, 0)),
            (0.63, (0, 255, 255)),
            (0.88, (0, 0, 255)),
            (1.00, (0, 0, 143)),
        ],
        "Viridis": [
            (0.00, (68, 1, 84)),
            (0.13, (72, 35, 116)),
            (0.25, (64, 67, 135)),
            (0.38, (52, 94, 141)),
            (0.50, (33, 144, 140)),
            (0.63, (42, 176, 110)),
            (0.75, (121, 209, 81)),
            (0.87, (189, 222, 38)),
            (1.00, (253, 231, 37)),
        ],
    }

    def _build_viz_tab(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # Top toolbar: Load Data + Clear (shared across all Viz panels)
        toolbar = QHBoxLayout()
        self._viz_load_btn = QPushButton("Load Data")
        self._viz_load_btn.clicked.connect(self._on_viz_load_data)
        toolbar.addWidget(self._viz_load_btn)
        self._viz_clear_btn = QPushButton("Clear")
        self._viz_clear_btn.clicked.connect(self._on_viz_clear)
        toolbar.addWidget(self._viz_clear_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self._viz_info_label = QLabel("Load _traces.csv + _diffusion.csv to visualise trajectories.")
        self._viz_info_label.setStyleSheet("color:#888; padding-left:4px;")
        self._viz_info_label.setWordWrap(True)
        layout.addWidget(self._viz_info_label)

        # Scrollable area for visualization panels
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:#1e1e1e; border:none;")
        scroll_widget = QWidget()
        self._viz_panel_layout = QVBoxLayout(scroll_widget)
        self._viz_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._viz_panel_layout.setContentsMargins(0, 0, 0, 0)
        self._viz_panel_layout.setSpacing(4)

        # --- Panel 1: Trajectory Color Map ---
        section = CollapsibleSection("Trajectory Color Map")
        section.set_font_size(12)

        # Controls row: Color by, Colormap, Min spin, Max spin
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Color by:"))
        self._viz_color_combo = QComboBox()
        self._viz_color_combo.addItems(["H", "log K"])
        self._viz_color_combo.currentTextChanged.connect(self._on_viz_color_changed)
        self._viz_color_combo.setFixedWidth(80)
        ctrl_row.addWidget(self._viz_color_combo)
        ctrl_row.addWidget(QLabel("Colormap:"))
        self._viz_cmap_combo = QComboBox()
        self._viz_cmap_combo.addItems(list(self._VIZ_COLORMAPS.keys()))
        self._viz_cmap_combo.currentTextChanged.connect(self._on_viz_cmap_changed)
        self._viz_cmap_combo.setFixedWidth(90)
        ctrl_row.addWidget(self._viz_cmap_combo)
        ctrl_row.addSpacing(12)
        ctrl_row.addWidget(QLabel("Min:"))
        self._viz_min_spin = QDoubleSpinBox()
        self._viz_min_spin.setRange(-20.0, 20.0)
        self._viz_min_spin.setDecimals(2)
        self._viz_min_spin.setSingleStep(0.05)
        self._viz_min_spin.setValue(0.0)
        self._viz_min_spin.setFixedWidth(75)
        self._viz_min_spin.valueChanged.connect(self._on_viz_spin_changed)
        ctrl_row.addWidget(self._viz_min_spin)
        ctrl_row.addWidget(QLabel("Max:"))
        self._viz_max_spin = QDoubleSpinBox()
        self._viz_max_spin.setRange(-20.0, 20.0)
        self._viz_max_spin.setDecimals(2)
        self._viz_max_spin.setSingleStep(0.05)
        self._viz_max_spin.setValue(1.0)
        self._viz_max_spin.setFixedWidth(75)
        self._viz_max_spin.valueChanged.connect(self._on_viz_spin_changed)
        ctrl_row.addWidget(self._viz_max_spin)
        ctrl_row.addSpacing(12)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27
        ctrl_row.addWidget(QLabel("Min len:"))
        self._viz_minlen_spin = QSpinBox()
        self._viz_minlen_spin.setRange(1, 9999)
        self._viz_minlen_spin.setValue(3)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27
        self._viz_minlen_spin.setFixedWidth(65)
        self._viz_minlen_spin.valueChanged.connect(self._on_viz_minlen_changed)
        ctrl_row.addWidget(self._viz_minlen_spin)
        self._viz_save_btn = QPushButton("Save")
        self._viz_save_btn.setFixedWidth(55)
        self._viz_save_btn.clicked.connect(self._on_viz_save)
        ctrl_row.addWidget(self._viz_save_btn)
        ctrl_row.addStretch()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27
        section.add_layout(ctrl_row)

        # Canvas row: range slider (left) | canvas (center) | color bar (right)
        canvas_row = QHBoxLayout()
        self._viz_range_slider = VRangeSlider()
        self._viz_range_slider.range_changed.connect(self._on_viz_range_slider_changed)
        canvas_row.addWidget(self._viz_range_slider)
        self._viz_scene = QGraphicsScene()
        self._viz_canvas = QGraphicsView(self._viz_scene)
        self._viz_canvas.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._viz_canvas.setStyleSheet("background:#1a1a1a; border:none;")
        self._viz_canvas.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._viz_canvas.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._viz_canvas.setMinimumHeight(350)
        canvas_row.addWidget(self._viz_canvas, 1)
        self._viz_cbar_scene = QGraphicsScene()
        self._viz_cbar_view = QGraphicsView(self._viz_cbar_scene)
        self._viz_cbar_view.setStyleSheet("background:#1a1a1a; border:none;")
        self._viz_cbar_view.setFixedWidth(80)
        self._viz_cbar_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._viz_cbar_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        canvas_row.addWidget(self._viz_cbar_view)
        section.add_layout(canvas_row)

        self._viz_panel_layout.addWidget(section)

        # (Future visualization panels go here)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)

        # State
        self._viz_traces_df = None
        self._viz_diffusion_df = None
        self._viz_traj_points = []  # list of (xs, ys) per trajectory
        self._viz_traj_values = np.array([])  # H or log K per trajectory
        self._viz_pixmap_item = None
        self._viz_data_min = 0.0  # data range for slider mapping
        self._viz_data_max = 1.0
        # Cached geometry for fast re-render  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._viz_screen_segs = []   # list of (sxs, sys) per trajectory (screen coords)
        self._viz_bounds = None      # (x_min, x_max, y_min, y_max, ML, MT, PW, PH, total_w, total_h)
        self._viz_lut_cache = {}     # colormap name → (256, 3) uint8 numpy array

        return widget
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _viz_build_lut(self, cmap_name):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Build a 256-entry RGB lookup table for a colormap (cached)."""
        if cmap_name in self._viz_lut_cache:
            return self._viz_lut_cache[cmap_name]
        stops = self._VIZ_COLORMAPS.get(cmap_name, self._VIZ_COLORMAPS["Jet"])
        lut = np.zeros((256, 3), dtype=np.uint8)
        for idx in range(256):
            t = idx / 255.0
            for i in range(len(stops) - 1):
                t0, c0 = stops[i]
                t1, c1 = stops[i + 1]
                if t <= t1:
                    f = (t - t0) / max(t1 - t0, 1e-12)
                    lut[idx] = [int(c0[0] + f * (c1[0] - c0[0])),
                                int(c0[1] + f * (c1[1] - c0[1])),
                                int(c0[2] + f * (c1[2] - c0[2]))]
                    break
            else:
                lut[idx] = stops[-1][1]
        self._viz_lut_cache[cmap_name] = lut
        return lut

    def _viz_precompute_geometry(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Precompute spatial bounds and screen coordinates for all trajectories."""
        if not self._viz_traj_points:
            self._viz_bounds = None
            self._viz_screen_segs = []
            return
        all_x = np.concatenate([pts[0] for pts in self._viz_traj_points])
        all_y = np.concatenate([pts[1] for pts in self._viz_traj_points])
        pad_x = max((all_x.max() - all_x.min()) * 0.05, 1.0)
        pad_y = max((all_y.max() - all_y.min()) * 0.05, 1.0)
        x_min, x_max = float(all_x.min() - pad_x), float(all_x.max() + pad_x)
        y_min, y_max = float(all_y.min() - pad_y), float(all_y.max() + pad_y)
        # Enforce equal range on both axes, centered on each axis's own midpoint
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        half_range = max(x_max - x_min, y_max - y_min) / 2
        x_min, x_max = x_center - half_range, x_center + half_range
        y_min, y_max = y_center - half_range, y_center + half_range
        ML, MT, PW, PH = 120, 60, 1000, 1000  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27
        MR, MB = 60, 100
        total_w = ML + PW + MR
        total_h = MT + PH + MB
        self._viz_bounds = (x_min, x_max, y_min, y_max, ML, MT, PW, PH, total_w, total_h)
        segs = []
        for xs, ys in self._viz_traj_points:
            sxs = ML + (xs - x_min) / (x_max - x_min) * PW
            sys_ = MT + (ys - y_min) / (y_max - y_min) * PH
            segs.append((sxs.astype(np.float32), sys_.astype(np.float32)))
        self._viz_screen_segs = segs

    def _on_viz_load_data(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Load single dataset for Viz tab — requires both traces and diffusion."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FreeTrace output CSV", "",
            "CSV files (*_traces.csv *_diffusion.csv);;All files (*)")
        if not path:
            return
        try:
            if '_traces.csv' in path:
                traces_path = path
                diffusion_path = path.replace('_traces.csv', '_diffusion.csv')
            elif '_diffusion.csv' in path:
                diffusion_path = path
                traces_path = path.replace('_diffusion.csv', '_traces.csv')
            else:
                QMessageBox.warning(self, "Invalid file",
                                    "Select a _traces.csv or _diffusion.csv file.")
                return

            if not os.path.exists(traces_path):
                QMessageBox.warning(self, "File not found", f"Traces file not found:\n{traces_path}")
                return
            if not os.path.exists(diffusion_path):
                QMessageBox.warning(self, "File not found",
                                    f"Diffusion file not found:\n{diffusion_path}\n"
                                    "Both _traces.csv and _diffusion.csv are required for Viz.")
                return

            tdf = pd.read_csv(traces_path)
            ddf = pd.read_csv(diffusion_path)
            if not {'traj_idx', 'frame', 'x', 'y'}.issubset(tdf.columns):
                QMessageBox.warning(self, "Invalid traces", "Missing required columns.")
                return
            if not {'traj_idx', 'H', 'K'}.issubset(ddf.columns):
                QMessageBox.warning(self, "Invalid diffusion", "Missing H/K columns.")
                return

            self._viz_traces_df = tdf
            self._viz_diffusion_df = ddf

            # Build per-trajectory point arrays and H/K values
            traj_points = []
            grouped = tdf.groupby('traj_idx')
            traj_ids = sorted(grouped.groups.keys())
            h_vals, k_vals = [], []
            ddf_indexed = ddf.set_index('traj_idx')
            for tid in traj_ids:
                grp = grouped.get_group(tid)
                traj_points.append((grp['x'].values, grp['y'].values))
                if tid in ddf_indexed.index:
                    row = ddf_indexed.loc[tid]
                    h_vals.append(float(row['H']) if np.isscalar(row['H']) else float(row['H'].iloc[0]))
                    k_vals.append(float(row['K']) if np.isscalar(row['K']) else float(row['K'].iloc[0]))
                else:
                    h_vals.append(np.nan)
                    k_vals.append(np.nan)

            self._viz_traj_points = traj_points
            self._viz_H = np.array(h_vals)
            self._viz_K = np.array(k_vals)

            fname = os.path.basename(traces_path).replace('_traces.csv', '')
            n_traj = len(traj_ids)
            self._viz_info_label.setText(f"Loaded {n_traj} trajectories from '{fname}'.")

            # Precompute screen coordinates and draw full scene
            self._viz_precompute_geometry()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            self._viz_set_default_range()
            self._viz_render_full()
        except Exception as e:
            QMessageBox.critical(self, "Error loading data", str(e))
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _viz_set_default_range(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Set min/max spinboxes and range slider to data range."""
        vals = self._viz_get_values()
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            return
        dmin = float(np.floor(valid.min() * 20) / 20)
        dmax = float(np.ceil(valid.max() * 20) / 20)
        self._viz_data_min = dmin
        self._viz_data_max = dmax
        # Default display range: 2.5–97.5 percentile
        q_low = float(np.percentile(valid, 2.5))
        q_high = float(np.percentile(valid, 97.5))
        rng = dmax - dmin
        low_frac = (q_low - dmin) / rng if rng > 1e-12 else 0.0
        high_frac = (q_high - dmin) / rng if rng > 1e-12 else 1.0
        # Block signals during bulk update
        self._viz_min_spin.blockSignals(True)
        self._viz_max_spin.blockSignals(True)
        self._viz_min_spin.setValue(q_low)
        self._viz_max_spin.setValue(q_high)
        self._viz_min_spin.blockSignals(False)
        self._viz_max_spin.blockSignals(False)
        self._viz_range_slider.blockSignals(True)
        self._viz_range_slider.set_range(low_frac, high_frac)
        self._viz_range_slider.blockSignals(False)

    def _on_viz_spin_changed(self, _val):
        """Spinbox changed → sync range slider and re-render."""
        if not hasattr(self, '_viz_H') or len(self._viz_H) == 0:
            return
        rng = self._viz_data_max - self._viz_data_min
        if rng > 1e-12:
            low_frac = (self._viz_min_spin.value() - self._viz_data_min) / rng
            high_frac = (self._viz_max_spin.value() - self._viz_data_min) / rng
            self._viz_range_slider.blockSignals(True)
            self._viz_range_slider.set_range(low_frac, high_frac)
            self._viz_range_slider.blockSignals(False)
        self._viz_render_dynamic()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _on_viz_range_slider_changed(self, low_frac, high_frac):
        """Range slider dragged → sync spinboxes and re-render."""
        rng = self._viz_data_max - self._viz_data_min
        vmin = self._viz_data_min + low_frac * rng
        vmax = self._viz_data_min + high_frac * rng
        self._viz_min_spin.blockSignals(True)
        self._viz_max_spin.blockSignals(True)
        self._viz_min_spin.setValue(vmin)
        self._viz_max_spin.setValue(vmax)
        self._viz_min_spin.blockSignals(False)
        self._viz_max_spin.blockSignals(False)
        self._viz_render_dynamic()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _viz_get_values(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27
        """Return per-trajectory values based on current color mode, with length filter applied."""
        mode = self._viz_color_combo.currentText()
        if mode == "H":
            vals = self._viz_H.copy() if hasattr(self, '_viz_H') else np.array([])
        else:  # log K
            k = self._viz_K if hasattr(self, '_viz_K') else np.array([])
            if len(k) == 0:
                return k
            vals = np.log10(np.clip(k, 1e-20, None))
        # Apply min length filter: set short trajectories to NaN
        min_len = self._viz_minlen_spin.value() if hasattr(self, '_viz_minlen_spin') else 1
        if min_len > 1 and hasattr(self, '_viz_traj_points') and len(vals) == len(self._viz_traj_points):
            for i, (xs, _ys) in enumerate(self._viz_traj_points):
                if len(xs) < min_len:
                    vals[i] = np.nan
        return vals  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27

    def _on_viz_color_changed(self, _text):
        if not hasattr(self, '_viz_H') or len(self._viz_H) == 0:
            return
        self._viz_set_default_range()
        self._viz_render_full()

    def _on_viz_cmap_changed(self, _text):
        if not hasattr(self, '_viz_H') or len(self._viz_H) == 0:
            return
        self._viz_render_dynamic()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _on_viz_minlen_changed(self, _val):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27
        if not hasattr(self, '_viz_H') or len(self._viz_H) == 0:
            return
        self._viz_set_default_range()
        self._viz_render_full()

    def _on_viz_clear(self):
        self._viz_scene.clear()
        self._viz_cbar_scene.clear()
        self._viz_traj_points = []
        self._viz_screen_segs = []  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._viz_bounds = None
        self._viz_H = np.array([])
        self._viz_K = np.array([])
        self._viz_traces_df = None
        self._viz_diffusion_df = None
        self._viz_pixmap_item = None
        self._viz_info_label.setText("Load _traces.csv + _diffusion.csv to visualise trajectories coloured by H or K.")

    def _on_viz_save(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Save the trajectory color map as a high-resolution PNG with transparent background."""
        if not self._viz_bounds or not self._viz_screen_segs:
            QMessageBox.information(self, "Nothing to save", "Load data first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Trajectory Map", "", "PNG images (*.png)")
        if not path:
            return
        if not path.lower().endswith('.png'):
            path += '.png'

        vals = self._viz_get_values()
        vmin = self._viz_min_spin.value()
        vmax = self._viz_max_spin.value()
        x_min, x_max, y_min, y_max, ML, MT, PW, PH, total_w, total_h = self._viz_bounds

        # Layout: trajectories on left, gap, colorbar + ticks on right
        scale = math.ceil(2048 / PH)  # plot area height ≥ 2048
        sPW, sPH = PW * scale, PH * scale
        cbar_gap = int(20 * scale)
        cbar_w = int(20 * scale)
        tick_w = int(60 * scale)  # space for tick labels
        title_h = int(20 * scale)
        out_w = sPW + cbar_gap + cbar_w + tick_w
        out_h = title_h + sPH

        cmap_name = self._viz_cmap_combo.currentText()
        lut = self._viz_build_lut(cmap_name)

        img = QImage(out_w, out_h, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))  # transparent
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Trajectories — map data coords directly to [0, sPW] x [title_h, title_h+sPH]
        val_range = max(vmax - vmin, 1e-12)
        t_arr = np.clip((vals - vmin) / val_range, 0.0, 1.0)
        idx_arr = np.clip((t_arr * 255).astype(int), 0, 255)
        nan_mask = np.isnan(vals)
        x_data_range = max(x_max - x_min, 1e-12)
        y_data_range = max(y_max - y_min, 1e-12)

        for i, (sxs_orig, sys_orig) in enumerate(self._viz_screen_segs):
            if nan_mask[i] if i < len(nan_mask) else True:
                color = QColor(80, 80, 80, 60)
            else:
                r, g, b = int(lut[idx_arr[i], 0]), int(lut[idx_arr[i], 1]), int(lut[idx_arr[i], 2])
                color = QColor(r, g, b, 200)
            pen = QPen(color, max(1.0, 0.5 * scale))
            painter.setPen(pen)
            n = len(sxs_orig)
            if n < 2:
                continue
            # Re-map from scene coords (ML-based) to export coords (0-based)
            exs = (sxs_orig - ML) / PW * sPW
            eys = (sys_orig - MT) / PH * sPH + title_h
            tpath = QPainterPath()
            tpath.moveTo(float(exs[0]), float(eys[0]))
            for j in range(1, n):
                tpath.lineTo(float(exs[j]), float(eys[j]))
            painter.drawPath(tpath)

        # Colorbar — 60% of plot height, vertically centred  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        cbar_h = int(sPH * 0.6)
        cbar_x = sPW + cbar_gap
        cbar_y = title_h + (sPH - cbar_h) // 2
        row_indices = np.clip(((1.0 - np.arange(cbar_h) / max(cbar_h - 1, 1)) * 255).astype(int), 0, 255)
        rgb = lut[row_indices]
        row_argb = np.zeros((cbar_h, 4), dtype=np.uint8)
        row_argb[:, 0] = rgb[:, 2]
        row_argb[:, 1] = rgb[:, 1]
        row_argb[:, 2] = rgb[:, 0]
        row_argb[:, 3] = 255
        img_data = np.tile(row_argb, (1, cbar_w)).reshape(cbar_h, cbar_w, 4)
        cbar_img = QImage(img_data.tobytes(), cbar_w, cbar_h, cbar_w * 4, QImage.Format.Format_ARGB32)
        painter.drawImage(cbar_x, cbar_y, cbar_img.copy())

        # Colorbar border
        painter.setPen(QPen(QColor(180, 180, 180), max(1, scale)))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawRect(cbar_x, cbar_y, cbar_w, cbar_h)

        # Colorbar ticks
        pen_text = QColor(180, 180, 180)
        font = painter.font()
        font.setPixelSize(max(12, int(12 * scale)))
        painter.setFont(font)
        painter.setPen(QPen(pen_text, 1))
        n_ticks = 5
        for ti in range(n_ticks + 1):
            frac = ti / n_ticks
            ty = cbar_y + int(cbar_h * (1.0 - frac))
            val = vmin + (vmax - vmin) * frac
            painter.drawLine(QPointF(cbar_x + cbar_w, ty),
                             QPointF(cbar_x + cbar_w + 3 * scale, ty))
            painter.drawText(QPointF(cbar_x + cbar_w + 5 * scale, ty + 4 * scale), f"{val:.2f}")

        # Colorbar title
        bold_font = painter.font()
        bold_font.setBold(True)
        painter.setFont(bold_font)
        mode = self._viz_color_combo.currentText()
        painter.drawText(QPointF(cbar_x, cbar_y - 5 * scale), mode)

        painter.end()
        img.save(path)
        self._viz_info_label.setText(f"Saved {out_w}×{out_h} px → {os.path.basename(path)}")
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _viz_value_to_color(self, val, vmin, vmax):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Map a scalar value to a color using the selected colormap."""
        if np.isnan(val):
            return QColor(80, 80, 80, 60)
        t = max(0.0, min(1.0, (val - vmin) / max(vmax - vmin, 1e-12)))
        cmap_name = self._viz_cmap_combo.currentText()
        stops = self._VIZ_COLORMAPS.get(cmap_name, self._VIZ_COLORMAPS["Jet"])
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t <= t1:
                f = (t - t0) / max(t1 - t0, 1e-12)
                r = int(c0[0] + f * (c1[0] - c0[0]))
                g = int(c0[1] + f * (c1[1] - c0[1]))
                b = int(c0[2] + f * (c1[2] - c0[2]))
                return QColor(r, g, b, 200)
        last = stops[-1][1]
        return QColor(last[0], last[1], last[2], 200)

    def _viz_render_full(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Full render: static scene (grid/axes) + dynamic overlay (trajectories/colorbar)."""
        self._viz_render_static()
        self._viz_render_dynamic()

    def _viz_render_static(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Render static scene elements: background, grid, axes, labels."""
        self._viz_scene.clear()
        self._viz_pixmap_item = None
        if not self._viz_bounds:
            return
        x_min, x_max, y_min, y_max, ML, MT, PW, PH, total_w, total_h = self._viz_bounds
        self._viz_scene.setSceneRect(0, 0, total_w, total_h)

        pen_grid = QPen(QColor(60, 60, 60), 0.5, Qt.PenStyle.DashLine)
        pen_axis = QPen(QColor(150, 150, 150), 1.5)
        pen_text = QColor(180, 180, 180)
        scene_font = QFont()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        scene_font.setPointSize(18)  # scaled for 1000x800 scene

        self._viz_scene.addRect(QRectF(ML, MT, PW, PH),
                                QPen(Qt.PenStyle.NoPen), QBrush(QColor(30, 30, 30)))

        # Use a single grid step for both axes  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27
        x_range = x_max - x_min
        y_range = y_max - y_min
        grid_step = ROICanvas._nice_step(max(x_range, y_range), 8)
        xv = math.ceil(x_min / grid_step) * grid_step
        while xv <= x_max:
            sx = ML + (xv - x_min) / (x_max - x_min) * PW
            if ML <= sx <= ML + PW:
                self._viz_scene.addLine(sx, MT, sx, MT + PH, pen_grid)
                t = self._viz_scene.addSimpleText(f"{xv:.0f}", scene_font)
                t.setBrush(pen_text)
                t.setPos(sx - 20, MT + PH + 8)
            xv += grid_step

        yv = math.ceil(y_min / grid_step) * grid_step
        while yv <= y_max:
            sy = MT + (yv - y_min) / (y_max - y_min) * PH
            if MT <= sy <= MT + PH:
                self._viz_scene.addLine(ML, sy, ML + PW, sy, pen_grid)
                t = self._viz_scene.addSimpleText(f"{yv:.0f}", scene_font)
                t.setBrush(pen_text)
                t.setPos(ML - 80, sy - 12)
            yv += grid_step

        self._viz_scene.addLine(ML, MT + PH, ML + PW, MT + PH, pen_axis)
        self._viz_scene.addLine(ML, MT, ML, MT + PH, pen_axis)

        xl = self._viz_scene.addSimpleText("X (pixels)", scene_font)
        xl.setBrush(pen_text)
        xl.setPos(ML + PW / 2 - 55, MT + PH + 45)
        yl = self._viz_scene.addSimpleText("Y (pixels)", scene_font)
        yl.setBrush(pen_text)
        yl.setRotation(-90)
        yl.setPos(20, MT + PH / 2 + 40)

    def _viz_render_dynamic(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Render only trajectories (pixmap) + colorbar. Uses cached screen coords + LUT."""
        if not self._viz_bounds or not self._viz_screen_segs:
            return
        # Remove old trajectory pixmap if present
        if self._viz_pixmap_item is not None:
            self._viz_scene.removeItem(self._viz_pixmap_item)
            self._viz_pixmap_item = None

        vals = self._viz_get_values()
        vmin = self._viz_min_spin.value()
        vmax = self._viz_max_spin.value()
        x_min, x_max, y_min, y_max, ML, MT, PW, PH, total_w, total_h = self._viz_bounds

        # Build LUT for current colormap
        cmap_name = self._viz_cmap_combo.currentText()
        lut = self._viz_build_lut(cmap_name)

        # Vectorised: map values → LUT indices
        val_range = max(vmax - vmin, 1e-12)
        t_arr = np.clip((vals - vmin) / val_range, 0.0, 1.0)
        idx_arr = np.clip((t_arr * 255).astype(int), 0, 255)
        nan_mask = np.isnan(vals)

        pix = QPixmap(total_w, total_h)
        pix.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        min_len = self._viz_minlen_spin.value() if hasattr(self, '_viz_minlen_spin') else 1  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-27
        for i, (sxs, sys_) in enumerate(self._viz_screen_segs):
            if len(sxs) < min_len:
                continue
            if nan_mask[i] if i < len(nan_mask) else True:
                color = QColor(80, 80, 80, 60)
            else:
                r, g, b = int(lut[idx_arr[i], 0]), int(lut[idx_arr[i], 1]), int(lut[idx_arr[i], 2])
                color = QColor(r, g, b, 200)
            pen = QPen(color, 1.0)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            pen.setCosmetic(True)
            painter.setPen(pen)
            n = len(sxs)
            if n < 2:
                continue
            path = QPainterPath()
            path.moveTo(float(sxs[0]), float(sys_[0]))
            for j in range(1, n):
                path.lineTo(float(sxs[j]), float(sys_[j]))
            painter.drawPath(path)

        painter.end()
        self._viz_pixmap_item = self._viz_scene.addPixmap(pix)
        self._viz_pixmap_item.setZValue(1)

        self._viz_canvas.fitInView(self._viz_scene.sceneRect(),
                                   Qt.AspectRatioMode.KeepAspectRatio)

        self._viz_render_colorbar(vmin, vmax)
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _viz_render_colorbar(self, vmin, vmax):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Render a vertical color bar with tick labels."""
        self._viz_cbar_scene.clear()
        bar_w, bar_h = 20, 300
        margin_top, margin_left = 30, 10
        total_w = margin_left + bar_w + 50
        total_h = margin_top + bar_h + 30
        self._viz_cbar_scene.setSceneRect(0, 0, total_w, total_h)

        # Draw gradient using LUT — numpy→QImage, no Python pixel loop
        cmap_name = self._viz_cmap_combo.currentText()
        lut = self._viz_build_lut(cmap_name)
        row_indices = np.clip(((1.0 - np.arange(bar_h) / max(bar_h - 1, 1)) * 255).astype(int), 0, 255)
        rgb = lut[row_indices]  # (bar_h, 3)
        # Build ARGB32 row data: each pixel = 0xFF_RR_GG_BB (little-endian: BB GG RR FF)
        row_argb = np.zeros((bar_h, 4), dtype=np.uint8)
        row_argb[:, 0] = rgb[:, 2]  # B
        row_argb[:, 1] = rgb[:, 1]  # G
        row_argb[:, 2] = rgb[:, 0]  # R
        row_argb[:, 3] = 255        # A
        # Tile across bar width
        img_data = np.tile(row_argb, (1, bar_w)).reshape(bar_h, bar_w, 4)
        img_bytes = img_data.tobytes()
        img = QImage(img_bytes, bar_w, bar_h, bar_w * 4, QImage.Format.Format_ARGB32)
        pix = QPixmap.fromImage(img.copy())  # copy() detaches from buffer
        pix_item = self._viz_cbar_scene.addPixmap(pix)
        pix_item.setPos(margin_left, margin_top)

        # Border
        self._viz_cbar_scene.addRect(QRectF(margin_left, margin_top, bar_w, bar_h),
                                     QPen(QColor(150, 150, 150), 1.0))

        # Tick labels
        pen_text = QColor(180, 180, 180)
        n_ticks = 5
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            y = margin_top + bar_h * (1.0 - frac)
            val = vmin + (vmax - vmin) * frac
            label = f"{val:.2f}"
            t = self._viz_cbar_scene.addSimpleText(label)
            t.setBrush(pen_text)
            t.setPos(margin_left + bar_w + 4, y - 6)
            # Tick mark
            self._viz_cbar_scene.addLine(margin_left + bar_w, y,
                                         margin_left + bar_w + 3, y,
                                         QPen(QColor(150, 150, 150), 1.0))

        # Title
        mode = self._viz_color_combo.currentText()
        title = self._viz_cbar_scene.addSimpleText(mode)
        title.setBrush(QColor(200, 200, 200))
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        title.setPos(margin_left, 5)

        self._viz_cbar_view.fitInView(self._viz_cbar_scene.sceneRect(),
                                      Qt.AspectRatioMode.KeepAspectRatio)
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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
        # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
        # Multi-dataset: reset σ_loc spinbox to 0 so each dataset uses its own auto rms
        # (per-dataset σ_loc plug-in). Single-dataset: spinbox already auto-filled at load.
        if n > 1:
            try:
                self._adv_stats_sigma_loc.setValue(0.0)
            except Exception:
                pass
        # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
        if n > 0:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            total_traj = sum(ds['traces_df']['traj_idx'].nunique() for ds in self._adv_stats_datasets)
            # Per-dataset CRLB-column availability + σ_loc summary (added 2026-04-28).
            # Corrected Cauchy multi-Δ requires the bg_median/bg_var/integrated_flux columns,
            # which only the latest FreeTrace localisation produces.
            datasets_with_crlb = [ds for ds in self._adv_stats_datasets if ds.get('has_crlb_cols')]
            n_with_crlb = len(datasets_with_crlb)
            if n_with_crlb == 0:
                crlb_msg = "  No CRLB columns found — re-run localisation with current FreeTrace to enable corrected Cauchy."
            else:
                # Aggregate σ_loc across the videos that have it
                all_sigma_px = np.concatenate([
                    ds['sigma_loc_px'][np.isfinite(ds['sigma_loc_px'])]
                    for ds in datasets_with_crlb if ds.get('sigma_loc_px') is not None
                ]) if any(ds.get('sigma_loc_px') is not None for ds in datasets_with_crlb) else np.array([])
                px_um = self._adv_stats_pixelsize.value() if hasattr(self, '_adv_stats_pixelsize') else None
                if all_sigma_px.size > 0 and px_um:
                    med_nm = float(np.median(all_sigma_px)) * px_um * 1000.0
                    rms_nm = float(np.sqrt(np.mean(all_sigma_px ** 2))) * px_um * 1000.0
                    sigma_str = f"σ_loc median={med_nm:.1f} nm, RMS={rms_nm:.1f} nm"
                else:
                    sigma_str = "σ_loc unavailable"
                if n_with_crlb == n:
                    crlb_msg = f"  CRLB columns present — corrected Cauchy enabled ({sigma_str})."
                else:
                    crlb_msg = f"  CRLB columns present in {n_with_crlb}/{n} videos ({sigma_str})."
            if n == 1:
                fname = self._adv_stats_datasets[0]['video_name']
                self._adv_stats_info_label.setText(
                    f"Loaded {total_traj} trajectories from '{fname}'.{crlb_msg}")
            else:
                self._adv_stats_info_label.setText(
                    f"Loaded {total_traj} trajectories from {n} videos.{crlb_msg}")
            self._on_run_adv_stats()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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

            # Look for matching _loc.csv to harvest CRLB columns. // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
            # ROI and classification exports produce subset traces files like
            # <v>_roi_2_traces.csv or <v>_region_1_traces.csv. The loc file lives at
            # the parent <v>_loc.csv, so we try the direct sibling first then walk
            # back the filename by stripping trailing "_<token>" up to 3 levels.
            base_no_ext = traces_path[:-len('_traces.csv')]
            loc_candidates = [base_no_ext + '_loc.csv']
            cur = base_no_ext
            for _ in range(3):
                if '_' not in os.path.basename(cur):
                    break
                cur = cur.rsplit('_', 1)[0]
                loc_candidates.append(cur + '_loc.csv')
            loc_path = next((p for p in loc_candidates if os.path.exists(p)), loc_candidates[0])
            loc_df = None
            has_crlb_cols = False
            sigma_loc_px = None
            sigma_loc_median_px = None
            sigma_loc_rms_px = None
            I_tot_arr = None  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
            psf_sigma_px_arr = None
            if os.path.exists(loc_path):
                try:
                    loc_df = pd.read_csv(loc_path)
                    has_crlb_cols = {'bg_median', 'bg_var', 'integrated_flux'}.issubset(loc_df.columns)
                    if has_crlb_cols:
                        # If the traces file is a subset of the parent loc file (ROI / class
                        # export), filter loc rows to only those that survived into the
                        # current traces — match by (frame, round(x,3), round(y,3)).
                        # Falls back to all loc rows if the join keeps too few rows.
                        if loc_path != base_no_ext + '_loc.csv':
                            try:
                                key_traces = set(zip(
                                    traces_df['frame'].astype(int).tolist(),
                                    np.round(traces_df['x'].to_numpy(), 3).tolist(),
                                    np.round(traces_df['y'].to_numpy(), 3).tolist(),
                                ))
                                key_loc = list(zip(
                                    loc_df['frame'].astype(int).tolist(),
                                    np.round(loc_df['x'].to_numpy(), 3).tolist(),
                                    np.round(loc_df['y'].to_numpy(), 3).tolist(),
                                ))
                                mask = np.array([k in key_traces for k in key_loc], dtype=bool)
                                if mask.sum() >= max(10, int(0.5 * len(traces_df))):
                                    loc_df = loc_df.iloc[mask].reset_index(drop=True)
                            except Exception:
                                pass
                        # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                        # Override loc.csv bg_var / integrated_flux with thesis-style annulus
                        # values when a raw TIFF is alongside, so σ_loc matches the thesis CRLB.
                        crlb_source = "loc.csv"
                        tif_path = _find_sibling_tif(loc_path, traces_path, video_name)
                        if tif_path is not None:
                            try:
                                rec = _recompute_bg_var_and_flux_from_tif(loc_df, tif_path)
                                loc_df = loc_df.copy()
                                loc_df['bg_var'] = rec['bg_var']
                                loc_df['bg_median'] = rec['bg_median']
                                loc_df['integrated_flux'] = rec['integrated_flux']
                                crlb_source = f"TIFF ({os.path.basename(tif_path)})"
                            except Exception as _e:
                                crlb_source = f"loc.csv (TIFF recompute failed: {_e})"
                        # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                        sl_data = _compute_sigma_loc_per_spot(loc_df)
                        sigma_loc_px = sl_data['sigma_loc_px']
                        I_tot_arr = sl_data['I_tot']
                        psf_sigma_px_arr = sl_data['psf_sigma_px']
                        finite = np.isfinite(sigma_loc_px)
                        if finite.any():
                            sigma_loc_median_px = float(np.median(sigma_loc_px[finite]))
                            # RMS pooling per feedback_sigma_loc_plugin.md
                            sigma_loc_rms_px = float(np.sqrt(np.mean(sigma_loc_px[finite] ** 2)))
                            # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                            # Pre-fill the toolbar σ_loc spinbox with the freshly-computed rms.
                            try:
                                self._adv_stats_sigma_loc.setValue(float(sigma_loc_rms_px))
                            except Exception:
                                pass
                            # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                except Exception:
                    loc_df = None

            self._adv_stats_datasets.append({
                'video_name': video_name,
                'traces_path': traces_path,
                'traces_df': traces_df,
                'loc_path': loc_path if loc_df is not None else None,
                'loc_df': loc_df,
                'has_crlb_cols': has_crlb_cols,
                'sigma_loc_px': sigma_loc_px,                  # per-spot, may be None
                'sigma_loc_median_px': sigma_loc_median_px,
                'sigma_loc_rms_px': sigma_loc_rms_px,
                'I_tot_arr': I_tot_arr,                        # capture-corrected photons per spot
                'psf_sigma_px_arr': psf_sigma_px_arr,           # PSF size per spot (for Thompson line)
                # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                'crlb_source': crlb_source if has_crlb_cols else None,
                # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
            })
        except Exception as e:
            QMessageBox.critical(self, "Error loading data", str(e))

    def _on_adv_stats_read_R(self):  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
        """Pick a video file and auto-fill pixel size, frame interval, and R from its metadata."""
        # Default starting dir: dir of first loaded dataset, or cwd
        start_dir = ""
        if self._adv_stats_datasets:
            start_dir = os.path.dirname(self._adv_stats_datasets[0]['traces_path'])
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video file (TIFF / ND2)", start_dir,
            "Microscopy videos (*.tif *.tiff *.nd2);;All files (*)",
        )
        if not path:
            return
        meta = _read_metadata_from_video(path)  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28 — dispatches TIFF/ND2
        # All-or-nothing per user request: only update if pixel size, frame interval, // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
        # AND R are all present. Otherwise show a warning listing what's missing and
        # leave all toolbar fields untouched.
        missing = []
        if meta.get('pixel_size_um') is None:
            missing.append('pixel size')
        if meta.get('finterval_s') is None:
            missing.append('frame interval')
        if meta.get('R') is None:
            missing.append('R (exposure/Δt)')
        msg = meta.get('message') or "No usable metadata."
        if missing:
            QMessageBox.warning(
                self, "Read metadata from video",
                f"{os.path.basename(path)}: {msg}\n\n"
                f"Missing: {', '.join(missing)}.\n\n"
                f"No fields were updated. Enter values manually.",
            )
            return
        self._adv_stats_pixelsize.setValue(float(meta['pixel_size_um']))
        # The toolbar field is labelled "Frame rate (s)" but stores the frame interval in seconds.
        self._adv_stats_framerate.setValue(float(meta['finterval_s']))
        self._adv_stats_R.setValue(float(meta['R']))
        self._adv_stats_status_label.setText(
            f"Read {os.path.basename(path)}: {msg} → updated pixel size, frame interval, R."
        )
        # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
        # Auto re-run Adv Stats now that the metadata is loaded, but only if a dataset is
        # already loaded and no run is currently in progress.
        if self._adv_stats_datasets and not (
            self._adv_stats_worker is not None and self._adv_stats_worker.isRunning()
        ):
            self._on_run_adv_stats()
        # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29

    def _on_run_adv_stats(self):
        """Run Advanced Stats preprocessing in background thread."""
        if not self._adv_stats_datasets:
            QMessageBox.warning(self, "No data", "Load data first.")
            return
        if self._adv_stats_worker is not None and self._adv_stats_worker.isRunning():
            return

        dataset_tuples = []
        sigma_loc_rms_per_dataset = []
        sigma_loc_spin = float(self._adv_stats_sigma_loc.value())
        # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
        # When more than one dataset is loaded, pool them into a single "merged" run
        # (trajectories concatenated with offset traj_idx; joint σ_loc rms across all
        # per-dataset values, unless the spinbox sets an explicit override). Single
        # dataset path is the standard case — no merging needed.
        if len(self._adv_stats_datasets) > 1:
            tid_offset = 0
            merged_traces = []
            for ds in self._adv_stats_datasets:
                tdf = ds['traces_df'].copy()
                if 'state' not in tdf.columns:
                    tdf['state'] = 0
                tdf['traj_idx'] = tdf['traj_idx'].astype(int) + tid_offset
                tid_offset = int(tdf['traj_idx'].max()) + 1
                merged_traces.append(tdf)
            merged = pd.concat(merged_traces, ignore_index=True)
            dataset_tuples.append(('merged', merged))
            if sigma_loc_spin > 0:
                sigma_loc_rms_per_dataset.append(sigma_loc_spin)
            else:
                vals = [float(ds.get('sigma_loc_rms_px')) for ds in self._adv_stats_datasets
                        if ds.get('sigma_loc_rms_px') is not None
                        and np.isfinite(ds.get('sigma_loc_rms_px'))]
                if vals:
                    sigma_loc_rms_per_dataset.append(float(np.sqrt(np.mean(np.asarray(vals) ** 2))))
                else:
                    sigma_loc_rms_per_dataset.append(None)
        else:
            for ds in self._adv_stats_datasets:
                tdf = ds['traces_df'].copy()
                if 'state' not in tdf.columns:
                    tdf['state'] = 0
                dataset_tuples.append((ds['video_name'], tdf))
                if sigma_loc_spin > 0:
                    sigma_loc_rms_per_dataset.append(sigma_loc_spin)
                else:
                    sigma_loc_rms_per_dataset.append(ds.get('sigma_loc_rms_px'))

        self._adv_stats_run_btn.setEnabled(False)
        merged_run = len(self._adv_stats_datasets) > 1
        self._adv_stats_status_label.setText(
            "Processing (merged)..." if merged_run else "Processing...")
        self._adv_stats_worker = AdvStatsWorker(
            dataset_tuples, self._adv_stats_pixelsize.value(),
            self._adv_stats_framerate.value(), self._adv_stats_cutoff.value(),
            R=self._adv_stats_R.value(),  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
            sigma_loc_rms_per_dataset=sigma_loc_rms_per_dataset,
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
        # Enable scan-data export only when at least one dataset has multi-Δ results.  // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
        self._adv_stats_save_data_btn.setEnabled(
            any(r.get('multi_delta_per_state') for r in results_list)
        )
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

        def _make_canvas(fig, save_name=None):  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(350)
            if save_name is not None:
                canvas._save_name = save_name
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
            # ---- 1D Displacement (Δx, Δy) — Gaussian distortion diagnostic ---- // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
            # TA-EA-SD (linear + log-log) and 1D Ratio + Cauchy fit moved to the Basic Stats tab
            # since they are noise-free / empirical analyses. Advanced Stats keeps 1D Displacement
            # because the Gaussian-fit deviation from a true Gaussian is a useful distortion
            # diagnostic. Phase 3 will add the corrected-Cauchy multi-Δ 6-panel figure here.
            section2 = CollapsibleSection("1D Displacement (Δx, Δy) — consecutive frames only, Δt = 1")

            disp_panel_idx = 0  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
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
                section2.add_widget(_make_canvas(fig2, save_name=f'displacement_1d_{disp_panel_idx}'))  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                disp_panel_idx += 1
            # Section 2 addWidget moved to the end — see panel-order block. // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29

            # ---- Corrected Cauchy multi-Δ scan: Ĥ(Δ) per state ---- // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
            # Phase 3 step 3d: noise-aware H estimator from per-Δ ratio chains.
            # Skipped per-dataset when σ_loc could not be computed (e.g. _loc.csv lacks
            # bg_median/bg_var/integrated_flux), or when MSD(τ=1) ≤ ~1.5·2σ_loc² (noise
            # floor — the corrected fit cannot recover H from pure noise).
            # Always render a diagnostic panel showing MSD(τ=1) vs the noise floor when
            # σ_loc is available, so the user can see WHY a scan was skipped.
            has_any_diag = any(r.get('noise_floor_per_state') for r in results_list)
            if has_any_diag:
                section_diag = CollapsibleSection("Noise-floor diagnostic — σ_loc(I) scatter + MSD(τ=1) vs 2σ_loc²")
                # Look up per-dataset per-spot arrays from self._adv_stats_datasets by name. // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                ds_by_name = {ds['video_name']: ds for ds in getattr(self, '_adv_stats_datasets', [])}
                px_um = self._adv_stats_pixelsize.value() if hasattr(self, '_adv_stats_pixelsize') else 1.0
                pix_nm = float(px_um) * 1000.0
                for r in results_list:
                    nf = r.get('noise_floor_per_state') or {}
                    if not nf:
                        continue
                    name = r['name']
                    safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
                    ds = ds_by_name.get(name) or {}
                    sigma_loc_px_arr = ds.get('sigma_loc_px')
                    I_tot_arr = ds.get('I_tot_arr')
                    psf_sigma_px_arr = ds.get('psf_sigma_px_arr')
                    bg_var_arr = ds['loc_df']['bg_var'].to_numpy() if (ds.get('loc_df') is not None and 'bg_var' in ds['loc_df'].columns) else None

                    # 3-column layout: scatter | bar | stats text
                    fig_nf = Figure(figsize=(13, 4.5), dpi=100, constrained_layout=True)
                    gs = fig_nf.add_gridspec(1, 3, width_ratios=[5, 4, 1], wspace=0.05)
                    ax_sc = fig_nf.add_subplot(gs[0])
                    ax_bar = fig_nf.add_subplot(gs[1])
                    ax_nf_stats = fig_nf.add_subplot(gs[2])
                    ax_nf_stats.axis('off')

                    # ---- LEFT: σ_loc (nm) vs I_tot (ADU) scatter ----
                    if sigma_loc_px_arr is not None and I_tot_arr is not None:
                        m = np.isfinite(sigma_loc_px_arr) & np.isfinite(I_tot_arr) & (I_tot_arr > 0)
                        if m.any():
                            sig_nm = sigma_loc_px_arr[m] * pix_nm
                            I_use = I_tot_arr[m]
                            ax_sc.scatter(I_use, sig_nm, c='tab:blue', s=2, alpha=0.5,
                                           label=f'spots (n={int(m.sum())})')
                            # Thompson theory line: σ² = (s² + a²/12)/N + 8π s⁴ b² / (a² N²)
                            if psf_sigma_px_arr is not None and bg_var_arr is not None:
                                m2 = m & np.isfinite(psf_sigma_px_arr) & np.isfinite(bg_var_arr) & (bg_var_arr > 0)
                                if m2.sum() > 10:
                                    s_nm_med = float(np.median(psf_sigma_px_arr[m2])) * pix_nm
                                    b_med = float(np.median(bg_var_arr[m2]))
                                    a = pix_nm
                                    Ngrid = np.logspace(np.log10(max(I_use.min(), 1.0)),
                                                        np.log10(I_use.max()), 200)
                                    th = np.sqrt((s_nm_med ** 2 + a ** 2 / 12.0) / Ngrid
                                                 + 8.0 * np.pi * s_nm_med ** 4 * b_med
                                                 / (a ** 2 * Ngrid ** 2))
                                    ax_sc.plot(Ngrid, th, 'r-', lw=1.6,
                                                label=f'Thompson (s={s_nm_med:.0f} nm, b²={b_med:.0f})')
                                    ax_sc.plot(Ngrid, s_nm_med / np.sqrt(Ngrid), 'k--', lw=0.8,
                                                label='shot-noise s/√N')
                            ax_sc.set_xscale('log'); ax_sc.set_yscale('log')
                            ax_sc.set_xlabel('I (ADU)')
                            ax_sc.set_ylabel('σ_loc (nm)')
                            ax_sc.set_title(f'Per-spot CRLB σ_loc vs I — {name}')
                            ax_sc.grid(True, alpha=0.3, which='both')
                            ax_sc.legend(fontsize=7, loc='best')
                        else:
                            ax_sc.text(0.5, 0.5, 'no valid spots', ha='center', va='center',
                                        transform=ax_sc.transAxes, color='#888')
                            ax_sc.axis('off')
                    else:
                        ax_sc.text(0.5, 0.5, 'σ_loc per-spot data unavailable',
                                    ha='center', va='center', transform=ax_sc.transAxes, color='#888')
                        ax_sc.axis('off')

                    # ---- RIGHT: MSD(τ) log-log per state with horizontal noise-floor 2σ² ----
                    # Replaces the earlier bar chart per user request: same diagnostic but
                    # in the thesis style — TA-EA MSD curve + dashed black noise floor line.
                    states = sorted(nf.keys())
                    state_pal = sns.color_palette('tab10', n_colors=max(len(r['total_states']), 1))
                    nf_lines = [(f'σ_loc rms = {r.get("sigma_loc_rms_px"):.4g} px', '#aaaaaa', None, None)]
                    sigma_loc_rms_px = r.get('sigma_loc_rms_px') or 0.0
                    noise_floor = 2.0 * float(sigma_loc_rms_px) ** 2
                    MAX_TAU_DIAG = 30
                    # Need raw traces — pull from the dataset record.
                    traces_df_full = ds.get('traces_df')
                    any_curve = False
                    md_keys = set((r.get('multi_delta_per_state') or {}).keys())  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                    for st in states:
                        msd1, _noise, snr = nf[st]
                        try:
                            color = state_pal[r['total_states'].index(st) % len(state_pal)]
                        except Exception:
                            color = state_pal[0]
                        ok = '✓ scan ran' if st in md_keys else '✗ scan skipped (no diffusion signal)'
                        nf_lines.append((f'State {st}: MSD(τ=1)={msd1:.4g}  S/N={snr:.2f}  {ok}',
                                         color, None, None))
                        # Compute TA-EA MSD up to MAX_TAU_DIAG
                        if traces_df_full is None:
                            continue
                        tdf = traces_df_full
                        if 'state' in tdf.columns:
                            tdf = tdf[tdf['state'] == st]
                        if tdf is None or len(tdf) == 0:
                            continue
                        trajs_st = []
                        for _, g in tdf.groupby('traj_idx'):
                            g = g.sort_values('frame')
                            if len(g) >= 2:
                                trajs_st.append(g[['frame', 'x', 'y']].to_numpy(dtype=np.float64))
                        if not trajs_st:
                            continue
                        taus = np.arange(1, MAX_TAU_DIAG + 1)
                        msd = np.full(MAX_TAU_DIAG, np.nan)
                        sem = np.full(MAX_TAU_DIAG, np.nan)
                        for ti, tau in enumerate(taus):
                            sds = []
                            for tr in trajs_st:
                                if len(tr) <= tau:
                                    continue
                                dx = tr[tau:, 1] - tr[:-tau, 1]
                                dy = tr[tau:, 2] - tr[:-tau, 2]
                                sds.append(np.concatenate([dx ** 2, dy ** 2]))
                            if sds:
                                a = np.concatenate(sds)
                                msd[ti] = float(a.mean())
                                sem[ti] = float(a.std(ddof=1) / np.sqrt(max(len(a), 1)))
                        m = np.isfinite(msd) & (msd > 0)
                        if not m.any():
                            continue
                        label = f'State {st} TA-EA MSD' if len(states) > 1 else 'TA-EA MSD'
                        ax_bar.errorbar(taus[m], msd[m], yerr=sem[m], fmt='o-',
                                         color=color, lw=1.2, ms=4, capsize=2, label=label)
                        any_curve = True
                    if noise_floor > 0:
                        ax_bar.axhline(noise_floor, ls='-.', color='black', lw=0.9, alpha=0.7,
                                        label=fr'noise floor $2\sigma_{{loc}}^2$={noise_floor:.3g}')
                    if any_curve:
                        ax_bar.set_xscale('log'); ax_bar.set_yscale('log')
                    ax_bar.set_xlabel('τ (frames)')
                    ax_bar.set_ylabel('TA-EA MSD (px²)')
                    ax_bar.set_title('MSD(τ) vs noise floor')
                    ax_bar.grid(True, which='both', alpha=0.3)
                    ax_bar.legend(fontsize=7, loc='best')

                    _fill_stats_panel(ax_nf_stats, nf_lines)
                    section_diag.add_widget(_make_canvas(fig_nf, save_name=f'noise_floor_{safe_name}'))
                # Section_diag addWidget moved to the end. // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29

            has_any_md = any(r.get('multi_delta_per_state') for r in results_list)
            if has_any_md:
                section3 = CollapsibleSection("Corrected Cauchy multi-Δ scan — Ĥ(Δ)")
                for r in results_list:
                    md_per_st = r.get('multi_delta_per_state') or {}
                    if not md_per_st:
                        continue
                    fig3, ax3, ax3_stats = _make_fig_with_stats()
                    name = r['name']
                    states_in_md = sorted(md_per_st.keys())
                    state_pal = sns.color_palette('tab10', n_colors=max(len(r['total_states']), 1))
                    stat_lines = []
                    for st in states_in_md:
                        scan = md_per_st[st]
                        deltas = np.asarray(scan['deltas'], dtype=float)
                        H_est = np.asarray(scan['H_est'], dtype=float)
                        n_ratios = np.asarray(scan['n_ratios'], dtype=int)
                        converged = np.asarray(scan.get('converged', np.zeros_like(deltas, dtype=bool)), dtype=bool)
                        sigma_H = np.asarray(scan.get('sigma_H', np.full_like(deltas, np.nan)), dtype=float)  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                        try:
                            color = state_pal[r['total_states'].index(st) % len(state_pal)]
                        except Exception:
                            color = state_pal[0]
                        finite = np.isfinite(H_est)
                        if not np.any(finite):
                            continue
                        # 95% CRLB band (1.96σ) — drawn first so points/line render on top.
                        band_mask = finite & np.isfinite(sigma_H)
                        if np.any(band_mask):
                            lo = np.clip(H_est[band_mask] - 1.96 * sigma_H[band_mask], 0.0, 1.0)
                            hi = np.clip(H_est[band_mask] + 1.96 * sigma_H[band_mask], 0.0, 1.0)
                            ax3.fill_between(deltas[band_mask], lo, hi, color=color,
                                             alpha=0.15, linewidth=0)
                        # Solid points for converged fits, hollow for non-converged.
                        conv_mask = finite & converged
                        nonconv_mask = finite & (~converged)
                        ax3.plot(deltas[finite], H_est[finite], '-', color=color, alpha=0.6,
                                 label=f'State {st}' if len(states_in_md) > 1 else name)
                        if np.any(conv_mask):
                            ax3.scatter(deltas[conv_mask], H_est[conv_mask], color=color,
                                        s=30, edgecolors='none')
                        if np.any(nonconv_mask):
                            ax3.scatter(deltas[nonconv_mask], H_est[nonconv_mask],
                                        facecolors='none', edgecolors=color, s=30, linewidth=1.0)
                        K_val = (r.get('K_est_per_state') or {}).get(st)
                        if K_val is not None and np.isfinite(K_val):
                            stat_lines.append((f'State {st}: K={K_val:.4g} px²/fr^(2H)',
                                               color, None, None))
                        stat_lines.append((f'  Δ_max={int(scan["delta_max"])}, n_ratios(Δ=1)={int(n_ratios[0])}',
                                           color, None, None))
                        if np.any(band_mask):
                            stat_lines.append((f'  CRLB σ_H(Δ=1)={float(sigma_H[band_mask][0]):.3f}',
                                               color, None, None))
                    sigma_loc_rms_px = r.get('sigma_loc_rms_px')
                    R_used = r.get('R_used')
                    if sigma_loc_rms_px is not None:
                        stat_lines.insert(0, (f'σ_loc rms = {sigma_loc_rms_px:.4g} px',
                                              '#aaaaaa', None, None))
                    if R_used is not None:
                        stat_lines.insert(1 if sigma_loc_rms_px is not None else 0,
                                          (f'R = {R_used:.3g}', '#aaaaaa', None, None))
                    # Trust bands: σ_Ĥ_CRLB ≤ τ AND |Ĥ-spread| ≤ τ under σ_loc·{√0.5, √2}.  // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                    # Two thresholds (τ ∈ {0.05, 0.01}); per-state masks intersected.
                    sigma_for_band = (float(sigma_loc_rms_px)
                                      if (sigma_loc_rms_px is not None
                                          and np.isfinite(sigma_loc_rms_px)) else 0.0)
                    R_for_band = float(R_used or 0.0)
                    TAU_LEVELS = [(0.05, 0.10), (0.01, 0.18)]  # (τ, axvspan alpha)
                    trust_masks = {tau: None for tau, _ in TAU_LEVELS}
                    deltas_band = None
                    for st in states_in_md:
                        K_st = (r.get('K_est_per_state') or {}).get(st)
                        if K_st is None or not np.isfinite(K_st) or K_st <= 0:
                            continue
                        scan_st = md_per_st[st]
                        deltas_st = np.asarray(scan_st['deltas'], dtype=float)
                        sigma_H_st = np.asarray(scan_st.get('sigma_H', np.full_like(deltas_st, np.nan)), dtype=float)
                        rho_arr_st = np.asarray(scan_st.get('rho_arr', np.full_like(deltas_st, np.nan)), dtype=float)
                        H_est_st = np.asarray(scan_st['H_est'], dtype=float)
                        spread = np.full(len(deltas_st), np.nan)
                        # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                        # Bias proxy via first-order error propagation (chapter01.tex
                        # eq:ch1_Hhat_sensitivity, generalised to R>0 via J_var/J_cov):
                        #   ∂ρ/∂σ²  = -K·(J_var(H,R,Δ) + J_cov(H,R,Δ)) / [2·(K·J_var + σ²)²]
                        #   ∂Ĥ/∂σ² = -∂ρ/∂σ² / ∂ρ/∂H        (∂ρ/∂H from cauchy_fit numerics)
                        # spread(Δ) = |∂Ĥ/∂σ²| · σ²   ← convention δσ² = σ² (100% relative).
                        try:
                            from cauchy_fit import (J_var as _Jv, J_cov as _Jc,
                                                     drho_dH_numerical as _drho_dH)
                        except Exception:
                            _Jv = _Jc = _drho_dH = None
                        sigma2 = float(sigma_for_band) ** 2
                        for i, dval in enumerate(deltas_st):
                            if not (np.isfinite(rho_arr_st[i]) and np.isfinite(H_est_st[i])):
                                continue
                            if _Jv is None or _Jc is None or _drho_dH is None:
                                spread[i] = float('inf')
                                continue
                            H_hat = float(H_est_st[i])
                            Jv = _Jv(H_hat, R_for_band, float(dval))
                            Jc = _Jc(H_hat, R_for_band, float(dval))
                            M = float(K_st) * Jv
                            denom = 2.0 * (M + sigma2) ** 2
                            if denom <= 0:
                                spread[i] = float('inf')
                                continue
                            drho_dsig2 = -float(K_st) * (Jv + Jc) / denom
                            d_rho_dH = _drho_dH(H_hat, float(dval), R_for_band,
                                                 float(K_st), sigma_for_band)
                            if not np.isfinite(d_rho_dH) or abs(d_rho_dH) < 1e-12:
                                spread[i] = float('inf')
                                continue
                            spread[i] = abs(drho_dsig2 / d_rho_dH) * sigma2
                        if deltas_band is None:
                            deltas_band = deltas_st
                        n = int(min(len(deltas_band), len(deltas_st)))
                        deltas_band = deltas_band[:n]
                        for tau, _ in TAU_LEVELS:
                            prec_ok = np.isfinite(sigma_H_st[:n]) & (sigma_H_st[:n] <= tau)
                            bias_ok = np.isfinite(spread[:n]) & (spread[:n] <= tau)
                            pass_st = prec_ok & bias_ok
                            if trust_masks[tau] is None:
                                trust_masks[tau] = pass_st.copy()
                            else:
                                trust_masks[tau] = trust_masks[tau][:n] & pass_st
                    if deltas_band is not None and len(deltas_band) > 0:
                        # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                        # Edge runs extend to the axis xlim margins so the colour reaches the figure edge.
                        def _runs_indices(mask):
                            in_run = False; lo_idx = None
                            for i, ok in enumerate(mask):
                                if ok and not in_run:
                                    lo_idx = i; in_run = True
                                elif (not ok) and in_run:
                                    yield (lo_idx, i - 1); in_run = False
                            if in_run:
                                yield (lo_idx, len(mask) - 1)

                        m05 = trust_masks.get(0.05)
                        m01 = trust_masks.get(0.01)
                        if m05 is not None and m01 is not None:
                            xlim_lo, xlim_hi = ax3.get_xlim()
                            n_db = int(len(deltas_band))
                            fail_05 = ~m05
                            borderline = m05 & (~m01)

                            def _to_xrange(lo_idx, hi_idx):
                                lx = xlim_lo if lo_idx == 0 else float(deltas_band[lo_idx]) - 0.5
                                hx = xlim_hi if hi_idx == n_db - 1 else float(deltas_band[hi_idx]) + 0.5
                                return lx, hx

                            for (li, hi_) in _runs_indices(fail_05):
                                lx, hx = _to_xrange(li, hi_)
                                ax3.axvspan(lx, hx, alpha=0.7, color='#5d4037',  # dark brown = fail-loose
                                            linewidth=0, zorder=0)
                            for (li, hi_) in _runs_indices(borderline):
                                lx, hx = _to_xrange(li, hi_)
                                ax3.axvspan(lx, hx, alpha=0.4, color='#d2b48c',  # tan = borderline
                                            linewidth=0, zorder=0)
                            ax3.set_xlim(xlim_lo, xlim_hi)
                        # Trust-band legend + per-τ Δ ranges. // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                        stat_lines.append(('Trust-band overlay:', '#aaaaaa', None, None))
                        stat_lines.append(('  dark brown = fail-loose (fails ε=0.05)',
                                            '#5d4037', None, None))
                        stat_lines.append(('  tan = borderline (passes ε=0.05, fails ε=0.01)',
                                            '#a87c4f', None, None))
                        stat_lines.append(('  no overlay = strict trust (passes ε=0.01)',
                                            '#aaaaaa', None, None))
                        # Trust-band Δ-range lines removed (per user request) — the visual band on // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                        # the plot already conveys this; the text duplicate was confusing.
                        # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                    # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                    ax3.axhline(0.5, linestyle=':', color='#888888', linewidth=0.8)
                    ax3.axhline(0.25, linestyle=':', color='#888888', linewidth=0.8)  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                    ax3.axhline(0.75, linestyle=':', color='#888888', linewidth=0.8)  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                    ax3.set_xlabel('Δ (frames)')
                    ax3.set_ylabel('Ĥ(Δ)')
                    ax3.set_ylim(-0.05, 1.05)
                    ax3.set_title(f'Corrected Cauchy Ĥ(Δ) — {name}')
                    ax3.grid(True, alpha=0.3)
                    ax3.legend(fontsize=7, loc='best')
                    _fill_stats_panel(ax3_stats, stat_lines)
                    safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
                    section3.add_widget(_make_canvas(fig3, save_name=f'multi_delta_H_{safe_name}'))
                # Section3 addWidget moved to the end. // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29

                # ---- Drift / sensitivity / reliability — 4 supplementary panels ----  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                # Drift Ĥ(Δ+1)−Ĥ(Δ) with ±1/2/3σ floor (built from sigma_H).
                # K-sensitivity: re-invert H at K·{0.8..1.2} via _invert_H_from_rho.
                # σ_loc-sensitivity: same with σ_loc·{0.8..1.2}.
                # Reliability map: pcolormesh of (Δ, n_eff_grid) → CRLB σ_H/√n_eff at Ĥ(Δ),
                # with empirical n_eff(Δ) overlaid as a cyan curve.
                from matplotlib.colors import LogNorm as _LogNorm
                try:
                    from cauchy_fit import sigma_H_crlb as _sigma_H_crlb
                except Exception:
                    _sigma_H_crlb = None
                section_zmat = CollapsibleSection("Pairwise Ĥ(Δ) significance matrix — Z = ΔH / √(σ²+σ²)")  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                section_drift = CollapsibleSection("Adjacent-Δ drift Ĥ(Δ+1)−Ĥ(Δ)")
                # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                # Replace separate K-sensitivity and σ_loc-sensitivity panels with a single
                # λ_noise = σ_loc² / K sweep (thesis Appendix A convention).
                section_lambda = CollapsibleSection("Ĥ(Δ) sensitivity to λ_noise = σ²_loc / K")
                section_REL = CollapsibleSection("Precision map — CRLB σ_H over (Δ, n_eff)")  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                added_zmat = added_drift = added_lambda = added_REL = False
                # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                for r in results_list:
                    md_per_st = r.get('multi_delta_per_state') or {}
                    if not md_per_st:
                        continue
                    name = r['name']
                    safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
                    states_in_md = sorted(md_per_st.keys())
                    state_pal = sns.color_palette('tab10', n_colors=max(len(r['total_states']), 1))
                    R_used = float(r.get('R_used') or 0.0)
                    K_per_state = r.get('K_est_per_state') or {}
                    sigma_loc_rms_px = r.get('sigma_loc_rms_px')
                    s_loc = float(sigma_loc_rms_px) if (sigma_loc_rms_px is not None and np.isfinite(sigma_loc_rms_px)) else 0.0

                    # ---- Pairwise Ĥ(Δ) significance Z-matrix ---- // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                    # Z_{ij} = (Ĥ(Δ_i) − Ĥ(Δ_j)) / √(σ_H(Δ_i)² + σ_H(Δ_j)²)
                    # Two-sided p_{ij} = 2·(1 − Φ(|Z|)) per pair. With ~75 Δ values we get
                    # ~2775 pairs — under H₀ (Ĥ truly constant) ~138 fall below p<0.05 by
                    # chance. We apply Benjamini–Hochberg FDR control (q≤0.05) which is
                    # valid under positive regression dependence — the Ĥ(Δ) chain satisfies
                    # this since adjacent Δ share most of the same ratio data.
                    from scipy.stats import norm as _norm  # erfc-backed CDF tail
                    for st in states_in_md:
                        scan = md_per_st[st]
                        deltas = np.asarray(scan['deltas'], dtype=float)
                        H_est = np.asarray(scan['H_est'], dtype=float)
                        sigma_H = np.asarray(scan.get('sigma_H', np.full_like(deltas, np.nan)), dtype=float)
                        finite = np.isfinite(H_est) & np.isfinite(sigma_H) & (sigma_H > 0)
                        if finite.sum() < 3:
                            continue
                        d_arr = deltas[finite]
                        H_arr = H_est[finite]
                        s_arr = sigma_H[finite]
                        n = len(d_arr)
                        Hi = H_arr[:, None]; Hj = H_arr[None, :]
                        si = s_arr[:, None]; sj = s_arr[None, :]
                        Z = (Hi - Hj) / np.sqrt(si ** 2 + sj ** 2)
                        absZ = np.abs(Z)
                        upper = np.triu_indices(n, k=1)
                        n_pairs = upper[0].size
                        absZ_upper = absZ[upper]
                        n_sig2 = int((absZ_upper > 2).sum())
                        n_sig3 = int((absZ_upper > 3).sum())
                        max_idx = np.unravel_index(np.argmax(absZ * np.triu(np.ones_like(absZ), k=1)), absZ.shape)
                        max_z = float(Z[max_idx])
                        max_pair = (int(d_arr[max_idx[0]]), int(d_arr[max_idx[1]]))

                        # ---- Benjamini–Hochberg FDR control at q≤0.05 ---- // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                        FDR_Q = 0.05
                        # Two-sided p per pair via norm survival (numerically stable for large |Z|).
                        p_upper = 2.0 * _norm.sf(absZ_upper)
                        order = np.argsort(p_upper)
                        p_sorted = p_upper[order]
                        N_tests = n_pairs
                        # Largest k such that p₍ₖ₎ ≤ q·k/N (1-indexed in formula → 0-indexed below).
                        thresholds = FDR_Q * (np.arange(1, N_tests + 1)) / N_tests
                        passing = p_sorted <= thresholds
                        if np.any(passing):
                            k_star = int(np.max(np.flatnonzero(passing)))
                            p_cut = float(p_sorted[k_star])
                            n_fdr = k_star + 1
                        else:
                            p_cut = 0.0
                            n_fdr = 0
                        # Build a boolean significance mask in the upper-triangle layout, then
                        # symmetrise for the 2D matrix.
                        sig_mask_upper = (p_upper <= p_cut) if n_fdr > 0 else np.zeros_like(p_upper, dtype=bool)
                        sig_mat = np.zeros((n, n), dtype=bool)
                        sig_mat[upper] = sig_mask_upper
                        sig_mat = sig_mat | sig_mat.T  # symmetric display

                        # Custom larger figure for the heatmap (default 10x4 is too short  // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                        # for a square Δ×Δ matrix). 13x8 with 4:1 width split → ~10x8 plot.
                        fig_z = Figure(figsize=(13, 8), dpi=100, constrained_layout=True)
                        gs_z = fig_z.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.02)
                        ax_z = fig_z.add_subplot(gs_z[0])
                        ax_z_stats = fig_z.add_subplot(gs_z[1])
                        ax_z_stats.axis('off')
                        # |Z| matrix with fixed scale [0, 5]: blue → white → red.
                        # |Z| is symmetric in (i,j), so the lower and upper triangles are mirror.
                        # |Z|=2 → p≈0.05, |Z|=3 → p≈0.003, |Z|=5 → p≈6e-7. Saturating at 5
                        # keeps the colormap interpretable rather than swamped by extreme cells.
                        edges = np.concatenate([d_arr - 0.5, [d_arr[-1] + 0.5]])
                        im = ax_z.pcolormesh(edges, edges, absZ, cmap='coolwarm',
                                              vmin=0.0, vmax=5.0, shading='auto')
                        # FDR-significant cells: outline with a black contour around the // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                        # boolean mask. Cells outside this contour are NOT significant after
                        # multiple-comparison correction (BH at q≤0.05) — even if their |Z|>2.
                        try:
                            if sig_mat.any() and (~sig_mat).any():
                                ax_z.contour(d_arr, d_arr, sig_mat.astype(float),
                                              levels=[0.5], colors=['black'], linewidths=1.4)
                        except Exception:
                            pass
                        # Light reference contours at |z|=2, |z|=3 (uncorrected thresholds).
                        try:
                            ax_z.contour(d_arr, d_arr, absZ,
                                          levels=[2.0, 3.0], colors=['#888888', '#444444'],
                                          linewidths=[0.5, 0.8], linestyles=['--', '--'])
                        except Exception:
                            pass
                        # Trust-band shading on the matrix: dim Δs outside the trust band. // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                        # Computed per-state from σ_Ĥ_CRLB (precision) AND |Ĥ|-spread under
                        # σ_loc·{√0.5, √2} (bias). Two thresholds (0.05, 0.01).
                        try:
                            sigma_for_band_z = (float(sigma_loc_rms_px)
                                                if (sigma_loc_rms_px is not None
                                                    and np.isfinite(sigma_loc_rms_px)) else 0.0)
                            R_for_band_z = float(R_used or 0.0)
                            K_st_z = (r.get('K_est_per_state') or {}).get(st)
                            sigma_H_arr_z = np.asarray(scan.get('sigma_H', np.full_like(deltas, np.nan)),
                                                       dtype=float)
                            rho_arr_z = np.asarray(scan.get('rho_arr', np.full_like(deltas, np.nan)),
                                                   dtype=float)
                            H_est_arr_z = np.asarray(scan['H_est'], dtype=float)
                            spread_z = np.full(len(deltas), np.nan)
                            # Analytic bias proxy = |∂Ĥ/∂σ²|·σ² (thesis eq:ch1_Hhat_sensitivity). // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                            try:
                                from cauchy_fit import (J_var as _Jv2, J_cov as _Jc2,
                                                         drho_dH_numerical as _drho_dH2)
                            except Exception:
                                _Jv2 = _Jc2 = _drho_dH2 = None
                            sigma2_z = float(sigma_for_band_z) ** 2
                            if (K_st_z is not None and np.isfinite(K_st_z) and K_st_z > 0
                                    and _Jv2 is not None):
                                for i, dval in enumerate(deltas):
                                    if not (np.isfinite(rho_arr_z[i]) and np.isfinite(H_est_arr_z[i])):
                                        continue
                                    H_hat = float(H_est_arr_z[i])
                                    Jv = _Jv2(H_hat, R_for_band_z, float(dval))
                                    Jc = _Jc2(H_hat, R_for_band_z, float(dval))
                                    M = float(K_st_z) * Jv
                                    denom = 2.0 * (M + sigma2_z) ** 2
                                    if denom <= 0:
                                        spread_z[i] = float('inf')
                                        continue
                                    drho_dsig2 = -float(K_st_z) * (Jv + Jc) / denom
                                    d_rho_dH = _drho_dH2(H_hat, float(dval), R_for_band_z,
                                                          float(K_st_z), sigma_for_band_z)
                                    if not np.isfinite(d_rho_dH) or abs(d_rho_dH) < 1e-12:
                                        spread_z[i] = float('inf')
                                        continue
                                    spread_z[i] = abs(drho_dsig2 / d_rho_dH) * sigma2_z
                            n_z = len(d_arr)
                            prec_05 = np.isfinite(sigma_H_arr_z[:n_z]) & (sigma_H_arr_z[:n_z] <= 0.05)
                            bias_05 = np.isfinite(spread_z[:n_z]) & (spread_z[:n_z] <= 0.05)
                            trust_05 = prec_05 & bias_05
                            prec_01 = np.isfinite(sigma_H_arr_z[:n_z]) & (sigma_H_arr_z[:n_z] <= 0.01)
                            bias_01 = np.isfinite(spread_z[:n_z]) & (spread_z[:n_z] <= 0.01)
                            trust_01 = prec_01 & bias_01

                            def _runs_where(mask, vals):
                                in_run = False; lo = None
                                for i, ok in enumerate(mask):
                                    if ok and not in_run:
                                        lo = float(vals[i]) - 0.5; in_run = True
                                    elif (not ok) and in_run:
                                        yield (lo, float(vals[i - 1]) + 0.5)
                                        in_run = False
                                if in_run:
                                    yield (lo, float(vals[-1]) + 0.5)

                            # Unified single-color overlay (no double-alpha at intersections). // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                            # 2D mask: dark gray if Δᵢ OR Δⱼ fails loose; light gray if either is
                            # borderline (and neither fails loose); no overlay if both pass strict.
                            n_z2 = int(len(d_arr))
                            fail_05_arr = (~trust_05).astype(bool)
                            border_arr = (trust_05 & (~trust_01)).astype(bool)
                            row_fail = fail_05_arr[:, None] | fail_05_arr[None, :]
                            row_border = border_arr[:, None] | border_arr[None, :]
                            overlay = np.zeros((n_z2, n_z2, 4), dtype=np.float32)
                            # dark brown #5d4037 (alpha 0.7) for FAIL-LOOSE
                            overlay[row_fail] = (0x5d / 255.0, 0x40 / 255.0, 0x37 / 255.0, 0.7)
                            # tan #d2b48c (alpha 0.4) for BORDERLINE only
                            border_only = row_border & (~row_fail)
                            overlay[border_only] = (0xd2 / 255.0, 0xb4 / 255.0, 0x8c / 255.0, 0.4)
                            ax_z.imshow(
                                overlay,
                                extent=[float(d_arr[0]) - 0.5, float(d_arr[-1]) + 0.5,
                                        float(d_arr[0]) - 0.5, float(d_arr[-1]) + 0.5],
                                origin='lower', aspect='auto',
                                interpolation='nearest', zorder=3.0,
                            )
                        except Exception:
                            pass
                        ax_z.set_xlabel('Δⱼ (frames)')
                        ax_z.set_ylabel('Δᵢ (frames)')
                        title_st = f' state {st}' if len(states_in_md) > 1 else ''
                        ax_z.set_title(f'Pairwise |Z|(Δᵢ,Δⱼ) — {name}{title_st}')
                        ax_z.set_aspect('equal')
                        cb = fig_z.colorbar(im, ax=ax_z, fraction=0.045, pad=0.02, extend='max')
                        cb.set_label('|Z| = |Ĥᵢ−Ĥⱼ|/√(σᵢ²+σⱼ²)   (saturated at 5)', fontsize=8)
                        cb.set_ticks([0, 2, 3, 5])
                        cb.set_ticklabels(['0  (no diff)', '2  (p≈0.05)', '3  (p≈3e-3)', '5  (p≈6e-7)'])
                        try:
                            color_st = state_pal[r['total_states'].index(st) % len(state_pal)]
                        except Exception:
                            color_st = state_pal[0]
                        # Expected counts under H₀ (true uniformity), for context. // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                        exp_p05 = 0.05 * n_pairs
                        exp_p003 = 0.0027 * n_pairs
                        z_lines = [
                            (f'{n} Δ values, {n_pairs} unique pairs', '#aaaaaa', None, None),
                            (f'|z|>2 (p<0.05): {n_sig2}/{n_pairs} = {100*n_sig2/max(n_pairs,1):.0f}%   (expected ≈{exp_p05:.0f})',
                             '#dd8866' if n_sig2 > 0 else '#aaaaaa', None, None),
                            (f'|z|>3 (p<0.003): {n_sig3}/{n_pairs} = {100*n_sig3/max(n_pairs,1):.0f}%   (expected ≈{exp_p003:.1f})',
                             '#cc4444' if n_sig3 > 0 else '#aaaaaa', None, None),
                            (f'BH-FDR significant (q≤{FDR_Q}): {n_fdr}/{n_pairs} = {100*n_fdr/max(n_pairs,1):.0f}%   (p_cut={p_cut:.2g})',
                             '#88cc44' if n_fdr > 0 else '#aaaaaa', None, None),
                            (f'max |z| = {abs(max_z):.2f} at (Δ={max_pair[0]}, Δ={max_pair[1]})',
                             color_st, None, None),
                            (f'Ĥ(Δ={max_pair[0]})={H_arr[max_idx[0]]:.3f}, Ĥ(Δ={max_pair[1]})={H_arr[max_idx[1]]:.3f}',
                             color_st, None, None),
                            ('Solid black contour: BH-FDR-significant region', '#aaaaaa', None, None),
                            ('Dashed grey: |z|=2, |z|=3 (uncorrected)', '#888888', None, None),
                            # Trust-band legend // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                            ('Trust-band overlay (rows + cols):', '#aaaaaa', None, None),
                            ('  dark brown = fail-loose Δ', '#5d4037', None, None),
                            ('  tan = borderline Δ', '#a87c4f', None, None),
                            ('  no overlay = strict trust', '#aaaaaa', None, None),
                        ]
                        _fill_stats_panel(ax_z_stats, z_lines)
                        canvas_z = _make_canvas(fig_z, save_name=f'zmat_{safe_name}_S{st}')
                        canvas_z.setMinimumHeight(700)  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
                        section_zmat.add_widget(canvas_z)
                        added_zmat = True

                    # ---- Drift panel ----
                    fig_d, ax_d, ax_d_stats = _make_fig_with_stats()
                    drift_lines = []
                    has_drift = False
                    for st in states_in_md:
                        scan = md_per_st[st]
                        deltas = np.asarray(scan['deltas'], dtype=float)
                        H_est = np.asarray(scan['H_est'], dtype=float)
                        sigma_H = np.asarray(scan.get('sigma_H', np.full_like(deltas, np.nan)), dtype=float)
                        finite = np.isfinite(H_est) & np.isfinite(sigma_H)
                        if finite.sum() < 2:
                            continue
                        try:
                            color = state_pal[r['total_states'].index(st) % len(state_pal)]
                        except Exception:
                            color = state_pal[0]
                        d_arr = deltas[finite]
                        H_arr = H_est[finite]
                        s_arr = sigma_H[finite]
                        d_adj = d_arr[1:]
                        drift = np.diff(H_arr)
                        floor = np.sqrt(s_arr[:-1] ** 2 + s_arr[1:] ** 2)
                        ax_d.fill_between(d_adj, -3 * floor, 3 * floor, color=color, alpha=0.08)
                        ax_d.fill_between(d_adj, -2 * floor, 2 * floor, color=color, alpha=0.14)
                        ax_d.fill_between(d_adj, -floor, floor, color=color, alpha=0.22)
                        ax_d.plot(d_adj, drift, 'o-', color=color, lw=1.2, markersize=3,
                                  label=f'State {st}' if len(states_in_md) > 1 else name)
                        has_drift = True
                    if has_drift:
                        ax_d.axhline(0, color='#888888', lw=0.8)
                        ax_d.set_xlabel('Δ (frames)')
                        ax_d.set_ylabel('Ĥ(Δ+1) − Ĥ(Δ)')
                        ax_d.set_title(f'Adjacent-Δ drift — {name}')
                        ax_d.grid(True, alpha=0.3)
                        ax_d.legend(fontsize=7, loc='best')
                        section_drift.add_widget(_make_canvas(fig_d, save_name=f'drift_{safe_name}'))
                        added_drift = True
                    else:
                        plt.close(fig_d)

                    # ---- λ_noise = σ²_loc / K sensitivity ---- // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                    # References are multipliers of the data-derived λ_baseline.
                    # m·λ_baseline = (σ_target² / K_fit), so σ_target = √m · σ_loc (K_fit fixed).
                    fig_L, ax_L, ax_L_stats = _make_fig_with_stats()
                    has_L = False
                    LAMBDA_MULTS = [0.1, 0.5, 1.0, 2.0, 10.0]
                    sens_pal = sns.color_palette('tab10', n_colors=len(LAMBDA_MULTS))
                    for st in states_in_md:
                        K_st = K_per_state.get(st)
                        if K_st is None or not np.isfinite(K_st) or K_st <= 0:
                            continue
                        scan = md_per_st[st]
                        deltas = np.asarray(scan['deltas'], dtype=float)
                        rho_arr = np.asarray(scan.get('rho_arr', np.full_like(deltas, np.nan)), dtype=float)
                        finite = np.isfinite(rho_arr)
                        if not np.any(finite):
                            continue
                        lambda_base = (s_loc * s_loc) / float(K_st)
                        for ci, mult in enumerate(LAMBDA_MULTS):
                            sigma_target = float(np.sqrt(mult)) * s_loc  # √m · σ_loc
                            lam_target = mult * lambda_base
                            H_l = np.full(len(deltas), np.nan)
                            for i, d in enumerate(deltas):
                                if not finite[i]:
                                    continue
                                H_l[i] = _invert_H_from_rho(float(rho_arr[i]), float(d),
                                                             R_used, float(K_st), sigma_target)
                            mk = np.isfinite(H_l)
                            is_baseline = (mult == 1.0)
                            ls = '-' if is_baseline else '--'
                            lw = 2.5 if is_baseline else 1.0
                            tag = ' (baseline)' if is_baseline else ''
                            label = (f'S{st} λ={lam_target:.3g} ({mult:g}×λ₀){tag}'
                                     if len(states_in_md) > 1
                                     else f'λ={lam_target:.3g} ({mult:g}×λ₀){tag}')
                            ax_L.plot(deltas[mk], H_l[mk], ls, color=sens_pal[ci],
                                      lw=lw, label=label)
                        has_L = True
                    if has_L:
                        ax_L.axhline(0.5, linestyle=':', color='#888888', linewidth=0.8)
                        ax_L.set_xlabel('Δ (frames)')
                        ax_L.set_ylabel('Ĥ (corrected)')
                        ax_L.set_ylim(-0.05, 1.05)
                        ax_L.set_title(f'Ĥ(Δ) sensitivity to λ_noise = σ²_loc / K — {name}')
                        ax_L.grid(True, alpha=0.3)
                        ax_L.legend(fontsize=7, loc='best', ncol=1)
                        section_lambda.add_widget(_make_canvas(fig_L, save_name=f'sens_lambda_{safe_name}'))
                        added_lambda = True
                    else:
                        plt.close(fig_L)
                    # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29

                    # ---- Reliability map ----
                    if _sigma_H_crlb is None:
                        continue
                    for st in states_in_md:
                        K_st = K_per_state.get(st)
                        if K_st is None or not np.isfinite(K_st) or K_st <= 0:
                            continue
                        scan = md_per_st[st]
                        deltas = np.asarray(scan['deltas'], dtype=float)
                        H_est = np.asarray(scan['H_est'], dtype=float)
                        n_eff = np.asarray(scan.get('n_eff', np.full_like(deltas, np.nan)), dtype=float)
                        finite_H = np.isfinite(H_est)
                        finite_n = np.isfinite(n_eff)
                        if finite_H.sum() < 2:
                            continue
                        try:
                            color = state_pal[r['total_states'].index(st) % len(state_pal)]
                        except Exception:
                            color = state_pal[0]
                        # σ_H(Δ; n_eff=1) per Δ at fitted Ĥ(Δ); Z = factor / √n_eff.
                        factors = np.full(len(deltas), np.nan)
                        for i, d in enumerate(deltas):
                            if finite_H[i]:
                                factors[i] = _sigma_H_crlb(float(H_est[i]), float(d),
                                                            R_used, float(K_st), s_loc, 1.0)
                        m_factors = np.isfinite(factors)
                        if m_factors.sum() < 2:
                            continue
                        d_grid = deltas[m_factors]
                        f_grid = factors[m_factors]
                        n_eff_grid = np.logspace(1, 5, 120)
                        Z_rel = f_grid[None, :] / np.sqrt(n_eff_grid[:, None])
                        Z_rel = np.clip(Z_rel, 1e-4, 1.0)

                        fig_R, ax_R, ax_R_stats = _make_fig_with_stats()
                        im = ax_R.pcolormesh(d_grid, n_eff_grid, Z_rel,
                                              norm=_LogNorm(vmin=0.005, vmax=0.5),
                                              cmap='RdYlGn_r', shading='auto')
                        cs = ax_R.contour(d_grid, n_eff_grid, Z_rel,
                                           levels=[0.01, 0.02, 0.05, 0.1, 0.2],
                                           colors='black', linewidths=0.6)
                        ax_R.clabel(cs, fmt='%.2f', fontsize=6)
                        if np.any(finite_n):
                            ax_R.plot(deltas[finite_n], n_eff[finite_n], '-', color='cyan', lw=1.6)
                            ax_R.scatter(deltas[finite_n], n_eff[finite_n], s=18, color='cyan',
                                          edgecolor='black', linewidth=0.4, zorder=5,
                                          label=f'empirical n_eff(Δ)' if len(states_in_md) == 1 else f'S{st} n_eff(Δ)')
                        ax_R.set_yscale('log')
                        ax_R.set_xlim(d_grid[0], d_grid[-1])
                        ax_R.set_ylim(n_eff_grid[0], n_eff_grid[-1])
                        ax_R.set_xlabel('Δ (frames)')
                        ax_R.set_ylabel('n_eff')
                        title_st = f' state {st}' if len(states_in_md) > 1 else ''
                        ax_R.set_title(f'Precision map — {name}{title_st}')  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                        cb = fig_R.colorbar(im, ax=ax_R, fraction=0.04, pad=0.02)
                        cb.set_label('CRLB σ_H', fontsize=8)
                        ax_R.legend(fontsize=7, loc='upper right')
                        ax_R_stats.axis('off')
                        rel_lines = [
                            (f'σ_loc rms = {s_loc:.4g} px', '#aaaaaa', None, None),
                            (f'R = {R_used:.3g}', '#aaaaaa', None, None),
                            (f'K (state {st}) = {K_st:.4g}', color, None, None),
                            ('Contours: σ_H levels', '#aaaaaa', None, None),
                        ]
                        _fill_stats_panel(ax_R_stats, rel_lines)
                        section_REL.add_widget(_make_canvas(fig_R, save_name=f'reliability_{safe_name}_S{st}'))
                        added_REL = True

            # Section ordering: Ĥ(Δ) first, then |Z| matrix, then 1D Disp / noise floor / drift / λ_noise / Reliability. // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
            ordered_sections = []
            if has_any_md:
                ordered_sections.append(section3)
                if added_zmat:
                    ordered_sections.append(section_zmat)
            ordered_sections.append(section2)
            if has_any_diag:
                ordered_sections.append(section_diag)
            if has_any_md:
                if added_drift:
                    ordered_sections.append(section_drift)
                if added_lambda:
                    ordered_sections.append(section_lambda)
                if added_REL:
                    ordered_sections.append(section_REL)
            for _sec in ordered_sections:
                self._adv_stats_plot_layout.addWidget(_sec)

        # Apply legend visibility from checkbox  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        if not show_legend:
            for canvas in self._adv_stats_canvases:
                for ax in canvas.figure.get_axes():
                    leg = ax.get_legend()
                    if leg:
                        leg.set_visible(False)
                canvas.draw_idle()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

    def _on_save_adv_stats_plots(self):
        """Save all Advanced Stats plot canvases as PNG files."""  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
        if not self._adv_stats_canvases:
            QMessageBox.warning(self, "No plots", "Run Advanced Stats first.")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Select directory to save plots")
        if not save_dir:
            return
        # Use names tagged on each canvas at render time when available; otherwise fall
        # back to a sequential index. Naming scheme:
        #   1D Displacement panels are tagged 'displacement_1d_<n>'; multi-Δ panels are
        #   tagged 'multi_delta_H_<dataset>'.
        saved = 0
        for i, canvas in enumerate(self._adv_stats_canvases):
            tag = getattr(canvas, '_save_name', None) or f'adv_plot_{i}'
            path = os.path.join(save_dir, f'{tag}.png')
            canvas.figure.savefig(path, dpi=150, bbox_inches='tight', transparent=True)
            saved += 1
        self._adv_stats_status_label.setText(f"Saved {saved} plots to {save_dir}")

    def _on_save_adv_stats_data(self):  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
        """Export multi-Δ scan results as one CSV per dataset×state.

        Columns: delta, H_estim, H_std, rho, n_eff, n_ratios, converged.
        Filename: <dataset>_S<state>_multi_delta.csv
        """
        results = self._adv_stats_results or []
        if not any(r.get('multi_delta_per_state') for r in results):
            QMessageBox.warning(self, "No scan data", "Run Advanced Stats first.")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Select directory to save scan data")
        if not save_dir:
            return
        written = 0
        for r in results:
            md_per_st = r.get('multi_delta_per_state') or {}
            if not md_per_st:
                continue
            name = r['name']
            safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
            K_per_state = r.get('K_est_per_state') or {}
            sigma_loc_rms_px = r.get('sigma_loc_rms_px')
            R_used = r.get('R_used')
            for st, scan in md_per_st.items():
                deltas = np.asarray(scan['deltas'], dtype=int)
                # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                df = pd.DataFrame({
                    'delta':     deltas,
                    'H_estim':   np.asarray(scan['H_est']),
                    'H_std':     np.asarray(scan.get('sigma_H', np.full(len(deltas), np.nan))),
                    'rho':       np.asarray(scan.get('rho_arr', np.full(len(deltas), np.nan))),
                    'n_eff':     np.asarray(scan.get('n_eff', np.full(len(deltas), np.nan))),
                    'n_ratios':  np.asarray(scan['n_ratios']),
                    'converged': np.asarray(scan.get('converged', np.zeros(len(deltas), dtype=bool))).astype(int),
                })
                # End modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
                # Header lines with the plug-in parameters used by the scan.
                header_lines = [
                    f'# dataset = {name}',
                    f'# state = {st}',
                    f'# K_est_px2_per_frame_2H = {K_per_state.get(st)}',
                    f'# sigma_loc_rms_px = {sigma_loc_rms_px}',
                    f'# R = {R_used}',
                    f'# delta_max = {int(scan["delta_max"])}',
                ]
                out_path = os.path.join(save_dir, f'{safe_name}_S{st}_multi_delta.csv')
                with open(out_path, 'w') as f:
                    f.write('\n'.join(header_lines) + '\n')
                    df.to_csv(f, index=False)
                written += 1
        self._adv_stats_status_label.setText(f"Saved {written} scan-data CSV(s) to {save_dir}")

    def _adv_stats_help_html(self):  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-29
        """One-paragraph plain-language description for each panel."""
        return """
<h2>Advanced Stats — panel guide</h2>
<p>Each Run produces a stack of collapsible panels in the order below. Sections that
require σ_loc (per-spot CRLB) are skipped when the loaded <code>_loc.csv</code> lacks
<code>bg_median</code> / <code>bg_var</code> / <code>integrated_flux</code> — re-run
localisation with the current FreeTrace to enable them.</p>

<h3>1. Corrected Cauchy Ĥ(Δ)  <small>(headline)</small></h3>
<p>Per-Δ Hurst estimate from the corrected Cauchy MLE on displacement-ratio chains,
with a 95% CRLB band shaded around it. Flat curve at H≈0.5 is Brownian; monotone
decrease with Δ suggests confinement or non-Markov memory; V-shaped curves are
characteristic of confined Rouse polymer dynamics. K_est (from short-lag MSD fit)
and σ_loc rms appear in the stats panel.</p>
<p><b>Trust-band overlay</b> (brown / tan): Δs are flagged untrustworthy when either
the statistical CRLB (σ_Ĥ) or the bias proxy (|Ĥ-spread| under σ_loc·{√0.5, √2})
exceeds a precision target ε. Two ε levels — ε=0.05 (loose) and ε=0.01 (strict).</p>
<ul>
  <li>No overlay → Δ inside the strict trust band (passes both ε); Ĥ(Δ) is robust.</li>
  <li><b>Tan</b> (light) → borderline (passes ε=0.05 but fails ε=0.01).</li>
  <li><b>Dark brown</b> → fail-loose (Ĥ(Δ) clearly untrustworthy at this Δ).</li>
</ul>
<p>The bias proxy uses a saturation guard: when both σ_loc perturbations clamp at the
same H bound (degenerate inversion), spread is forced to +∞.</p>

<h3>2. Pairwise |Z| significance matrix</h3>
<p>For each pair of Δ values, the matrix shows
|Ĥ(Δᵢ)−Ĥ(Δⱼ)| / √(σ_H(Δᵢ)² + σ_H(Δⱼ)²) — the number of CRLB standard deviations
separating two H estimates. Cells inside the solid black contour are significant
under Benjamini–Hochberg FDR control at q≤0.05 (≤5% expected false positives among
the flagged cells). Dashed grey lines are the uncorrected |z|=2 / |z|=3 thresholds
(reference only — under H₀ with ~2,400 pairs, ~120 will exceed |z|=2 by chance).
Stats panel reports observed vs expected counts and the BH cutoff p-value.</p>
<p><b>Trust-band overlay</b>: same brown/tan convention as panel 1, applied as
row+column overlays (single-color, no double-alpha at intersections). Cells where
either Δᵢ or Δⱼ falls in the fail-loose region are dark brown; cells where either is
borderline (and neither fails loose) are tan; cells with both Δs in the strict
trust band are unshaded.</p>

<h3>3. 1D Displacement (Δx, Δy)</h3>
<p>Histogram of consecutive-frame jumps in x and y, with a Gaussian fit overlaid. A
clean Gaussian indicates a single diffusive regime; visible heavy tails or a central
spike suggest mixed populations or motion blur.</p>

<h3>4. Noise-floor diagnostic</h3>
<p><b>Left:</b> per-spot σ_loc (CRLB-derived) plotted against per-spot photon count
<i>I</i>. The red Thompson line is the theoretical Cramér–Rao bound at the dataset's
median PSF size and background. Spots hugging the red line means localisation is
shot-noise-limited (good); systematic offset above means residual systematic error
(drift, calibration, etc.).</p>
<p><b>Right:</b> ensemble TA-MSD(τ) on log-log axes with the noise floor 2σ_loc²
drawn as a dashed black horizontal line. The MSD curve must sit clearly above the
floor and grow with τ for diffusion to be detectable. Curves that flatten near the
floor at small τ are dominated by localisation noise and the corrected Cauchy fit
will be skipped (status flagged in the stats panel).</p>

<h3>5. Adjacent-Δ drift Ĥ(Δ+1)−Ĥ(Δ)</h3>
<p>The first-difference of Ĥ(Δ), with ±1σ / ±2σ / ±3σ floors built from the
neighbouring sigma_H values. Excursions outside the ±2σ band signal "real" jumps in
Ĥ at that Δ. Useful to identify the Δ at which the curve transitions between
regimes (e.g., from caged to free).</p>

<h3>6. Ĥ(Δ) sensitivity to λ_noise = σ²_loc / K</h3>
<p>Single panel that subsumes the previous K and σ_loc sensitivity panels. Holds
K = K_fit fixed and varies σ_loc to multiply the data-derived λ_baseline by
{0.1×, 0.5×, 1×, 2×, 10×}. So σ_target = √m · σ_loc for each multiplier m. The 1×
curve (solid, bold) is the baseline at the data's own (K_fit, σ_loc); the 0.5×
curve halves λ (cleaner-than-data), 2× doubles λ (noisier-than-data), etc. When
the curves agree across multipliers at large Δ, diffusion dominates and Ĥ(Δ) is
robust to noise; where they fan out (typically small Δ) you're in the
noise-dominated regime and Ĥ(Δ) at that lag is bias-prone.</p>

<h3>7. Precision map — CRLB σ_H over (Δ, n_eff)</h3>
<p>A 2-D heatmap showing the theoretical CRLB σ_H you'd get at any combination of Δ
and effective sample size n_eff. Contours mark σ_H levels (0.01, 0.02, 0.05, 0.1,
0.2). The cyan curve overlays your data's empirical n_eff(Δ). Reading: where the cyan
curve sits relative to the contours tells you the <i>statistical</i> precision floor
on Ĥ for each Δ. Note: this map only reports precision (CRLB); the systematic bias
from σ_loc misspecification is shown by the trust-band overlay on panels 1 and 2,
not here.</p>

<h3>Toolbar fields</h3>
<ul>
  <li><b>Load Data</b>: pick a <code>_traces.csv</code>; the GUI auto-pairs with
      a sibling <code>_loc.csv</code> (or its parent for ROI / region exports) and a
      sibling TIFF (when present, used to recompute <code>bg_var</code> /
      <code>integrated_flux</code> with the thesis-style annulus convention).</li>
  <li><b>σ_loc</b>: localisation precision in pixels, used as the noise plug-in for
      both K_est and the multi-Δ Cauchy fit. Auto-filled on data load from the
      computed CRLB rms; can be edited to override (e.g., to plug in a fiducial-bead
      calibration or to match the thesis value).</li>
  <li><b>Pixel size (μm/px)</b>, <b>Frame rate (s)</b>: scaling for displacements
      and times in the 1D Displacement / TA-MSD plots only — the multi-Δ scan and
      K_est are computed in raw pixel/frame units.</li>
  <li><b>Min traj length</b>: trajectories shorter than this are dropped.</li>
  <li><b>R</b>: motion-blur fraction τ_exp/Δt ∈ [0,1]. 0 = instantaneous capture; 1 =
      full-frame integration.</li>
  <li><b>Read metadata from video</b>: pick a TIFF/ND2 to auto-fill pixel size,
      frame interval, and R from ImageJ / NIS-Elements metadata. All-or-nothing —
      missing fields trigger a warning and the toolbar is left untouched. On
      success, Adv Stats is automatically re-run.</li>
  <li><b>▶ Run Advanced Stats</b>: launches the background worker.</li>
  <li><b>Save Plots</b>: PNG of every rendered panel in the chosen folder.</li>
  <li><b>Save Data</b>: per-(dataset×state) CSV of the multi-Δ scan
      (<code>delta, H_estim, H_std, rho, n_eff, n_ratios, converged</code>) plus
      K, σ_loc, R, and Δ_max in the comment-line metadata header.</li>
</ul>
"""

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

        # Progress  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        prog_row = QHBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        prog_row.addWidget(self._progress_bar, stretch=1)

        self._elapsed_label = QLabel("")
        self._elapsed_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._elapsed_label.setMinimumWidth(120)
        prog_row.addWidget(self._elapsed_label)
        layout.addLayout(prog_row)

        self._stage_label = QLabel("")
        self._stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._stage_label)

        # Elapsed / ETA timer
        self._run_timer = QTimer(self)
        self._run_timer.setInterval(1000)
        self._run_timer.timeout.connect(self._update_elapsed)
        self._run_start_time = 0.0
        self._progress_history = []  # list of (time, pct) for recent-rate ETA  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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
        self._elapsed_label.setStyleSheet(f"color:#aaa; font-size:{f(12)}px;")  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
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
        self._progress_bar.setFormat("%p%")
        self._stage_label.setText("")
        self._elapsed_label.setText("")
        self._run_start_time = time.monotonic()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._progress_history = []
        self._run_timer.start()
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

    def _update_progress(self, value: int, label: str):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._progress_bar.setValue(value)
        # Show stage label on the progress bar itself
        self._progress_bar.setFormat(f"%p%  —  {label}" if label else "%p%")
        self._stage_label.setText(label)
        # Record progress for ETA estimation (keep last 8 samples)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        now = time.monotonic()
        if not self._progress_history or value > self._progress_history[-1][1]:
            self._progress_history.append((now, value))
            if len(self._progress_history) > 8:
                self._progress_history = self._progress_history[-8:]
        # Window title: show batch info while running
        if label and ("[" in label and "]" in label):
            batch_tag = label[:label.index("]") + 1]
            self.setWindowTitle(f"FreeTrace v{_VERSION} — {batch_tag}")
        elif value >= 100:
            self.setWindowTitle(f"FreeTrace v{_VERSION}")

    def _update_elapsed(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        elapsed = time.monotonic() - self._run_start_time
        mins, secs = divmod(int(elapsed), 60)
        txt = f"{mins:02d}:{secs:02d}"
        # ETA based on recent progress rate (last few samples)
        hist = self._progress_history
        if len(hist) >= 2:
            t0, p0 = hist[0]
            t1, p1 = hist[-1]
            dp = p1 - p0
            dt = t1 - t0
            if dp > 0 and dt > 0:
                rate = dp / dt  # percent per second
                remaining_pct = 100 - p1
                eta_remaining = remaining_pct / rate
                if eta_remaining > 0:
                    rm, rs = divmod(int(eta_remaining), 60)
                    txt += f"  (~{rm:02d}:{rs:02d} left)"
        self._elapsed_label.setText(txt)

    def _on_finished(self, success: bool, message: str):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        self._run_timer.stop()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self.setWindowTitle(f"FreeTrace v{_VERSION}")
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
        if n_vids == 1:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            fname = self._loaded_datasets[0]['video_name']
            self._analysis_info_label.setText(
                f"Loaded {total} trajectories from '{fname}'. Draw a boundary to classify.")
        else:
            self._analysis_info_label.setText(
                f"Loaded {total} trajectories from {n_vids} videos. Draw a boundary to classify.")
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

    # ------------------------------------------------------------------  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    # ROI tab slots
    # ------------------------------------------------------------------
    def _on_roi_mode_changed(self, mode_text):
        self._roi_canvas.set_draw_mode(mode_text)

    def _on_roi_classify_mode_changed(self, mode_text):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        self._roi_canvas.set_classify_mode(mode_text)

    def _on_roi_load_data(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Load a single dataset (traces required, diffusion optional) for ROI tab."""
        start_dir = self._output_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FreeTrace output CSV (_traces.csv or _diffusion.csv)",
            start_dir,
            "FreeTrace CSV (*_traces.csv *_diffusion.csv);;All CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        self._roi_datasets = []
        self._roi_load_single(path)

    def _roi_load_single(self, selected_path):
        """Load a single video's data for ROI tab."""
        try:
            if '_traces.csv' in selected_path:
                traces_path = selected_path
                diffusion_path = selected_path.replace('_traces.csv', '_diffusion.csv')
            elif '_diffusion.csv' in selected_path:
                diffusion_path = selected_path
                traces_path = selected_path.replace('_diffusion.csv', '_traces.csv')
            else:
                QMessageBox.warning(self, "Unrecognized file",
                                    "Please select a file ending with _traces.csv or _diffusion.csv")
                return

            # Skip duplicates
            for ds in self._roi_datasets:
                if ds['traces_path'] == traces_path:
                    return

            if not os.path.exists(traces_path):
                QMessageBox.warning(self, "File not found",
                                    f"Traces file not found:\n{traces_path}")
                return

            traces_df = pd.read_csv(traces_path)
            required = {'traj_idx', 'frame', 'x', 'y'}
            if not required.issubset(traces_df.columns):
                missing = required - set(traces_df.columns)
                QMessageBox.warning(self, "Invalid traces file",
                                    f"Missing columns: {', '.join(sorted(missing))}")
                return

            # Diffusion is optional
            diffusion_df = None
            if os.path.exists(diffusion_path):
                df_diff = pd.read_csv(diffusion_path)
                if {'traj_idx', 'H', 'K'}.issubset(df_diff.columns):
                    diffusion_df = df_diff

            fname = os.path.basename(traces_path)
            video_name = fname.replace('_traces.csv', '')

            self._roi_datasets.append({
                'video_name': video_name,
                'traces_path': traces_path,
                'diffusion_path': diffusion_path if diffusion_df is not None else None,
                'traces_df': traces_df,
                'diffusion_df': diffusion_df,
            })

            self._rebuild_roi_canvas()
        except Exception as e:
            QMessageBox.critical(self, "Error loading ROI data", str(e))

    def _rebuild_roi_canvas(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Combine all loaded ROI datasets and update the ROI canvas."""
        if not self._roi_datasets:
            return

        # Build per-trajectory point lists and mean positions across all datasets
        all_mean_x, all_mean_y = [], []
        all_traj_points = []  # list of (xs, ys) per trajectory
        for ds in self._roi_datasets:
            tdf = ds['traces_df']
            for _, grp in tdf.groupby('traj_idx'):
                xs = grp['x'].values.astype(float)
                ys = grp['y'].values.astype(float)
                all_traj_points.append((xs, ys))
                all_mean_x.append(float(xs.mean()))
                all_mean_y.append(float(ys.mean()))

        combined_idx = np.arange(len(all_mean_x))
        combined_mean_x = np.array(all_mean_x)
        combined_mean_y = np.array(all_mean_y)

        self._roi_canvas.set_data(combined_idx, all_traj_points,
                                  combined_mean_x, combined_mean_y)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

        total = len(combined_mean_x)
        n_vids = len(self._roi_datasets)
        has_diff = any(ds['diffusion_df'] is not None for ds in self._roi_datasets)
        hk_str = " (with H-K data)" if has_diff else ""
        if n_vids == 1:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            fname = self._roi_datasets[0]['video_name']
            self._roi_info_label.setText(
                f"Loaded {total} trajectories from '{fname}'{hk_str}. Draw shapes to define ROIs.")
        else:
            self._roi_info_label.setText(
                f"Loaded {total} trajectories from {n_vids} videos{hk_str}. Draw shapes to define ROIs.")
        self._update_roi_stats()

    def _on_roi_clear(self):
        self._roi_canvas.clear_roi()

    def _on_roi_changed(self):
        """Called when ROI boundaries change — update stats and H-K scatter."""
        self._update_roi_stats()
        self._update_roi_hk_scatter()

    def _update_roi_stats(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Update the ROI stats panel with trajectory counts per ROI."""
        data = self._roi_canvas.get_roi_data()
        labels = data['labels']

        if not self._roi_datasets:
            self._roi_stats_label.setText("No data loaded.")
            return

        total = len(data['X'])
        if labels is None:
            self._roi_stats_label.setText(
                f"<b>{total}</b> trajectories loaded. Draw shapes to define ROIs.")
            return

        # Collect H, K arrays for per-ROI mean (if diffusion data available)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        all_H, all_K = [], []
        has_diff = False
        for ds in self._roi_datasets:
            if ds['diffusion_df'] is not None:
                has_diff = True
                tdf = ds['traces_df']
                ddf = ds['diffusion_df']
                means = tdf.groupby('traj_idx')[['x', 'y']].mean()
                merged = means.join(ddf.set_index('traj_idx')[['H', 'K']], how='left')
                all_H.append(merged['H'].values)
                all_K.append(merged['K'].values)
            else:
                n_traj = ds['traces_df']['traj_idx'].nunique()
                all_H.append(np.full(n_traj, np.nan))
                all_K.append(np.full(n_traj, np.nan))
        H_arr = np.concatenate(all_H) if all_H else np.array([])
        K_arr = np.concatenate(all_K) if all_K else np.array([])

        # Separate assigned (>=0) from excluded (<0)
        n_excluded = data.get('n_excluded', 0)
        assigned = labels[labels >= 0]
        unique = sorted(set(assigned.tolist())) if len(assigned) > 0 else []
        rows = ""
        colors = ROICanvas._ROI_COLORS
        for roi_id in unique:
            mask = labels == roi_id
            count = int(np.sum(mask))
            pct = count / total * 100 if total > 0 else 0
            c = colors[roi_id % len(colors)]
            cstr = f"rgb({c.red()},{c.green()},{c.blue()})"
            # Per-ROI mean H, K
            hk_cols = ""
            if has_diff and len(H_arr) == len(labels):
                h_vals = H_arr[mask]
                k_vals = K_arr[mask]
                h_valid = h_vals[~np.isnan(h_vals)]
                k_valid = k_vals[~np.isnan(k_vals)]
                mean_h = f"{np.mean(h_valid):.3f}" if len(h_valid) > 0 else "—"
                mean_k = f"{np.mean(k_valid):.2e}" if len(k_valid) > 0 else "—"
                hk_cols = (f"<td style='padding:4px 12px;'>{mean_h}</td>"
                           f"<td style='padding:4px 12px;'>{mean_k}</td>")
            rows += (f"<tr><td style='color:{cstr}; padding:4px 12px; font-weight:bold;'>"
                     f"roi{roi_id}</td>"
                     f"<td style='padding:4px 12px;'>{count}</td>"
                     f"<td style='padding:4px 12px;'>{pct:.1f}%</td>{hk_cols}</tr>")

        # Show excluded row if any
        excluded_html = ""
        if n_excluded > 0:
            pct_excl = n_excluded / total * 100 if total > 0 else 0
            hk_excl = ""
            if has_diff and len(H_arr) == len(labels):
                hk_excl = "<td style='padding:4px 12px;'>—</td><td style='padding:4px 12px;'>—</td>"
            rows += (f"<tr><td style='color:#666; padding:4px 12px; font-style:italic;'>"
                     f"excluded</td>"
                     f"<td style='padding:4px 12px;'>{n_excluded}</td>"
                     f"<td style='padding:4px 12px;'>{pct_excl:.1f}%</td>{hk_excl}</tr>")
            excluded_html = f" ({n_excluded} excluded — cross ROI boundaries)"

        hk_header = ""
        if has_diff:
            hk_header = ("<th style='padding:4px 12px;'>Mean H</th>"
                         "<th style='padding:4px 12px;'>Mean K</th>")
        html = (f"<p><b>{total}</b> trajectories, <b>{len(unique)}</b> ROI(s){excluded_html}</p>"
                "<table style='border-collapse:collapse;'>"
                f"<tr style='color:#888;'><th style='padding:4px 12px;'>ROI</th>"
                f"<th style='padding:4px 12px;'>Count</th>"
                f"<th style='padding:4px 12px;'>%</th>{hk_header}</tr>"
                f"{rows}</table>")
        self._roi_stats_label.setText(html)  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _update_roi_hk_scatter(self):
        """Render H-K scatter colored by ROI labels (if diffusion data available)."""
        self._roi_hk_scene.clear()

        data = self._roi_canvas.get_roi_data()
        labels = data['labels']

        # Collect H, K values aligned with ROI canvas order
        all_H, all_K = [], []
        has_diff = False
        for ds in self._roi_datasets:
            if ds['diffusion_df'] is not None:
                has_diff = True
                tdf = ds['traces_df']
                ddf = ds['diffusion_df']
                # Mean positions per traj (same order as ROI canvas)
                means = tdf.groupby('traj_idx')[['x', 'y']].mean()
                # Align diffusion H, K with trajectory order
                merged = means.join(ddf.set_index('traj_idx')[['H', 'K']], how='left')
                all_H.append(merged['H'].values)
                all_K.append(merged['K'].values)
            else:
                # No diffusion data for this dataset — fill NaN
                n_traj = ds['traces_df']['traj_idx'].nunique()
                all_H.append(np.full(n_traj, np.nan))
                all_K.append(np.full(n_traj, np.nan))

        if not has_diff:
            self._roi_hk_title.setText("H-K Scatter Colored by ROI (no diffusion data)")
            return

        self._roi_hk_title.setText("H-K Scatter Colored by ROI")
        H = np.concatenate(all_H)
        K = np.concatenate(all_K)
        valid = ~(np.isnan(H) | np.isnan(K))

        if not np.any(valid):
            return

        safe_K = np.clip(K, 1e-10, None)
        log_K = np.log10(safe_K)

        # Plot dimensions (mirror HKGatingCanvas layout)
        ML, MT, PW, PH = 120, 60, 1000, 800  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        MR, MB = 60, 100  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        total_w = ML + PW + MR
        total_h = MT + PH + MB
        self._roi_hk_scene.setSceneRect(0, 0, total_w, total_h)

        h_min, h_max = 0.0, 1.0
        logk_min = float(np.floor(np.nanmin(log_K[valid]) - 0.5))
        logk_max = float(np.ceil(np.nanmax(log_K[valid]) + 0.5))

        def h_to_x(h): return ML + (h - h_min) / (h_max - h_min) * PW
        def lk_to_y(lk): return MT + (1.0 - (lk - logk_min) / (logk_max - logk_min)) * PH

        # Background & grid
        pen_grid = QPen(QColor(60, 60, 60), 0.5, Qt.PenStyle.DashLine)
        pen_axis = QPen(QColor(150, 150, 150), 1.5)
        pen_text = QColor(180, 180, 180)
        scene_font = QFont()  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        scene_font.setPointSize(18)  # scaled for 1000x800 scene

        self._roi_hk_scene.addRect(QRectF(ML, MT, PW, PH),
                                   QPen(Qt.PenStyle.NoPen), QBrush(QColor(30, 30, 30)))

        for hv in np.arange(0.0, 1.01, 0.1):
            x = h_to_x(hv)
            self._roi_hk_scene.addLine(x, MT, x, MT + PH, pen_grid)
            t = self._roi_hk_scene.addSimpleText(f"{hv:.1f}", scene_font)
            t.setBrush(pen_text)
            t.setPos(x - 18, MT + PH + 8)

        for lkv in range(int(logk_min), int(logk_max) + 1):
            y = lk_to_y(lkv)
            self._roi_hk_scene.addLine(ML, y, ML + PW, y, pen_grid)
            t = self._roi_hk_scene.addSimpleText(f"1e{lkv}", scene_font)
            t.setBrush(pen_text)
            t.setPos(ML - 80, y - 12)

        self._roi_hk_scene.addLine(ML, MT + PH, ML + PW, MT + PH, pen_axis)
        self._roi_hk_scene.addLine(ML, MT, ML, MT + PH, pen_axis)

        xl = self._roi_hk_scene.addSimpleText("H (Hurst exponent)", scene_font)
        xl.setBrush(pen_text)
        xl.setPos(ML + PW / 2 - 100, MT + PH + 45)
        yl = self._roi_hk_scene.addSimpleText("K", scene_font)
        yl.setBrush(pen_text)
        yl.setPos(8, MT + PH / 2 - 12)

        # Render dots as pixmap
        pix = QPixmap(total_w, total_h)
        pix.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        colors = ROICanvas._ROI_COLORS
        default_color = QColor(180, 180, 180, 160)
        dot_r = 5.0  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

        for i in range(len(H)):
            if not valid[i]:
                continue
            x = h_to_x(H[i])
            y = lk_to_y(log_K[i])
            if x < ML or x > ML + PW or y < MT or y > MT + PH:
                continue
            if labels is not None:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                lbl = int(labels[i])
                if lbl < 0:
                    c = QColor(80, 80, 80, 60)  # excluded
                else:
                    c = colors[lbl % len(colors)]
            else:
                c = default_color
            painter.setBrush(QBrush(c))
            painter.drawEllipse(QPointF(x, y), dot_r, dot_r)

        painter.end()
        pix_item = self._roi_hk_scene.addPixmap(pix)
        pix_item.setZValue(1)

        self._roi_hk_canvas.fitInView(self._roi_hk_scene.sceneRect(),
                                      Qt.AspectRatioMode.KeepAspectRatio)

    def _on_roi_load_boundary(self):  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        """Load ROI shapes from a JSON, ImageJ .roi, or .zip file."""
        start_dir = self._output_dir or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ROI boundary file",
            start_dir,
            "All ROI files (*.json *.roi *.zip);;ROI JSON (*.json);;ImageJ ROI (*.roi);;ImageJ ROI ZIP (*.zip);;All files (*)"
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == '.roi' or ext == '.zip':
                shapes, skipped = _load_imagej_rois(path)
                if not shapes:
                    QMessageBox.warning(self, "No shapes",
                                        "No supported ROI shapes found in file."
                                        + (f"\n({skipped} unsupported ROI(s) skipped)" if skipped else ""))
                    return
                msg = f"Loaded {len(shapes)} ROI shape(s)"
                if skipped > 0:
                    msg += f" ({skipped} unsupported skipped)"
                self._roi_info_label.setText(msg)
                self._roi_canvas.set_shapes_data(shapes)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                shapes = data.get('shapes', [])
                if not shapes:
                    QMessageBox.warning(self, "No shapes", "No ROI shape data found in file.")
                    return
                self._roi_canvas.set_shapes_data(shapes)
        except Exception as e:
            QMessageBox.critical(self, "Error loading ROI", str(e))
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

    def _on_roi_export(self):
        """Export trajectories per ROI to CSV files."""
        data = self._roi_canvas.get_roi_data()
        if data['labels'] is None:
            QMessageBox.information(self, "No ROI",
                                    "Draw shapes first to define ROIs.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select export folder",
                                                     self._output_dir or "")
        if not save_dir:
            return

        try:
            labels = data['labels']
            offset = 0
            exported_files = []

            for ds in self._roi_datasets:
                tdf = ds['traces_df']
                means = tdf.groupby('traj_idx')[['x', 'y']].mean()
                n = len(means)
                ds_labels = labels[offset:offset + n]
                vname = ds['video_name']
                traj_ids_ordered = means.index.values

                unique_rois = sorted(set(ds_labels.tolist()))
                for roi_id in unique_rois:
                    if roi_id < 0:  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
                        suffix = "roi_excluded"
                    else:
                        suffix = f"roi_{roi_id}"
                    mask = ds_labels == roi_id
                    roi_traj_ids = set(traj_ids_ordered[mask].tolist())

                    # Export traces
                    traj_sub = tdf[tdf['traj_idx'].isin(roi_traj_ids)]
                    traj_sub.to_csv(
                        os.path.join(save_dir, f"{vname}_{suffix}_traces.csv"),
                        index=False
                    )

                    # Export diffusion if available
                    if ds['diffusion_df'] is not None:
                        diff_sub = ds['diffusion_df'][
                            ds['diffusion_df']['traj_idx'].isin(roi_traj_ids)]
                        diff_sub.to_csv(
                            os.path.join(save_dir, f"{vname}_{suffix}_diffusion.csv"),
                            index=False
                        )

                    exported_files.append(f"{vname}_{suffix}")

                offset += n

            # Save shape information
            shapes_data = self._roi_canvas.get_shapes_data()
            shapes_path = os.path.join(save_dir, "roi_boundaries.json")
            with open(shapes_path, 'w') as f:
                json.dump({'shapes': shapes_data}, f, indent=2)

            QMessageBox.information(self, "Export complete",
                                    f"Exported {len(exported_files)} ROI files "
                                    f"from {len(self._roi_datasets)} video(s) to:\n{save_dir}\n\n"
                                    f"Shapes saved to: roi_boundaries.json")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))
    # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

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
    # Ensure Ctrl+C in terminal cleanly shuts down the GUI and child processes  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Timer allows Python to process signals (SIGINT) between Qt events
    _sig_timer = QTimer()
    _sig_timer.start(200)
    _sig_timer.timeout.connect(lambda: None)
    win = FreeTraceGUI()
    win.show()
    ret = app.exec()
    # Cleanup temp arrow icons
    import shutil as _shutil
    _shutil.rmtree(_arrow_dir, ignore_errors=True)
    sys.exit(ret) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16


if __name__ == "__main__":
    main()
