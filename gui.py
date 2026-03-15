# Made by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
# FreeTrace GUI — launches the freetrace binary with a graphical interface.
# Launch with:  python gui.py
"""
FreeTrace GUI — run localization and tracking by clicking.
Requires PyQt6: pip install PyQt6
"""
import math
import os
import sys
import subprocess
import shutil

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, QProcess
from PyQt6.QtGui import QPixmap, QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
    QDoubleSpinBox, QSpinBox, QFileDialog, QTextEdit, QSplitter,
    QTabWidget, QScrollArea, QProgressBar, QMessageBox, QRadioButton,
    QButtonGroup,
)

# Base window size — font sizes are defined relative to this
_BASE_W, _BASE_H = 1920, 1080


def _find_freetrace_binary():
    """Find the freetrace binary. Checks: same dir as exe, build/, PATH."""
    # When frozen by PyInstaller, __file__ points to a temp dir (_MEIPASS).
    # Use sys.executable dir so gui.exe finds freetrace.exe next to itself. # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    if getattr(sys, 'frozen', False):
        script_dir = os.path.dirname(sys.executable)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "freetrace"),
        os.path.join(script_dir, "freetrace.exe"),
        os.path.join(script_dir, "build", "freetrace"),
        os.path.join(script_dir, "build", "freetrace.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    # Try PATH
    found = shutil.which("freetrace")
    if found:
        return found
    return None


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

    def cancel(self):
        self._cancel = True
        if self._process and self._process.poll() is None:
            self._process.terminate()

    def run(self):
        try:
            cmd = [self.binary] + self.args
            self.log.emit(f"$ {' '.join(cmd)}")
            self.log.emit("")
            self.progress.emit(5, "Running...")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in self._process.stdout:
                line = line.rstrip("\n")
                self.log.emit(line)

                # Parse progress from output
                if "Localization" in line and "===" in line:
                    self.progress.emit(10, "Localization")
                elif "Tracking" in line and "===" in line:
                    self.progress.emit(55, "Tracking")
                elif "Batch complete" in line:
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
# Main window
# ---------------------------------------------------------------------------
class FreeTraceGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FreeTrace")
        self.setMinimumSize(_BASE_W, _BASE_H)
        self._worker = None
        self._output_dir = None
        self._result_widgets = []
        self._binary = _find_freetrace_binary()
        # Debounce timer
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._apply_fonts)
        self._setup_ui()
        self._apply_fonts()

    # ------------------------------------------------------------------
    # Scale helpers
    # ------------------------------------------------------------------
    def _scale(self) -> float:
        s = math.sqrt(self.width() * self.height()) / math.sqrt(_BASE_W * _BASE_H)
        return max(0.8, min(2.5, s))

    def _f(self, base_px: int) -> int:
        return max(8, round(base_px * self._scale()))

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        root.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([380, 670])

    # ---- left panel (controls) ----------------------------------------
    def _build_left_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(420)
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

        self._postprocess = QCheckBox("Post-processing")
        self._postprocess.setChecked(False)
        self._postprocess.setToolTip("Enable trajectory post-processing.")
        adv_grid.addWidget(self._postprocess, 3, 0, 1, 2)

        self._adv_sec.add_layout(adv_grid)
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
        try:
            self._no_results_label.setStyleSheet(
                f"color:#666; font-size:{f(15)}px; margin:40px;"
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

        if self._postprocess.isChecked():
            args.append("--postprocess")

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

    def _load_results(self, output_dir: str):
        # Clear previous dynamic result widgets
        for w in self._result_widgets:
            self._results_layout.removeWidget(w)
            w.deleteLater()
        self._result_widgets.clear()

        image_suffixes = {
            "Trajectory Map": "_traces.png",
            "Localisation Density": "_loc_2d_density.png",
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
def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    app.setApplicationName("FreeTrace")
    win = FreeTraceGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
