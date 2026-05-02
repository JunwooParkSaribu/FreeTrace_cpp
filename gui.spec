# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
# PyInstaller spec file for building FreeTrace GUI (one-folder mode for fast startup)
# Usage: pyinstaller gui.spec
# Requires: pip install pyinstaller PyQt6
#
# One-folder mode avoids temp-extraction on every launch, which dramatically
# improves startup time on Windows (especially with Windows Defender).

import os

block_cipher = None
script_dir = SPECPATH

a = Analysis(
    [os.path.join(script_dir, 'gui.py')],
    pathex=[script_dir],
    binaries=[],
    datas=[
        (os.path.join(script_dir, 'icon'), 'icon'),
        # Destination "freetrace_python" (not "python") to avoid case-insensitive
        # filesystem collisions on macOS/Windows with PyInstaller's bundled Python binary.
        (os.path.join(script_dir, 'python'), 'freetrace_python'),  # cauchy_fit.py + cauchy_neff_cov_H*.npz cov tables
    ],
    hiddenimports=['PyQt6.sip', 'numpy', 'pandas', 'tifffile', 'nd2', 'matplotlib', 'seaborn', 'scipy', 'scipy.optimize', 'google.genai', 'google.genai.types'],  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-22
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        # ---- Heavy deps the C++ GUI does NOT use (avoid bloat); fBm inference is done // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-05-02
        # by the freetrace.exe binary via ONNX Runtime, not Python TF/keras.
        'tensorflow', 'tensorflow_cpu_aws', 'tensorflow_intel',
        'tf_keras', 'keras', 'tensorboard', 'tensorboard_data_server',
        'jax', 'jaxlib', 'flax', 'optax',
        'torch', 'torchvision', 'torchaudio',
        'numba', 'llvmlite',
        'cupy', 'cupy_backends',
        'sklearn', 'skimage',
        'cv2', 'opencv_python', 'opencv_contrib_python',
        'notebook', 'jupyter', 'IPython', 'ipykernel', 'ipywidgets',
        'pytest', 'pylint', 'mypy', 'black', 'flake8',
        'nvidia',  # NVIDIA pip wheels (CUDA libs); Mac doesn't need them, Windows ships its own DLLs
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FreeTrace_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=os.path.join(script_dir, 'icon', 'freetrace_icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='FreeTrace_GUI',
)
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
