# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
# PyInstaller spec file for building FreeTrace.app on macOS
# Usage: pyinstaller gui_macos.spec
# Requires: pip install pyinstaller PyQt6

import os
import sys

block_cipher = None
script_dir = SPECPATH

# Check for .icns icon, fall back to .png
icns_path = os.path.join(script_dir, 'icon', 'freetrace_icon.icns')
png_path = os.path.join(script_dir, 'icon', 'freetrace_icon.png')
icon_file = icns_path if os.path.exists(icns_path) else png_path

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
        # by the freetrace binary via ONNX Runtime, not Python TF/keras.
        'tensorflow', 'tensorflow_cpu_aws', 'tensorflow_intel', 'tensorflow_macos',
        'tf_keras', 'keras', 'tensorboard', 'tensorboard_data_server',
        'jax', 'jaxlib', 'flax', 'optax',
        'torch', 'torchvision', 'torchaudio',
        'numba', 'llvmlite',
        'cupy', 'cupy_backends',
        'sklearn', 'skimage',
        'cv2', 'opencv_python', 'opencv_contrib_python',
        'notebook', 'jupyter', 'IPython', 'ipykernel', 'ipywidgets',
        'pytest', 'pylint', 'mypy', 'black', 'flake8',
        'nvidia',  # NVIDIA pip wheels (CUDA libs); not needed on macOS
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
    name='FreeTrace',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='FreeTrace',
)

app = BUNDLE(
    coll,
    name='FreeTrace.app',
    icon=icon_file,
    bundle_identifier='fr.sorbonne-universite.freetrace',
    info_plist={
        'CFBundleDisplayName': 'FreeTrace',
        'CFBundleShortVersionString': '1.6.3.0',  # Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
        'NSHighResolutionCapable': True,
    },
)
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
