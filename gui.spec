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
    datas=[(os.path.join(script_dir, 'icon'), 'icon')],
    hiddenimports=['PyQt6.sip', 'numpy', 'pandas', 'tifffile', 'nd2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'PIL',
        'tkinter',
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
