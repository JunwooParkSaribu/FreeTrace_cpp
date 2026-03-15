# Made by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
# PyInstaller spec file for building gui.exe
# Usage: pyinstaller gui.spec
# Requires: pip install pyinstaller PyQt6

import os

block_cipher = None
script_dir = SPECPATH

a = Analysis(
    [os.path.join(script_dir, 'gui.py')],
    pathex=[script_dir],
    binaries=[],
    datas=[],
    hiddenimports=['PyQt6.sip'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'numpy', 'scipy', 'pandas', 'PIL',
        'tkinter', 'unittest', 'xml', 'email', 'http',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FreeTrace_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,           # No console window — it's a GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,               # Set to 'icon.ico' if you have one
)
