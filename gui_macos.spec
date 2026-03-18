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
    datas=[(os.path.join(script_dir, 'icon'), 'icon')],
    hiddenimports=['PyQt6.sip', 'numpy', 'pandas', 'tifffile', 'nd2'], # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'PIL', # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
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
        'CFBundleShortVersionString': '1.6.1.0',  # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-18
        'NSHighResolutionCapable': True,
    },
)
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
