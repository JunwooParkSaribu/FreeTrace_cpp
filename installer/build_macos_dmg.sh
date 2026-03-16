#!/bin/bash
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
# Build FreeTrace macOS .dmg installer (Apple Silicon)
# Everything is bundled inside a single FreeTrace.app — double-click to launch.
#
# Usage:
#   ./installer/build_macos_dmg.sh [--ort-dir <path>]
#
# Prerequisites:
#   - macOS with Xcode command-line tools
#   - Homebrew: brew install libtiff libpng
#   - ONNX Runtime ARM64 downloaded and extracted
#
# If --ort-dir is not specified, the script downloads ONNX Runtime automatically.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION="1.6.0.4"
ORT_VERSION="1.24.3"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-arm64-${ORT_VERSION}.tgz"
ORT_DIR=""
DMG_NAME="FreeTrace_v${VERSION}_macos_arm64"
STAGING="$PROJECT_DIR/dmg_staging"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ort-dir) ORT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== FreeTrace macOS .dmg Builder ==="
echo "Version: $VERSION"
echo "Project: $PROJECT_DIR"

# --- Step 1: Download ONNX Runtime if not provided ---
if [[ -z "$ORT_DIR" ]]; then
    ORT_DIR="$PROJECT_DIR/onnxruntime-osx-arm64-${ORT_VERSION}"
    if [[ ! -d "$ORT_DIR" ]]; then
        echo "Downloading ONNX Runtime ${ORT_VERSION}..."
        curl -L -o "$PROJECT_DIR/ort.tgz" "$ORT_URL"
        tar xzf "$PROJECT_DIR/ort.tgz" -C "$PROJECT_DIR"
        rm "$PROJECT_DIR/ort.tgz"
    fi
fi
echo "ONNX Runtime: $ORT_DIR"

# --- Step 2: Build FreeTrace binary ---
echo "Building FreeTrace..."
BUILD_DIR="$PROJECT_DIR/build_dmg"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ONNXRUNTIME=ON \
    -DONNXRUNTIME_DIR="$ORT_DIR"

make -j$(sysctl -n hw.ncpu)

# --- Step 3: Build GUI .app with PyInstaller ---
echo "Building standalone GUI app..."
rm -rf "$STAGING"

# Convert PNG icon to icns if iconutil is available
if command -v iconutil &>/dev/null && [ -f "$PROJECT_DIR/icon/freetrace_icon.png" ]; then
    ICONSET="$PROJECT_DIR/icon/freetrace_icon.iconset"
    mkdir -p "$ICONSET"
    sips -z 16 16     "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_16x16.png" 2>/dev/null
    sips -z 32 32     "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_16x16@2x.png" 2>/dev/null
    sips -z 32 32     "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_32x32.png" 2>/dev/null
    sips -z 64 64     "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_32x32@2x.png" 2>/dev/null
    sips -z 128 128   "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_128x128.png" 2>/dev/null
    sips -z 256 256   "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_128x128@2x.png" 2>/dev/null
    sips -z 256 256   "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_256x256.png" 2>/dev/null
    sips -z 512 512   "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_256x256@2x.png" 2>/dev/null
    sips -z 512 512   "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_512x512.png" 2>/dev/null
    sips -z 1024 1024 "$PROJECT_DIR/icon/freetrace_icon.png" --out "$ICONSET/icon_512x512@2x.png" 2>/dev/null
    iconutil -c icns "$ICONSET" -o "$PROJECT_DIR/icon/freetrace_icon.icns" 2>/dev/null
    rm -rf "$ICONSET"
fi

# Build GUI with PyInstaller
APP_BUILT=false
if command -v python3 &>/dev/null; then
    python3 -m pip install pyinstaller PyQt6 --quiet 2>/dev/null || true
    cd "$PROJECT_DIR"
    if python3 -m PyInstaller gui_macos.spec --noconfirm --clean 2>&1; then
        if [ -d "$PROJECT_DIR/dist/FreeTrace.app" ]; then
            APP_BUILT=true
            echo "GUI app built successfully."
        fi
    fi
fi

if [ "$APP_BUILT" = false ]; then
    echo "ERROR: PyInstaller build failed. Cannot create .app bundle."
    rm -rf "$PROJECT_DIR/dist" "$PROJECT_DIR/build"
    exit 1
fi

# --- Step 4: Bundle everything inside the .app ---
echo "Bundling binary, models, and libraries inside .app..."
APP_DIR="$PROJECT_DIR/dist/FreeTrace.app"
MACOS_DIR="$APP_DIR/Contents/MacOS"
RESOURCES_DIR="$APP_DIR/Contents/Resources"

# Copy freetrace binary into Contents/MacOS/ (renamed to avoid case collision with PyInstaller exe)
cp "$BUILD_DIR/freetrace" "$MACOS_DIR/freetrace-bin"
chmod +x "$MACOS_DIR/freetrace-bin"

# Copy models into Contents/Resources/models/
mkdir -p "$RESOURCES_DIR/models"
cp "$PROJECT_DIR"/models/*.onnx "$RESOURCES_DIR/models/" 2>/dev/null || true
cp "$PROJECT_DIR"/models/*.bin "$RESOURCES_DIR/models/" 2>/dev/null || true

# Copy ONNX Runtime dylibs into Contents/MacOS/lib/
mkdir -p "$MACOS_DIR/lib"
cp "$ORT_DIR"/lib/libonnxruntime*.dylib "$MACOS_DIR/lib/"

# --- Step 5: Fix dylib paths ---
echo "Fixing dylib paths..."
for dylib in "$MACOS_DIR/lib"/libonnxruntime*.dylib; do
    dylib_name=$(basename "$dylib")
    install_name_tool -change \
        "@rpath/$dylib_name" \
        "@executable_path/lib/$dylib_name" \
        "$MACOS_DIR/freetrace-bin" 2>/dev/null || true
    install_name_tool -change \
        "$ORT_DIR/lib/$dylib_name" \
        "@executable_path/lib/$dylib_name" \
        "$MACOS_DIR/freetrace-bin" 2>/dev/null || true
    install_name_tool -id \
        "@executable_path/lib/$dylib_name" \
        "$dylib" 2>/dev/null || true
done

echo "Verifying dylib references:"
otool -L "$MACOS_DIR/freetrace-bin" | grep -i onnx || echo "(no ONNX Runtime references found)"

# --- Step 6: Ad-hoc code sign ---
echo "Code signing (ad-hoc)..."
codesign --force --sign - "$MACOS_DIR/freetrace-bin"
for dylib in "$MACOS_DIR/lib"/*.dylib; do
    codesign --force --sign - "$dylib"
done
codesign --force --deep --sign - "$APP_DIR"

# --- Step 7: Stage for DMG ---
echo "Staging DMG contents..."
mkdir -p "$STAGING"
ditto "$APP_DIR" "$STAGING/FreeTrace.app"

# Create Applications symlink for drag-and-drop install
ln -s /Applications "$STAGING/Applications"

# Add README
cat > "$STAGING/README.txt" << README
FreeTrace v${VERSION} — macOS (Apple Silicon)

Single-molecule tracking software with fBm inference.

Installation:
  Drag "FreeTrace.app" to Applications.
  Double-click to launch.

CLI (optional):
  "/Applications/FreeTrace.app/Contents/MacOS/freetrace-bin" <input.tiff> <output_dir>

If macOS blocks the app (Gatekeeper):
  Right-click → Open, or run:
  xattr -cr "/Applications/FreeTrace.app"

https://github.com/JunwooParkSaribu/FreeTrace_cpp
README

rm -rf "$PROJECT_DIR/dist" "$PROJECT_DIR/build"

# --- Step 8: Create .dmg ---
echo "Creating .dmg..."
DMG_PATH="$PROJECT_DIR/$DMG_NAME.dmg"
rm -f "$DMG_PATH"

echo "Staging contents:"
ls -la "$STAGING/"

CONTENT_SIZE_KB=$(du -sk "$STAGING" | cut -f1)
DMG_SIZE_MB=$(( (CONTENT_SIZE_KB / 1024) + 100 ))
echo "Content: ${CONTENT_SIZE_KB}KB, DMG volume: ${DMG_SIZE_MB}MB"

MOUNT_POINT="$PROJECT_DIR/_dmg_mount_$$"
mkdir -p "$MOUNT_POINT"

hdiutil create -volname "FreeTrace" -size "${DMG_SIZE_MB}m" -fs HFS+ -ov "$DMG_PATH.rw.dmg"
hdiutil attach "$DMG_PATH.rw.dmg" -mountpoint "$MOUNT_POINT" -nobrowse
ditto "$STAGING/" "$MOUNT_POINT/"
echo "Mounted DMG contents:"
ls -la "$MOUNT_POINT/"
hdiutil detach "$MOUNT_POINT"
hdiutil convert "$DMG_PATH.rw.dmg" -format UDZO -o "$DMG_PATH"
rm -f "$DMG_PATH.rw.dmg"
rmdir "$MOUNT_POINT" 2>/dev/null || true

# Cleanup
rm -rf "$STAGING" "$BUILD_DIR"

echo ""
echo "=== Done ==="
echo "DMG: $DMG_PATH"
echo "Size: $(du -h "$DMG_PATH" | cut -f1)"
echo ""
echo "DMG contents: FreeTrace.app (drag to Applications to install)"
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
