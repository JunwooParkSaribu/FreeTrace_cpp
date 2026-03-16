#!/bin/bash
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
# Build FreeTrace macOS .dmg installer (Apple Silicon)
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

# --- Step 2: Build FreeTrace ---
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

# --- Step 3: Create staging directory ---
echo "Staging files..."
rm -rf "$STAGING"
mkdir -p "$STAGING/FreeTrace"
mkdir -p "$STAGING/FreeTrace/lib"
mkdir -p "$STAGING/FreeTrace/models"

# Copy binary
cp "$BUILD_DIR/freetrace" "$STAGING/FreeTrace/"
chmod +x "$STAGING/FreeTrace/freetrace"

# Copy models
cp "$PROJECT_DIR"/models/*.onnx "$STAGING/FreeTrace/models/" 2>/dev/null || true
cp "$PROJECT_DIR"/models/*.bin "$STAGING/FreeTrace/models/" 2>/dev/null || true

# Copy ONNX Runtime dylibs
cp "$ORT_DIR"/lib/libonnxruntime*.dylib "$STAGING/FreeTrace/lib/"

# --- Step 3b: Build standalone GUI app --- # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
echo "Building standalone GUI app..."

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
GUI_BUILT=false
if command -v python3 &>/dev/null; then
    python3 -m pip install pyinstaller PyQt6 --quiet 2>/dev/null
    if python3 -m PyInstaller "$PROJECT_DIR/gui_macos.spec" \
        --noconfirm --clean \
        --distpath "$PROJECT_DIR/dist_gui" \
        --workpath "$PROJECT_DIR/build_pyinstaller" 2>&1; then
        if [ -d "$PROJECT_DIR/dist_gui/FreeTrace GUI.app" ]; then
            cp -R "$PROJECT_DIR/dist_gui/FreeTrace GUI.app" "$STAGING/FreeTrace GUI.app"
            GUI_BUILT=true
            echo "GUI app built successfully."
        fi
    fi
    rm -rf "$PROJECT_DIR/dist_gui" "$PROJECT_DIR/build_pyinstaller"
fi

if [ "$GUI_BUILT" = false ]; then
    echo "WARNING: Could not build GUI app. Including gui.py as fallback."
    cp "$PROJECT_DIR/gui.py" "$STAGING/FreeTrace/" 2>/dev/null || true
fi # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

# --- Step 4: Fix dylib paths ---
echo "Fixing dylib paths..."

# Fix the binary's reference to libonnxruntime
for dylib in "$STAGING/FreeTrace/lib"/libonnxruntime*.dylib; do
    dylib_name=$(basename "$dylib")
    # Change the binary's reference to use @executable_path/lib/
    install_name_tool -change \
        "@rpath/$dylib_name" \
        "@executable_path/lib/$dylib_name" \
        "$STAGING/FreeTrace/freetrace" 2>/dev/null || true
    # Also try the absolute path variant
    install_name_tool -change \
        "$ORT_DIR/lib/$dylib_name" \
        "@executable_path/lib/$dylib_name" \
        "$STAGING/FreeTrace/freetrace" 2>/dev/null || true
    # Update the dylib's own install name
    install_name_tool -id \
        "@executable_path/lib/$dylib_name" \
        "$dylib" 2>/dev/null || true
done

# Verify
echo "Verifying dylib references:"
otool -L "$STAGING/FreeTrace/freetrace" | grep -i onnx || echo "(no ONNX Runtime references — statically linked or not found)"

# --- Step 5: Create wrapper script ---
cat > "$STAGING/FreeTrace/run_freetrace.sh" << 'WRAPPER'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export DYLD_LIBRARY_PATH="$DIR/lib:$DYLD_LIBRARY_PATH"
exec "$DIR/freetrace" "$@"
WRAPPER
chmod +x "$STAGING/FreeTrace/run_freetrace.sh"

# --- Step 6: Create README --- # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
cat > "$STAGING/FreeTrace/README.txt" << README
FreeTrace v${VERSION} — macOS (Apple Silicon)

Single-molecule tracking software with fBm inference.

GUI:
  Double-click "FreeTrace GUI.app" to launch the graphical interface.

CLI:
  ./FreeTrace/freetrace <input.tiff> <output_dir> [options]
  ./FreeTrace/freetrace batch <input_folder> <output_dir> [options]

For full documentation: https://github.com/JunwooParkSaribu/FreeTrace_cpp

If macOS blocks the app (Gatekeeper):
  Right-click → Open, or run:
  xattr -cr /path/to/FreeTrace/
README

# --- Step 7: Ad-hoc code sign ---
echo "Code signing (ad-hoc)..."
codesign --force --sign - "$STAGING/FreeTrace/freetrace"
for dylib in "$STAGING/FreeTrace/lib"/*.dylib; do
    codesign --force --sign - "$dylib"
done
# Sign the GUI app if it exists # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
if [ -d "$STAGING/FreeTrace GUI.app" ]; then
    codesign --force --deep --sign - "$STAGING/FreeTrace GUI.app"
fi

# --- Step 8: Create .dmg ---
echo "Creating .dmg..."
DMG_PATH="$PROJECT_DIR/$DMG_NAME.dmg"
rm -f "$DMG_PATH"

# Calculate required size (content + 50MB headroom) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
CONTENT_SIZE_KB=$(du -sk "$STAGING" | cut -f1)
DMG_SIZE_MB=$(( (CONTENT_SIZE_KB / 1024) + 50 ))
echo "Content: ${CONTENT_SIZE_KB}KB, DMG volume: ${DMG_SIZE_MB}MB"

# Create a read-write DMG first, copy files, then convert to compressed
hdiutil create -volname "FreeTrace" -size "${DMG_SIZE_MB}m" -fs HFS+ -ov "$DMG_PATH.rw.dmg"
hdiutil attach "$DMG_PATH.rw.dmg" -mountpoint /tmp/freetrace_dmg_mount
cp -R "$STAGING"/* /tmp/freetrace_dmg_mount/
hdiutil detach /tmp/freetrace_dmg_mount
hdiutil convert "$DMG_PATH.rw.dmg" -format UDZO -o "$DMG_PATH"
rm -f "$DMG_PATH.rw.dmg"

# Cleanup
rm -rf "$STAGING" "$BUILD_DIR"

echo ""
echo "=== Done ==="
echo "DMG: $DMG_PATH"
echo "Size: $(du -h "$DMG_PATH" | cut -f1)"
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
