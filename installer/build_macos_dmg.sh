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

# Copy GUI
cp "$PROJECT_DIR/gui.py" "$STAGING/FreeTrace/" 2>/dev/null || true

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

# --- Step 6: Create README ---
cat > "$STAGING/FreeTrace/README.txt" << README
FreeTrace v${VERSION} — macOS (Apple Silicon)

Single-molecule tracking software with fBm inference.

USAGE:
  ./freetrace <input.tiff> <output_dir> [options]
  ./freetrace batch <input_folder> <output_dir> [options]

GUI (requires Python 3.10+ and PyQt6):
  pip install PyQt6
  python gui.py

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

# --- Step 8: Create .dmg ---
echo "Creating .dmg..."
DMG_PATH="$PROJECT_DIR/$DMG_NAME.dmg"
rm -f "$DMG_PATH"

hdiutil create \
    -volname "FreeTrace" \
    -srcfolder "$STAGING" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

# Cleanup
rm -rf "$STAGING" "$BUILD_DIR"

echo ""
echo "=== Done ==="
echo "DMG: $DMG_PATH"
echo "Size: $(du -h "$DMG_PATH" | cut -f1)"
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
