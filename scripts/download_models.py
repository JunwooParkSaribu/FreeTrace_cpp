#!/usr/bin/env python3
"""Download FreeTrace models and convert to ONNX for the C++ version. # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

Usage:
    python3 scripts/download_models.py [--output-dir models]

Downloads the Keras models from the FreeTrace server, converts the needed
ones (reg_model_3, reg_model_5, reg_model_8, reg_k_model) to ONNX format,
and also converts qt_99.npz to qt_99.bin.

Requirements:
    pip install tensorflow keras tf2onnx onnx ml_dtypes>=0.5.0
"""

import argparse
import os
import sys
import shutil
import tempfile
import zipfile
import urllib.request

# Same URLs as FreeTrace Python (model_downloader.py)
_MODEL_URLS = {
    "3.10": "https://psilo.sorbonne-universite.fr/index.php/s/o8SZrWt4HePY8js/download/models_2_14.zip",
    "default": "https://psilo.sorbonne-universite.fr/index.php/s/w9PrAQbxsNJrEFc/download/models_2_17.zip",
}

# Models needed by C++ (trajectory lengths used in tracking)
NEEDED_ALPHA_MODELS = [3, 5, 8]
NEEDED_K_MODEL = "reg_k_model"


def get_model_url():
    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    return _MODEL_URLS.get(minor, _MODEL_URLS["default"])


def download_with_progress(url, dest):
    """Download a file with a progress bar."""
    print(f"  Downloading from:\n    {url}")
    import ssl
    try:
        req = urllib.request.urlopen(url)
    except urllib.error.URLError:
        # Fallback: disable SSL verification (some systems lack certs)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.urlopen(url, context=ctx)
    total = int(req.headers.get("Content-Length", 0))
    downloaded = 0
    block_size = 1024 * 64

    with open(dest, "wb") as f:
        while True:
            buf = req.read(block_size)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total > 0:
                pct = downloaded * 100 // total
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)
    print()


def extract_zip(zip_path, dest_dir):
    """Extract zip, handling nested directory structure."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # If extracted into a single subdirectory, move contents up
    items = os.listdir(dest_dir)
    if len(items) == 1 and os.path.isdir(os.path.join(dest_dir, items[0])):
        nested = os.path.join(dest_dir, items[0])
        for f in os.listdir(nested):
            shutil.move(os.path.join(nested, f), os.path.join(dest_dir, f))
        os.rmdir(nested)


def convert_keras_to_onnx(keras_path, onnx_path, input_shape):
    """Convert a Keras model to ONNX format."""
    import numpy as np
    from keras.models import load_model

    print(f"  Loading  {os.path.basename(keras_path)} ...")
    model = load_model(keras_path)

    # Call model once (required before export)
    dummy = np.zeros(input_shape, dtype=np.float32)
    model(dummy)

    print(f"  Exporting {os.path.basename(onnx_path)} ...")
    model.export(onnx_path, format="onnx")
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"    -> {size_mb:.1f} MB")


def convert_qt99(npz_path, bin_path):
    """Convert qt_99.npz to the binary format expected by C++."""
    import numpy as np
    import struct

    data = np.load(npz_path)
    qt = data["qt_99"]  # shape: (n_lengths, n_quantiles) float64
    rows, cols = qt.shape

    with open(bin_path, "wb") as f:
        f.write(struct.pack("<II", rows, cols))
        for r in range(rows):
            for c in range(cols):
                f.write(struct.pack("<d", float(qt[r, c])))

    print(f"  Converted qt_99.npz -> qt_99.bin ({rows}x{cols})")


def main():
    parser = argparse.ArgumentParser(description="Download and convert FreeTrace models to ONNX")
    parser.add_argument("--output-dir", default="models", help="Output directory (default: models)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check if models already exist
    needed_files = [f"reg_model_{n}.onnx" for n in NEEDED_ALPHA_MODELS] + [f"{NEEDED_K_MODEL}.onnx"]
    existing = [f for f in needed_files if os.path.exists(os.path.join(output_dir, f))]
    if len(existing) == len(needed_files):
        print("All ONNX models already exist. Nothing to do.")
        print("  To re-download, delete the .onnx files first.")
        return

    # Download
    print("\n=== Step 1: Download models ===")
    url = get_model_url()
    tmp_dir = tempfile.mkdtemp(prefix="freetrace_models_")
    zip_path = os.path.join(tmp_dir, "models.zip")

    try:
        download_with_progress(url, zip_path)

        print("  Extracting...")
        extract_dir = os.path.join(tmp_dir, "extracted")
        os.makedirs(extract_dir)
        extract_zip(zip_path, extract_dir)

        # List extracted files
        keras_files = [f for f in os.listdir(extract_dir) if f.endswith(".keras")]
        npz_files = [f for f in os.listdir(extract_dir) if f.endswith(".npz")]
        print(f"  Found {len(keras_files)} .keras models, {len(npz_files)} .npz files")

        # Convert Keras models to ONNX
        print("\n=== Step 2: Convert to ONNX ===")

        # Suppress TF warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        for n in NEEDED_ALPHA_MODELS:
            keras_file = f"reg_model_{n}.keras"
            onnx_file = f"reg_model_{n}.onnx"
            keras_path = os.path.join(extract_dir, keras_file)
            onnx_path = os.path.join(output_dir, onnx_file)

            if os.path.exists(onnx_path):
                print(f"  {onnx_file} already exists, skipping")
                continue

            if not os.path.exists(keras_path):
                print(f"  WARNING: {keras_file} not found in download, skipping")
                continue

            # Input shape: (batch, seq_len, 1, 3)
            convert_keras_to_onnx(keras_path, onnx_path, (1, n, 1, 3))

        # K model
        k_keras = os.path.join(extract_dir, f"{NEEDED_K_MODEL}.keras")
        k_onnx = os.path.join(output_dir, f"{NEEDED_K_MODEL}.onnx")
        if os.path.exists(k_onnx):
            print(f"  {NEEDED_K_MODEL}.onnx already exists, skipping")
        elif os.path.exists(k_keras):
            convert_keras_to_onnx(k_keras, k_onnx, (1, 1))
        else:
            print(f"  WARNING: {NEEDED_K_MODEL}.keras not found")

        # Convert qt_99.npz if present and qt_99.bin doesn't exist
        print("\n=== Step 3: Convert data files ===")
        qt_npz = os.path.join(extract_dir, "qt_99.npz")
        qt_bin = os.path.join(output_dir, "qt_99.bin")
        if os.path.exists(qt_bin):
            print(f"  qt_99.bin already exists, skipping")
        elif os.path.exists(qt_npz):
            convert_qt99(qt_npz, qt_bin)
        else:
            print(f"  WARNING: qt_99.npz not found in download")

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Summary
    print("\n=== Done ===")
    for f in needed_files:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK  {f} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  {f}")
    print() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13


if __name__ == "__main__":
    main()
