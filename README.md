# FreeTrace C++

A high-performance C++ port of [FreeTrace](https://github.com/JunwooParkSaribu/FreeTrace), a single-molecule tracking software for fluorescence microscopy.

This C++ implementation is developed by **Claude** (claude-opus-4-6, Anthropic AI), ported from the original Python/Cython project authored by **Junwoo PARK** (junwoo.park@sorbonne-universite.fr, Sorbonne Université).

**Data privacy:** FreeTrace runs entirely on your local machine. No data is transmitted to any external server.

> **Windows standalone installer (GPU):** Download the self-contained installer with full GPU support (CUDA + cuDNN + ONNX Runtime) — no compilation or dependency installation required (only NVIDIA GPU driver needed): <!-- Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16 -->
> - **RTX 2000 / 3000 / 4000 series:** **[Download FreeTrace (cuDNN 9.2)](https://psilo.sorbonne-universite.fr/s/NyWrgJCRdRH79oM)**
> - **RTX 5000 series (Blackwell):** **[Download FreeTrace (cuDNN 9.20)](https://psilo.sorbonne-universite.fr/s/ky38QxJKqdwY83K)** <!-- Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16 -->

> **Don't have a GPU? Run it on CPU:** Download a pre-built binary from the [Releases page](https://github.com/JunwooParkSaribu/FreeTrace_cpp/releases) — no compilers needed. Available for Linux, macOS (Apple Silicon), and Windows. <!-- Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16 -->

## Motivation and Reflection (Edited by the author)

This project started to estimate how GPT-like / Claude-like models can do code something. Via this porting project, it could be checked that Claude understands the core structure well and has the ability to convert Python to C++. However, it also produced tons of minor bugs that Claude didn't realise. These minor bugs could only be fixed by Claude itself under detailed human guidance. Moreover, this porting project can be relatively easily completed with appropriate human guidance, since FreeTrace C++ has the ground truth (FreeTrace Python). However, the new projects will require more strict step-by-step supervision by Humans to avoid a large number of minor/major bugs that Claude cannot catch.

---

**Table of Contents**

- [Usage](#usage) — CLI commands, options, outputs
- [FreeTrace GUI](#freetrace-gui) — graphical interface
- [About](#about) — algorithm overview and pipeline diagram
- [Build](#build) — build instructions for Linux, macOS, and Windows
- [Windows Installer](#windows-installer) — automated installer build for Windows
- [Project Structure](#project-structure) — source files and models
- [Verification](#verification) — correctness testing results
- [Performance](#performance) — benchmarks vs Python
- [Paper](#paper) — bioRxiv preprint link
- [Links](#links) — related projects and license

---

## Usage

```bash
# Full pipeline: localization + tracking
./freetrace video.tiff results/

# Localization only
./freetrace localize video.tiff results/

# Tracking only (from existing loc CSV)
./freetrace track results/video_loc.csv results/ 100 --tiff video.tiff
```

### Options

**Localization:** `--window N` (default: 7), `--threshold F` (default: 1.0), `--shift N` (default: 1), `--cpu` (force CPU mode)

**Tracking:** `--depth N` (default: 3), `--cutoff N` (default: 3), `--jump F` (default: auto), `--tiff PATH`, `--no-fbm` (disable NN, fixed alpha/K), `--postprocess`, `--quiet`

### Tracking Modes

| | **fBm mode ON** (default) | **fBm mode OFF** (`--no-fbm`) |
|---|---|---|
| **Alpha / K** | Predicted by ConvLSTM / Dense NN | Fixed: alpha=1.0, K=0.3 |
| **H-K output** | `_diffusion.csv` + `_diffusion_distribution.png` | Not produced |
| **With GPU** | Localization + NN on GPU | Localization on GPU |
| **Without GPU** | Localization + NN on CPU | Localization on CPU |

### Outputs

- `{name}_loc.csv` — localization CSV (columns: `frame, x, y, z, xvar, yvar, rho, norm_cst, intensity, window_size`)
- `{name}_loc_2d_density.png` — particle density image
- `{name}_traces.csv` — trajectory CSV
- `{name}_traces.png` — trajectory visualization
- `{name}_diffusion.csv` — H and K per trajectory (fBm mode only)
- `{name}_diffusion_distribution.png` — H-K distribution plot (fBm mode only)

### As a library

```cpp
#include "localization.h"
#include "tracking.h"

// Localization
freetrace::run("input.tiff", "output_dir/", /*window_size=*/7, /*threshold=*/1.0f, /*shift=*/1, /*verbose=*/true);

// Tracking (defaults: depth=3, cutoff=3, jump=auto, fBm=ON)
freetrace::TrackingConfig config;
config.tiff_path = "input.tiff";
freetrace::run_tracking("output_dir/input_loc.csv", "output_dir/", 100, config);

// To disable fBm: config.fbm_mode = false; config.use_nn = false; config.hk_output = false;
```

## FreeTrace GUI

FreeTrace includes a graphical interface built with PyQt6. The GUI wraps the command-line binary, providing file/batch selection, parameter controls, real-time progress tracking, and result visualization.

**Running the GUI:**
```bash
pip install PyQt6
python gui.py
```

**Building a standalone executable (Windows):**
```bash
pip install pyinstaller PyQt6
python -m PyInstaller gui.spec --noconfirm --clean
# Output: dist/FreeTrace.exe
```

The GUI automatically finds the `freetrace` binary in the same directory, `build/`, or system PATH.

## About

FreeTrace localizes and tracks fluorescent particles in microscopy video data (TIFF stacks).

**Localization:** Background estimation → sliding-window likelihood detection (Gaussian PSF) → NMS → sub-pixel Gaussian fitting (Guo's iterative weighted least-squares with Householder QR solver).

**Tracking:** Greedy segmentation → jump threshold estimation → directed graph construction → multi-hypothesis optimization (2^depth alternatives per subgraph) → fBm Cauchy cost with qt_99 abnormal detection → sliding-window forecast → trajectory visualization (libpng).

### Pipeline Diagram

```
                         ┌─────────────────────────────────┐
                         │         LOCALIZATION            │
                         │                                 │
  GPU available          │  background ──► crop ──► likelihood ──► NMS ──► Guo
  & no --cpu?  ──YES──►  │     [CUDA]       [CUDA]    [CUDA]      [CPU]   [CPU]
               │         │                                 │
               NO──────► │     [CPU]        [CPU]     [CPU]       [CPU]   [CPU]
                         │                                 │
                         └────────────┬────────────────────┘
                                      │ *_loc.csv
                         ┌────────────▼────────────────────┐
                         │    TRACKING (CPU + GPU NN)      │
                         │                                 │
  fBm mode ON? ──YES──►  │  segmentation ──► forecast      │
  (default)    │         │       (NN α,K prediction)      │──► traces + diffusion
               │         │                                 │
               NO──────► │  segmentation ──► forecast      │
  (--no-fbm)             │       (fixed α=1, K=0.5)       │──► traces only
                         └─────────────────────────────────┘
```

GPU is auto-detected at startup. Tracking NN inference is GPU-accelerated via ONNX Runtime CUDA when available (e.g., sample0: ~380s CPU → 17s GPU).

## Build

FreeTrace C++ supports **Linux**, **macOS**, and **Windows**.

### Requirements

| Dependency | Required? | Purpose |
|---|---|---|
| C++17 compiler | **Yes** | GCC 7+, Clang 5+, or MSVC 2019+ |
| libtiff | **Yes** (or OpenCV) | Reading TIFF microscopy stacks |
| libpng | Recommended | Trajectory visualization images |
| ONNX Runtime | Optional | NN inference for fBm mode |
| NVIDIA GPU + CUDA 12.x | Optional | GPU-accelerated localization and NN inference |

---

### Linux (Ubuntu / Debian)

```bash
sudo apt update && sudo apt install -y build-essential cmake libtiff-dev libpng-dev
cd /path/to/FreeTrace_cpp
```

**Without fBm** (basic tracking):
```bash
mkdir build && cd build && cmake .. && make -j$(nproc)
```

**With fBm + GPU** (recommended):
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-gpu-1.24.3.tgz
tar xzf onnxruntime-linux-x64-gpu-1.24.3.tgz
mkdir -p build_gpu && cd build_gpu
cmake .. -DUSE_CUDA=ON -DUSE_ONNXRUNTIME=ON -DONNXRUNTIME_DIR=$(pwd)/../onnxruntime-linux-x64-gpu-1.24.3
make -j$(nproc)

export LD_LIBRARY_PATH=$(pwd)/../onnxruntime-linux-x64-gpu-1.24.3/lib:$LD_LIBRARY_PATH
./freetrace video.tiff results/
```

> Use `nvcc --version` (not `nvidia-smi`) to verify your CUDA toolkit version. If CMake cannot find `nvcc`, add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

**With fBm, CPU only** (no GPU needed):
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-1.24.3.tgz
tar xzf onnxruntime-linux-x64-1.24.3.tgz
mkdir build && cd build
cmake .. -DUSE_ONNXRUNTIME=ON -DONNXRUNTIME_DIR=$(pwd)/../onnxruntime-linux-x64-1.24.3
make -j$(nproc)

export LD_LIBRARY_PATH=$(pwd)/../onnxruntime-linux-x64-1.24.3/lib:$LD_LIBRARY_PATH
./freetrace video.tiff results/
```

---

### macOS

```bash
brew install cmake libtiff libpng
cd /path/to/FreeTrace_cpp
```

> Apple Silicon (M1/M2/M3/M4) does not support AVX2. The code falls back to `std::sort` automatically.

**Without fBm:**
```bash
mkdir build && cd build && cmake .. && make -j$(sysctl -n hw.ncpu)
```

**With fBm** (CPU only — macOS has no CUDA):
```bash
# Apple Silicon (Intel Mac: use onnxruntime-osx-x86_64-1.24.3.tgz instead)
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-osx-arm64-1.24.3.tgz
tar xzf onnxruntime-osx-arm64-1.24.3.tgz
mkdir build && cd build
cmake .. -DUSE_ONNXRUNTIME=ON -DONNXRUNTIME_DIR=$(pwd)/../onnxruntime-osx-arm64-1.24.3
make -j$(sysctl -n hw.ncpu)

export DYLD_LIBRARY_PATH=$(pwd)/../onnxruntime-osx-arm64-1.24.3/lib:$DYLD_LIBRARY_PATH
./freetrace video.tiff results/
```

---

### Windows

Install in this order (GPU support):
1. [**Visual Studio 2022**](https://visualstudio.microsoft.com/vs/older-downloads/) — select "Desktop development with C++"
2. [**CMake**](https://cmake.org/download/)

Run from **Developer Command Prompt**:

```powershell
# Install vcpkg (one-time)
cd C:\; git clone https://github.com/microsoft/vcpkg.git; cd vcpkg; .\bootstrap-vcpkg.bat; .\vcpkg install tiff:x64-windows libpng:x64-windows

# Go to FreeTrace_cpp
cd C:\path\to\FreeTrace_cpp
```

**Without fBm:**
```powershell
mkdir build; cd build; cmake .. -G "Visual Studio 17 2022" "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"; cmake --build . --config Release
```

**With fBm (GPU):** requires [**CUDA Toolkit 12.x**](https://developer.nvidia.com/cuda-downloads) (install after VS 2022) and [**cuDNN 9.x**](https://developer.nvidia.com/cudnn-downloads) (CUDA 12, Windows, Tarball). Use cuDNN 9.2 for RTX 2000/3000/4000, or cuDNN 9.20+ for RTX 5000 (Blackwell).
```powershell
Invoke-WebRequest -Uri https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-win-x64-gpu-1.24.3.zip -OutFile ort.zip; tar -xf ort.zip; mkdir build; cd build; cmake .. -G "Visual Studio 17 2022" "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake" "-DUSE_ONNXRUNTIME=ON" "-DONNXRUNTIME_DIR=..\onnxruntime-win-x64-gpu-1.24.3"; cmake --build . --config Release; copy ..\onnxruntime-win-x64-gpu-1.24.3\lib\*.dll Release\
```
> **Important:** After extracting cuDNN, copy **all DLLs** from cuDNN's `bin/` folder into the `Release/` folder. Without this, you will get `cudnn64_9.dll not found` errors at runtime.

> For CPU-only fBm, use `onnxruntime-win-x64-1.24.3.zip` instead.
>
> **Note:** CUDA 12.x requires Visual Studio 2022. Newer VS versions (2025/2026) are not yet supported by NVIDIA.

## Windows Installer

For end-user distribution on Windows, an automated installer build script is provided. It downloads all dependencies (cuDNN, ONNX Runtime, vcpkg libraries, Inno Setup), builds `freetrace.exe` and `FreeTrace.exe` (GUI), and creates a self-contained installer.

**Prerequisites:** Visual Studio 2022 (C++), CMake, Python, Git, CUDA 12.x

```powershell
cd installer
.\build_installer.bat
```

The script performs 4 steps:
1. **Build** `freetrace.exe` (C++ CLI binary via CMake/MSBuild)
2. **Build** `FreeTrace.exe` (PyQt6 GUI via PyInstaller)
3. **Stage** all files (executables, models, DLLs) into `installer_staging/`
4. **Create** the installer via Inno Setup → `installer/FreeTrace_<version>_win64_setup.exe`

The resulting installer is fully self-contained — users only need an NVIDIA GPU driver.

## Project Structure

```
FreeTrace_cpp/
├── CMakeLists.txt           # Build configuration
├── include/
│   ├── image_pad.h          # Image2D, statistics, likelihood, cropping
│   ├── regression.h         # Gaussian fitting (Guo algorithm)
│   ├── cost_function.h      # fBm Cauchy log-PDF cost function
│   ├── localization.h       # Localization pipeline
│   ├── tracking.h           # Tracking pipeline
│   ├── nn_inference.h       # NN models and inference
│   └── gpu_module.h         # GPU acceleration interface
├── src/
│   ├── main.cpp             # CLI entry point
│   ├── image_pad.cpp        # Image operations
│   ├── regression.cpp       # Guo algorithm with 6x6 QR solver
│   ├── cost_function.cpp    # Cost function for trajectory linking
│   ├── localization.cpp     # Localization pipeline
│   ├── tracking.cpp         # Tracking pipeline (~2900 lines)
│   ├── nn_inference.cpp     # ONNX Runtime NN inference
│   ├── nn_inference_stub.cpp# CPU-only fallback (no ONNX Runtime)
│   ├── gpu_module.cu        # CUDA kernels
│   └── gpu_module_stub.cpp  # CPU fallback (no CUDA)
├── models/
│   ├── qt_99.bin            # Abnormal detection thresholds
│   ├── reg_model_*.onnx     # Alpha ConvLSTM models (window 3/5/8)
│   ├── reg_k_model.onnx     # K Dense model (ONNX fallback)
│   ├── k_model_weights.bin  # K Dense model weights (fast path)
│   └── traj_colors.bin      # Trajectory visualization colors
├── gui.py                   # FreeTrace GUI (PyQt6)
├── gui.spec                 # PyInstaller spec for FreeTrace.exe
└── installer/
    ├── build_installer.bat  # Automated Windows installer build
    └── freetrace_installer.iss  # Inno Setup script
```

## Verification

Verified on 6 test datasets (100–501 frames, 120×110 to 512×512).

**Localization** (6 samples × 6 configs = 36 tests, fixed batch_size=100):
All matched detections show max position error < 0.00001 px. Count mismatches (1–3 detections per test) are caused by CuPy GPU vs CPU float arithmetic at NMS threshold boundaries — accepted as equivalent.

**Tracking (fBm OFF)** — 6 samples × 2 jump configs (7, 10) = 12 tests:
**12/12 perfect** — 100% point match, 100% trajectory match, 0.000000 px max diff.

**Tracking (fBm ON, fixed jump)** — 6 samples × 2 jump configs (7, 10) = 12 tests:
- jump=7: **6/6 perfect** — 100% point match, 100% trajectory match.
- jump=10: **6/6 pass** — 99.8–100% point match. Minor trajectory splits/merges in dense regions due to TF vs ONNX float divergence (~1e-5 in alpha predictions).

**Tracking (fBm ON, auto jump)** — 6 samples:
**6/6 perfect** — 100% point match, 100% trajectory match, 0.000000 px max diff.

## Performance

Tracking benchmarks on 6 test datasets (Linux x86_64, NVIDIA GPU, depth=3, cutoff=3):

**fBm OFF:**

| Dataset | Frames | Python (s) | C++ (s) | Speedup |
|---------|--------|-----------|---------|---------|
| testsample0 (120×110, 500fr) | jump=7 | 4.1 | 1.0 | **4.1x** |
| testsample4 (120×110, 501fr) | jump=10 | 12.8 | 4.2 | **3.0x** |
| testsample5 (512×512, 100fr) | jump=7 | 23.3 | 7.0 | **3.3x** |
| testsample5 (512×512, 100fr) | jump=10 | 28.0 | 8.9 | **3.1x** |

**fBm ON:**

| Dataset | Frames | Python (s) | C++ (s) | Speedup |
|---------|--------|-----------|---------|---------|
| testsample0 (120×110, 500fr) | jump=7 | 23.9 | 10.0 | **2.4x** |
| testsample4 (120×110, 501fr) | jump=10 | 82.5 | 21.9 | **3.8x** |
| testsample5 (512×512, 100fr) | jump=7 | 77.0 | 14.4 | **5.3x** |
| testsample5 (512×512, 100fr) | jump=10 | 144.5 | 21.1 | **6.8x** |
| testsample5 (512×512, 100fr) | jump=auto | 98.8 | 18.4 | **5.4x** |

Speedup increases with larger datasets and jump thresholds. C++ batches NN calls efficiently via ONNX Runtime.

## Paper

- **FreeTrace** on bioRxiv: https://doi.org/10.64898/2026.01.08.698486

## Links

- **FreeTrace (Python)**: https://github.com/JunwooParkSaribu/FreeTrace
- **Author**: Junwoo PARK — Sorbonne Université
- **License**: GPLv3+

---

*This C++ port is developed by Claude (claude-opus-4-6, Anthropic AI) from the original Python/Cython codebase by Junwoo PARK.*
