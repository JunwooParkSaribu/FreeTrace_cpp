# FreeTrace C++

A high-performance C++ port of [FreeTrace](https://github.com/JunwooParkSaribu/FreeTrace), a single-molecule tracking software for fluorescence microscopy.

This C++ implementation is developed by **Claude** (claude-opus-4-6, Anthropic AI), ported from the original Python/Cython project authored by **Junwoo PARK** (junwoo.park@sorbonne-universite.fr, Sorbonne Université).

## Quick Reference — Tracking Modes

| | **fBm mode ON** (default) | **fBm mode OFF** (`--no-fbm`) |
|---|---|---|
| **NN models** | Loaded (alpha & K inferred per trajectory) | Not loaded |
| **Alpha / K** | Predicted by ConvLSTM / Dense NN | Fixed: alpha=1.0, K=0.3 |
| **H-K output** | `_diffusion.csv` + `_diffusion_distribution.png` | Not produced |
| **With GPU** | NN runs on GPU via ONNX Runtime CUDA (fast) | No difference |
| **Without GPU** | NN runs on CPU via ONNX Runtime (slower) | No difference |

**Defaults:**
- fBm mode: **ON** (NN inference + H-K output)
- Graph depth: **3** (evaluates 2^3 = 8 alternatives per subgraph)
- Cutoff: **3** (minimum trajectory length)
- Jump threshold: **auto** (inferred from data; use `--jump` to override)

**GPU behavior:** FreeTrace automatically detects GPU availability via ONNX Runtime CUDA provider. At startup, it always prints which mode is active:
- `NN inference: GPU (CUDA) — fast` when GPU is available
- `NN inference: CPU — this may be slower than GPU` when falling back to CPU

GPU requires building with the ONNX Runtime **GPU** package and setting `LD_LIBRARY_PATH` at runtime (see [Build Instructions](#build)).

**Quick start:**
```bash
# Full pipeline: localization + tracking (simplest usage)
./freetrace video.tiff results/

# Localization only
./freetrace localize video.tiff results/

# Tracking only (from existing loc CSV)
./freetrace track results/video_loc.csv results/ 100 --tiff video.tiff
```

> **First time?** See [Build Instructions](#build) for step-by-step setup on Linux, macOS, and Windows.

## About

FreeTrace localizes and tracks fluorescent particles in microscopy video data (TIFF stacks). // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

### Localization // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
1. **Background estimation** — iterative mode-based per-frame background with threshold computation
2. **Sliding-window likelihood detection** — log-likelihood ratio maps using Gaussian PSF templates
3. **Non-maximum suppression (NMS)** — score-sorted detection with spatial masking
4. **Sub-pixel Gaussian fitting** — Guo's iterative weighted least-squares algorithm with Householder QR solver

The backward pass (multi-scale deflation for overlapping particles) is structurally implemented but disabled by default (`deflation=0`), matching the Python FreeTrace default. Deflation is disabled because it fails critically on low SNR images.

### Tracking // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
1. **Segmentation** — greedy nearest-neighbor matching to extract jump distributions // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
2. **Jump threshold estimation** — simplified GMM (std-based) or user-specified fixed threshold // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
3. **Graph-based linking** — directed graph construction with distance-based edge building // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
4. **Multi-hypothesis optimization** — greedy path selection evaluating 2^depth alternatives per subgraph // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
5. **fBm Cauchy cost** — fractional Brownian motion cost function with qt_99 abnormal detection // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
6. **Sliding-window forecast** — main tracking loop advancing through time steps // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
7. **Trajectory visualization** — colored trajectory overlay on black background (libpng) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

### Verification // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

**Localization**: 930/930 detections match, max position error 0.0005 px, max rho error 1.6e-6. // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

**Tracking (without NN)**: // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
- sample0 (100 frames, 108×102): 1770/1773 Python points in C++ (99.8%)
- sample1 (350 frames, 110×120): 1382/1384 Python points in C++ (99.9%)
- Minor differences from greedy tie-breaking order (std::map vs Python dict iteration)

**Tracking (with NN, GPU)** — Python TF+GPU vs C++ ONNX+GPU: // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

| Dataset | Frames | Py Pts | C++ Pts | Match% | MaxDiff (px) |
|---------|--------|--------|---------|--------|-------------|
| sample0 | 100 | 1816 | 1814 | 99.89% | 0.000488 |
| sample1 | 350 | 1431 | 1432 | 99.93% | 0.000610 |
| sample2 | 2001 | 19291 | 19295 | 99.97% | 0.000697 |
| sample3 | 2001 | 15262 | 15262 | 99.98% | 0.000696 |
| sample4 | 5000 | 19874 | 19873 | 99.98% | 0.000690 |
| sample5 | 1001 | 4244 | 4244 | 100% | 0.000589 |
| sample6 | 40 | 592 | 592 | 100% | 0.000616 |
| **TOTAL** | | **62510** | **62512** | **99.98%** | |

Tiny differences (~0.01%) are due to TF vs ONNX Runtime floating-point divergence. When both sides use ONNX Runtime, sample0 and sample1 achieve 100% exact match.

**NN prediction standalone**: 20/20 BM/fBM trajectories match within 1e-6 (worst 3.58e-07) when both use ONNX Runtime.

## Project Structure

```
FreeTrace_cpp/
├── CMakeLists.txt
├── include/
│   ├── image_pad.h        # Image2D struct, statistics, likelihood, cropping, noise
│   ├── regression.h       # Gaussian fitting (Guo algorithm), coefficient packing
│   ├── cost_function.h    # fBm Cauchy log-PDF cost function
│   ├── localization.h     # Full localization pipeline structs and declarations
│   └── tracking.h         # Graph-based tracking pipeline declarations // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
├── src/
│   ├── main.cpp           # CLI entry point (localization + tracking modes)
│   ├── image_pad.cpp      # Image operations (cropping, likelihood, mapping, noise)
│   ├── regression.cpp     # Guo algorithm with inline 6x6 QR solver
│   ├── cost_function.cpp  # Cost function for trajectory linking
│   ├── localization.cpp   # Complete localization pipeline
│   ├── tracking.cpp       # Complete tracking pipeline (~2900 lines)
│   ├── nn_inference.cpp   # ONNX Runtime NN inference (alpha + K prediction) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
│   └── gpu_module_stub.cpp # GPU module stub
├── include/
│   └── nn_inference.h     # NN models struct and function declarations
└── models/
    ├── qt_99.bin           # Abnormal detection thresholds (from Python qt_99.npz)
    ├── reg_model_3.onnx    # Alpha ConvLSTM model (window=3)
    ├── reg_model_5.onnx    # Alpha ConvLSTM model (window=5)
    ├── reg_model_8.onnx    # Alpha ConvLSTM model (window=8)
    ├── reg_k_model.onnx    # K Dense model (ONNX, fallback)
    └── k_model_weights.bin # K Dense model weights (direct computation, fast path)
```

## Ported Modules

| Module | Status | Key Functions |
|--------|--------|---------------|
| `image_pad` | Complete | image_cropping, likelihood, mapping, add_block_noise, boundary_smoothing |
| `regression` | Complete | guo_algorithm (6x6 Householder QR), pack_vars, unpack_coefs |
| `cost_function` | Complete | predict_cauchy (fBm Cauchy log-PDF) |
| `localization` | Complete | Full forward + backward pipeline: read_tiff, compute_background, gauss_psf, region_max_filter, image_regression, bi_variate_normal_pdf, subtract_pdf, batch processing |
| `tracking` | Complete | DiGraph, segmentation, greedy matching, fBm Cauchy cost, multi-hypothesis optimization, forecast loop, trajectory visualization | // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
| `nn_inference` | Complete | ONNX Runtime alpha/K prediction (GPU+CPU), ConvLSTM alpha models, Dense K model (direct fast path + ONNX fallback), batched inference |

## Build

FreeTrace C++ supports **Linux**, **macOS**, and **Windows**. You can build with either **CMake** or a **direct compiler command**.

### Requirements

| Dependency | Required? | Purpose |
|---|---|---|
| C++17 compiler | **Yes** | GCC 7+, Clang 5+, or MSVC 2019+ |
| libtiff | **Yes** (or OpenCV) | Reading TIFF microscopy stacks |
| libpng | Recommended | Trajectory visualization images |
| ONNX Runtime | Optional | NN inference for fBm mode (default ON; use `--no-fbm` to disable) |
| NVIDIA GPU + CUDA 12.x | Optional | GPU-accelerated NN inference |

---

### Linux (Ubuntu / Debian)

#### Step 1: Install dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake libtiff-dev libpng-dev
```

#### Step 2a: Build without fBm (basic tracking only)

**Using CMake:**
```bash
cd FreeTrace_cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Or direct compile:**
```bash
cd FreeTrace_cpp
g++ -std=c++17 -O2 -mavx2 -DUSE_LIBTIFF -DUSE_LIBPNG -Iinclude \
    src/main.cpp src/image_pad.cpp src/regression.cpp src/cost_function.cpp \
    src/localization.cpp src/tracking.cpp src/nn_inference_stub.cpp src/gpu_module_stub.cpp \
    -o freetrace -ltiff -lpng
```

#### Step 2b: Build with fBm support (ONNX Runtime GPU — recommended)

1. Download ONNX Runtime GPU. Choose the package matching your **CUDA toolkit version** (check with `nvcc --version`):
```bash
cd FreeTrace_cpp

# CUDA 12.x (most common):
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-gpu-1.24.3.tgz
tar xzf onnxruntime-linux-x64-gpu-1.24.3.tgz
ORT_DIR=onnxruntime-linux-x64-gpu-1.24.3

# CUDA 13.x:
# wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-gpu_cuda13-1.24.3.tgz
# tar xzf onnxruntime-linux-x64-gpu_cuda13-1.24.3.tgz
# ORT_DIR=onnxruntime-linux-x64-gpu-1.24.3
```

> **Important**: Your NVIDIA driver may report a higher CUDA version than what's actually installed. Use `nvcc --version` (not `nvidia-smi`) to check the toolkit version. If `nvcc` is not found, install the CUDA toolkit: `sudo apt install nvidia-cuda-toolkit`.

2. Build:

**Using CMake:**
```bash
mkdir build && cd build
cmake .. -DUSE_ONNXRUNTIME=ON -DONNXRUNTIME_DIR=$(pwd)/../$ORT_DIR
make -j$(nproc)
```

**Or direct compile:**
```bash
g++ -std=c++17 -O2 -mavx2 \
    -DUSE_LIBTIFF -DUSE_LIBPNG -DUSE_ONNXRUNTIME \
    -Iinclude -I$ORT_DIR/include \
    src/main.cpp src/image_pad.cpp src/regression.cpp src/cost_function.cpp \
    src/localization.cpp src/tracking.cpp src/nn_inference.cpp src/gpu_module_stub.cpp \
    -o freetrace -ltiff -lpng -L$ORT_DIR/lib -lonnxruntime \
    -Wl,-rpath,\$ORIGIN/$ORT_DIR/lib
```

3. Run (set library path for CUDA provider):
```bash
export LD_LIBRARY_PATH=$(pwd)/$ORT_DIR/lib:$LD_LIBRARY_PATH
./freetrace track loc.csv output/ 100 --tiff video.tiff
```

FreeTrace will print `NN inference: GPU (CUDA) — fast` if GPU is detected, or `NN inference: CPU — this may be slower than GPU` if it falls back to CPU.

#### Step 2c: Build with fBm support (ONNX Runtime CPU only — no GPU needed)

1. Download ONNX Runtime CPU:
```bash
cd FreeTrace_cpp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-1.24.3.tgz
tar xzf onnxruntime-linux-x64-1.24.3.tgz
```

2. Build:

**Using CMake:**
```bash
mkdir build && cd build
cmake .. -DUSE_ONNXRUNTIME=ON -DONNXRUNTIME_DIR=$(pwd)/../onnxruntime-linux-x64-1.24.3
make -j$(nproc)
```

**Or direct compile:**
```bash
ORT=onnxruntime-linux-x64-1.24.3
g++ -std=c++17 -O2 -mavx2 \
    -DUSE_LIBTIFF -DUSE_LIBPNG -DUSE_ONNXRUNTIME \
    -Iinclude -I$ORT/include \
    src/main.cpp src/image_pad.cpp src/regression.cpp src/cost_function.cpp \
    src/localization.cpp src/tracking.cpp src/nn_inference.cpp src/gpu_module_stub.cpp \
    -o freetrace -ltiff -lpng -L$ORT/lib -lonnxruntime \
    -Wl,-rpath,\$ORIGIN/$ORT/lib
```

3. Run:
```bash
export LD_LIBRARY_PATH=$(pwd)/onnxruntime-linux-x64-1.24.3/lib:$LD_LIBRARY_PATH
./freetrace track loc.csv output/ 100 --tiff video.tiff
```

---

### macOS

#### Step 1: Install dependencies

Using [Homebrew](https://brew.sh/):
```bash
brew install cmake libtiff libpng
```

#### Step 2a: Build without fBm (basic tracking only)

**Using CMake:**
```bash
cd FreeTrace_cpp
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

**Or direct compile:**
```bash
cd FreeTrace_cpp
clang++ -std=c++17 -O2 -DUSE_LIBTIFF -DUSE_LIBPNG -Iinclude \
    -I$(brew --prefix libtiff)/include -I$(brew --prefix libpng)/include \
    src/main.cpp src/image_pad.cpp src/regression.cpp src/cost_function.cpp \
    src/localization.cpp src/tracking.cpp src/nn_inference_stub.cpp src/gpu_module_stub.cpp \
    -o freetrace \
    -L$(brew --prefix libtiff)/lib -ltiff \
    -L$(brew --prefix libpng)/lib -lpng
```

> **Note**: Apple Silicon Macs (M1/M2/M3/M4) do not support AVX2. Do **not** add `-mavx2` — the code automatically falls back to `std::sort`. In rare cases where multiple trajectories have identical costs, tie-breaking order may differ slightly from the x86 build, but tracking results are functionally equivalent.

#### Step 2b: Build with fBm support (ONNX Runtime CPU)

macOS does not support CUDA, so fBm mode runs on CPU via ONNX Runtime.

1. Download ONNX Runtime for macOS:
```bash
cd FreeTrace_cpp
# Apple Silicon (M1/M2/M3/M4):
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-osx-arm64-1.24.3.tgz
tar xzf onnxruntime-osx-arm64-1.24.3.tgz
ORT=onnxruntime-osx-arm64-1.24.3

# Intel Mac:
# curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-osx-x86_64-1.24.3.tgz
# tar xzf onnxruntime-osx-x86_64-1.24.3.tgz
# ORT=onnxruntime-osx-x86_64-1.24.3
```

2. Build:

**Using CMake:**
```bash
mkdir build && cd build
cmake .. -DUSE_ONNXRUNTIME=ON -DONNXRUNTIME_DIR=$(pwd)/../$ORT
make -j$(sysctl -n hw.ncpu)
```

**Or direct compile:**
```bash
clang++ -std=c++17 -O2 \
    -DUSE_LIBTIFF -DUSE_LIBPNG -DUSE_ONNXRUNTIME \
    -Iinclude -I$ORT/include \
    -I$(brew --prefix libtiff)/include -I$(brew --prefix libpng)/include \
    src/main.cpp src/image_pad.cpp src/regression.cpp src/cost_function.cpp \
    src/localization.cpp src/tracking.cpp src/nn_inference.cpp src/gpu_module_stub.cpp \
    -o freetrace \
    -L$(brew --prefix libtiff)/lib -ltiff \
    -L$(brew --prefix libpng)/lib -lpng \
    -L$ORT/lib -lonnxruntime \
    -Wl,-rpath,@executable_path/$ORT/lib
```

3. Run:
```bash
export DYLD_LIBRARY_PATH=$(pwd)/$ORT/lib:$DYLD_LIBRARY_PATH
./freetrace track loc.csv output/ 100 --tiff video.tiff
```

---

### Windows

#### Step 1: Install prerequisites

1. Install **Visual Studio 2019 or later** with the "Desktop development with C++" workload, or install **Build Tools for Visual Studio**.
2. Install **CMake** (https://cmake.org/download/) — select "Add to PATH" during installation.
3. Install **vcpkg** for dependency management:
```powershell
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

4. Install dependencies via vcpkg:
```powershell
.\vcpkg install tiff:x64-windows libpng:x64-windows
```

#### Step 2a: Build without fBm (basic tracking only)

Open **Developer Command Prompt for VS** (or **x64 Native Tools Command Prompt**):

```powershell
cd FreeTrace_cpp
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

The output binary is at `build\Release\freetrace.exe`.

#### Step 2b: Build with fBm support (ONNX Runtime GPU)

1. Download ONNX Runtime GPU for Windows:
```powershell
cd FreeTrace_cpp
Invoke-WebRequest -Uri https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-win-x64-gpu-1.24.3.zip -OutFile onnxruntime-win-x64-gpu-1.24.3.zip
tar -xf onnxruntime-win-x64-gpu-1.24.3.zip
```

2. Build:
```powershell
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
         -DUSE_ONNXRUNTIME=ON ^
         -DONNXRUNTIME_DIR=..\onnxruntime-win-x64-gpu-1.24.3
cmake --build . --config Release
```

3. Run (copy ONNX Runtime DLLs next to the executable, or add to PATH):
```powershell
copy ..\onnxruntime-win-x64-gpu-1.24.3\lib\*.dll Release\
cd Release
freetrace.exe track loc.csv output\ 100 --tiff video.tiff
```

> **Note**: Requires NVIDIA GPU with CUDA 12.x. FreeTrace will print `NN inference: GPU (CUDA) — fast` or `NN inference: CPU — this may be slower than GPU` at startup.

#### Step 2c: Build with fBm support (ONNX Runtime CPU only)

Same steps as above, but download the CPU package:
```powershell
Invoke-WebRequest -Uri https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-win-x64-1.24.3.zip -OutFile onnxruntime-win-x64-1.24.3.zip
tar -xf onnxruntime-win-x64-1.24.3.zip
```
Then use `-DONNXRUNTIME_DIR=..\onnxruntime-win-x64-1.24.3` in the cmake command.

---

### Build with OpenCV (alternative to libtiff, any platform)

If you prefer OpenCV for TIFF I/O instead of libtiff:
```bash
# Linux/macOS
g++ -std=c++17 -O2 -DUSE_OPENCV -Iinclude \
    src/main.cpp src/image_pad.cpp src/regression.cpp src/cost_function.cpp \
    src/localization.cpp src/tracking.cpp src/nn_inference_stub.cpp src/gpu_module_stub.cpp \
    -o freetrace $(pkg-config --cflags --libs opencv4)
```

---

### Verify the build

After building, verify FreeTrace runs correctly:
```bash
./freetrace --help
```

Expected output:
```
FreeTrace C++
Usage:
  Tracking:
    freetrace track <loc.csv> <output_dir> <nb_frames> [options]
  Options:
    --depth N        Graph depth (default: 3)
    --cutoff N       Cutoff (default: 3)
    --jump F         Maximum jump distance in px (default: auto)
    --tiff PATH      TIFF file for image dimensions and output naming
    --no-fbm         Disable fBm mode (no NN, fixed alpha/K, no H-K output)
    --postprocess    Enable post-processing
    --quiet          Suppress status messages

  Localization:
    freetrace <input.tiff> <output_dir> [window_size] [threshold] [shift]
```

## Usage

### Full pipeline (localization + tracking) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

The simplest way to use FreeTrace — just provide a TIFF and output directory:
```bash
./freetrace <input.tiff> <output_dir> [options]
```

**Example:**
```bash
# Run everything with defaults (fBm ON, auto jump threshold)
./freetrace video.tiff results/

# With custom options
./freetrace video.tiff results/ --window 9 --threshold 1.5 --depth 4 --jump 10.0
```

This runs localization first, then automatically feeds the result into tracking.

**Outputs:**
- `{name}_loc.csv` — localization CSV
- `{name}_loc_2d_density.png` — particle density image
- `{name}_traces.csv` — trajectory CSV
- `{name}_traces.png` — trajectory visualization
- `{name}_diffusion.csv` — H and K per trajectory (fBm mode only)
- `{name}_diffusion_distribution.png` — H-K distribution plot (fBm mode only)

### Localization only // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
```bash
./freetrace localize <input.tiff> <output_dir> [options]
```

**Example:**
```bash
./freetrace localize video.tiff results/ --window 7 --threshold 1.0 --shift 1
```

**Localization options:**
- `--window N` — window size (default: 7)
- `--threshold F` — detection threshold multiplier (default: 1.0)
- `--shift N` — shift (default: 1)

**Output columns**: `frame, x, y, z, xvar, yvar, rho, norm_cst, intensity, window_size` — identical to the Python FreeTrace output format.

### Tracking only // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
```bash
./freetrace track <loc.csv> <output_dir> <nb_frames> [options]
```

**Example:**
```bash
# Default tracking (fBm ON, auto jump threshold)
./freetrace track results/video_loc.csv results/ 100 --tiff video.tiff

# Custom jump threshold
./freetrace track results/video_loc.csv results/ 100 --tiff video.tiff --jump 10.0

# Without fBm (no NN, fixed alpha/K)
./freetrace track results/video_loc.csv results/ 100 --tiff video.tiff --no-fbm
```

**Tracking options:**
- `--depth N` — graph search depth (default: 3, evaluates 2^N alternatives per subgraph)
- `--cutoff N` — minimum trajectory length to keep (default: 3)
- `--jump F` — maximum jump distance in pixels (default: auto-inferred from data)
- `--tiff path` — TIFF file for image dimensions and output naming
- `--no-fbm` — disable fBm mode: no NN, uses fixed alpha=1.0/K=0.3, no H-K output
- `--postprocess` — enable post-processing of trajectories
- `--quiet` — suppress status messages

### As a library
```cpp
#include "localization.h"
#include "tracking.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Localization
freetrace::run("input.tiff", "output_dir/",
               /*window_size=*/7,
               /*threshold=*/1.0f,
               /*shift=*/1,
               /*verbose=*/true);

// Tracking (defaults: depth=3, cutoff=3, jump=auto, fBm=ON) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
freetrace::TrackingConfig config;
config.tiff_path = "input.tiff";       // for output naming + image dimensions
freetrace::run_tracking("output_dir/input_loc.csv", "output_dir/", 100, config);

// To disable fBm mode:
// config.fbm_mode = false; config.use_nn = false; config.hk_output = false;
```

## Paper

- **FreeTrace** on bioRxiv: https://doi.org/10.64898/2026.01.08.698486 <!-- Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13 -->

## Original Project

- **FreeTrace (Python)**: https://github.com/JunwooParkSaribu/FreeTrace
- **Author**: Junwoo PARK — Sorbonne Université
- **License**: GPLv3+

## License

This C++ port follows the same license as the original FreeTrace project (GPLv3+).

---

*This C++ port is developed by Claude (claude-opus-4-6, Anthropic AI) from the original Python/Cython codebase by Junwoo PARK.*
