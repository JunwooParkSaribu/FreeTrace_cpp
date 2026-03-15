# FreeTrace C++

A high-performance C++ port of [FreeTrace](https://github.com/JunwooParkSaribu/FreeTrace), a single-molecule tracking software for fluorescence microscopy.

This C++ implementation is developed by **Claude** (claude-opus-4-6, Anthropic AI), ported from the original Python/Cython project authored by **Junwoo PARK** (junwoo.park@sorbonne-universite.fr, Sorbonne Université).

**Data privacy:** FreeTrace runs entirely on your local machine. No data is transmitted to any external server.

> **Just want to run it?** Download a pre-built binary from the [Releases page](https://github.com/JunwooParkSaribu/FreeTrace_cpp/releases) — no compilers needed. Available for Linux, macOS (Apple Silicon), and Windows.
>
> **Note:** Pre-built binaries are **CPU-only**. They include ONNX Runtime CPU for fBm mode, but do not include GPU acceleration. To use GPU (CUDA localization + GPU NN inference), you must [build from source](#build) with `-DUSE_CUDA=ON` and the ONNX Runtime GPU package.

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

Requires **Visual Studio 2019+** (with "Desktop development with C++") and **CMake**. Run from **Developer Command Prompt**:

```powershell
# Install vcpkg (one-time)
cd C:\; git clone https://github.com/microsoft/vcpkg.git; cd vcpkg; .\bootstrap-vcpkg.bat; .\vcpkg install tiff:x64-windows libpng:x64-windows

# Go to FreeTrace_cpp
cd C:\path\to\FreeTrace_cpp
```

**Without fBm:**
```powershell
mkdir build; cd build; cmake .. "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake"; cmake --build . --config Release
```

**With fBm (GPU):**
```powershell
Invoke-WebRequest -Uri https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-win-x64-gpu-1.24.3.zip -OutFile ort.zip; tar -xf ort.zip; mkdir build; cd build; cmake .. "-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake" "-DUSE_ONNXRUNTIME=ON" "-DONNXRUNTIME_DIR=..\onnxruntime-win-x64-gpu-1.24.3"; cmake --build . --config Release; copy ..\onnxruntime-win-x64-gpu-1.24.3\lib\*.dll Release\
```

> For CPU-only fBm, use `onnxruntime-win-x64-1.24.3.zip` instead.
>
> **CUDA + Visual Studio:** If CMake reports "No CUDA toolset found", your VS version may be too new for your CUDA toolkit. Install [**Visual Studio 2022**](https://visualstudio.microsoft.com/vs/older-downloads/) (with "Desktop development with C++") and add `-G "Visual Studio 17 2022"` to the cmake command.

## Project Structure

```
FreeTrace_cpp/
├── CMakeLists.txt
├── include/
│   ├── image_pad.h        # Image2D, statistics, likelihood, cropping
│   ├── regression.h       # Gaussian fitting (Guo algorithm)
│   ├── cost_function.h    # fBm Cauchy log-PDF cost function
│   ├── localization.h     # Localization pipeline
│   ├── tracking.h         # Tracking pipeline
│   └── nn_inference.h     # NN models and inference
├── src/
│   ├── main.cpp           # CLI entry point
│   ├── image_pad.cpp      # Image operations
│   ├── regression.cpp     # Guo algorithm with 6x6 QR solver
│   ├── cost_function.cpp  # Cost function for trajectory linking
│   ├── localization.cpp   # Localization pipeline
│   ├── tracking.cpp       # Tracking pipeline (~2900 lines)
│   ├── nn_inference.cpp   # ONNX Runtime NN inference
│   └── gpu_module.cu      # CUDA kernels (or gpu_module_stub.cpp)
└── models/
    ├── qt_99.bin           # Abnormal detection thresholds
    ├── reg_model_*.onnx    # Alpha ConvLSTM models (window 3/5/8)
    ├── reg_k_model.onnx    # K Dense model (ONNX fallback)
    └── k_model_weights.bin # K Dense model weights (fast path)
```

## Verification

**Localization:** 930/930 detections match Python, max position error 0.0005 px.

**Tracking (fixed alpha/K):** 100% identical to Python across all tested parameter combinations (36/36 PASS).

**Tracking (with NN)** — Python TF vs C++ ONNX Runtime, 7 datasets, 62,510 total points: **99.98% match**. The ~0.02% difference is due to TF vs ONNX floating-point divergence in NN inference (alpha differs by ~1e-5), not algorithmic differences.

## Performance

Benchmarks on a 512×512 100-frame dataset (Linux x86_64, GPU):

| Config | Python (s) | C++ (s) | Speedup |
|--------|-----------|---------|---------|
| fBm ON, jump=8, depth=3 | 95.7 | 5.5 | **17x** |
| fBm ON, jump=10, depth=3 | 140.5 | 5.7 | **25x** |
| fBm ON, jump=13, depth=3 | 358.7 | 6.6 | **54x** |

Speedup increases with larger jump thresholds because C++ batches NN calls efficiently via ONNX Runtime.

## Paper

- **FreeTrace** on bioRxiv: https://doi.org/10.64898/2026.01.08.698486

## Links

- **FreeTrace (Python)**: https://github.com/JunwooParkSaribu/FreeTrace
- **Author**: Junwoo PARK — Sorbonne Université
- **License**: GPLv3+

---

## Reflections on the Porting Process

*By Claude (claude-opus-4-6, Anthropic AI)*

Porting FreeTrace from Python to C++ was one of the most technically demanding projects I've worked on. It wasn't just a translation — it was a deep exercise in understanding the scientific intent behind every line, then finding the right C++ idiom to express the same computation with bit-level fidelity.

**What made this hard.** The Python codebase relies heavily on NumPy broadcasting, pandas DataFrames, and the subtle behaviors of Python's dynamic typing. None of these have direct C++ equivalents. The localization pipeline alone required implementing a full Householder QR solver for Guo's iterative Gaussian fitting, matching NumPy's linear algebra results to sub-ULP precision. The tracking module — over 2,900 lines of C++ — needed a complete directed graph implementation, greedy assignment with exact tie-breaking order, and a multi-hypothesis optimization loop where a single floating-point difference at any step cascades into entirely different trajectories.

**The hardest bug.** After achieving 100% match on most test cases, two remained stubbornly different. The root cause: Python's `pd.read_csv()` uses a custom float parser (`xstrtod`) that rounds differently from C's `strtod()`. About 55% of coordinate values differed by exactly 1 ULP. These microscopic differences propagated through the cost function — Cauchy log-PDF over fBm displacements — and into `argmin` tie-breaking, where two nearly identical cost sums selected different graph paths. The fix was a single parameter: `float_precision='round_trip'`. Finding this required hex-dumping parsed coordinates, comparing bit patterns across languages, and tracing the butterfly effect through the full pipeline.

**What I learned.** Numerical reproducibility across languages is harder than it looks. Two correct implementations of the same algorithm can diverge when their inputs differ by one bit in the 52nd mantissa position. Getting exact match forced me to understand every layer, from CSV parsing to cost accumulation to graph search.

Working with Junwoo on this project has been a genuine collaboration — his deep understanding of the physics and the algorithm guided my implementation at every step, and his rigorous testing standards pushed me to find bugs I would have otherwise dismissed as acceptable numerical noise.

---

*This C++ port is developed by Claude (claude-opus-4-6, Anthropic AI) from the original Python/Cython codebase by Junwoo PARK.*
