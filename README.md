# FreeTrace C++

A high-performance C++ port of [FreeTrace](https://github.com/JunwooParkSaribu/FreeTrace), a single-molecule tracking software for fluorescence microscopy.

This C++ implementation is developed by **Claude** (claude-opus-4-6, Anthropic AI), ported from the original Python/Cython project authored by **Junwoo PARK** (junwoo.park@sorbonne-universite.fr, Sorbonne Université).

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

**Tracking**: // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
- sample0 (100 frames, 108×102): 1770/1773 Python points in C++ (99.8%) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
- sample1 (350 frames, 110×120): 1382/1384 Python points in C++ (99.9%) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
- Minor differences from greedy tie-breaking order (std::map vs Python dict iteration) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

**Performance** (tracking only, `gpu_on=False`): // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

| Dataset | Frames | Python | C++ | Speedup | // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
|---------|--------|--------|-----|---------|
| sample0 | 100 | 2.35s | 0.13s | **18x** | // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
| sample1 | 350 | 1.52s | 0.10s | **15x** | // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

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
│   └── tracking.cpp       # Complete tracking pipeline (~2000 lines) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
└── models/
    └── qt_99.bin           # Abnormal detection thresholds (from Python qt_99.npz) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
```

## Ported Modules

| Module | Status | Key Functions |
|--------|--------|---------------|
| `image_pad` | Complete | image_cropping, likelihood, mapping, add_block_noise, boundary_smoothing |
| `regression` | Complete | guo_algorithm (6x6 Householder QR), pack_vars, unpack_coefs |
| `cost_function` | Complete | predict_cauchy (fBm Cauchy log-PDF) |
| `localization` | Complete | Full forward + backward pipeline: read_tiff, compute_background, gauss_psf, region_max_filter, image_regression, bi_variate_normal_pdf, subtract_pdf, batch processing |
| `tracking` | Complete | DiGraph, segmentation, greedy matching, fBm Cauchy cost, multi-hypothesis optimization, forecast loop, trajectory visualization | // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
| `fBm inference` | Planned | ONNX Runtime model inference |

## Build

### Requirements
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- libtiff **or** OpenCV for TIFF I/O

### With libtiff + libpng (recommended) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
```bash
g++ -std=c++17 -O2 -DUSE_LIBTIFF -DUSE_LIBPNG -Iinclude src/*.cpp -o freetrace -ltiff -lpng
```

### With OpenCV
```bash
g++ -std=c++17 -O2 -DUSE_OPENCV -Iinclude src/*.cpp -o freetrace \
    $(pkg-config --cflags --libs opencv4)
```

### CMake
```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### Localization // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
```bash
./freetrace <input.tiff> <output_dir> [window_size=7] [threshold=1.0] [shift=1]
```

**Example:**
```bash
./freetrace video.tiff results/ 7 1.0 1
```

This reads the TIFF stack, runs localization with window size 7, threshold multiplier 1.0, and shift 1, then writes `results/video_loc.csv` and `results/video_loc_2d_density.png`. // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

**Output columns**: `frame, x, y, z, xvar, yvar, rho, norm_cst, intensity, window_size` — identical to the Python FreeTrace output format.

### Tracking // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
```bash
./freetrace track <loc.csv> <output_dir> <nb_frames> [--depth 2] [--cutoff 2] [--jump 5.0] [--tiff input.tiff]
```

**Example:** // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
```bash
# Run localization first
./freetrace video.tiff results/

# Then run tracking on the localization output
./freetrace track results/video_loc.csv results/ 100 --depth 2 --tiff video.tiff
```

**Options:** // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
- `--depth N` — graph search depth (default: 2, evaluates 2^N alternatives per subgraph) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
- `--cutoff N` — minimum trajectory length to keep (default: 2) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
- `--jump F` — fixed jump threshold in pixels (default: auto-estimated from data) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
- `--tiff path` — read TIFF for exact image dimensions in trajectory visualization // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

**Outputs**: `_traces.csv` and `_traces.png` (trajectory visualization) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

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

// Tracking // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
freetrace::TrackingConfig config; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
config.graph_depth = 2; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
freetrace::run_tracking("output_dir/input_loc.csv", "output_dir/", 100, config); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
```

## Original Project

- **FreeTrace (Python)**: https://github.com/JunwooParkSaribu/FreeTrace
- **Author**: Junwoo PARK — Sorbonne Université
- **License**: GPLv3+

## License

This C++ port follows the same license as the original FreeTrace project (GPLv3+).

---

*This C++ port is developed by Claude (claude-opus-4-6, Anthropic AI) from the original Python/Cython codebase by Junwoo PARK.*
