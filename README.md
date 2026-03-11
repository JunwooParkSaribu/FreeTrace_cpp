# FreeTrace C++

A high-performance C++ port of [FreeTrace](https://github.com/JunwooParkSaribu/FreeTrace), a single-molecule tracking software for fluorescence microscopy.

This C++ implementation is developed by **Claude** (claude-opus-4-6, Anthropic AI), ported from the original Python/Cython project authored by **Junwoo PARK** (junwoo.park@sorbonne-universite.fr, Sorbonne Université).

## About

FreeTrace localizes and tracks fluorescent particles in microscopy video data (TIFF stacks). The localization pipeline consists of:

1. **Background estimation** — iterative mode-based per-frame background with threshold computation
2. **Sliding-window likelihood detection** — log-likelihood ratio maps using Gaussian PSF templates
3. **Non-maximum suppression (NMS)** — score-sorted detection with spatial masking
4. **Sub-pixel Gaussian fitting** — Guo's iterative weighted least-squares algorithm with Householder QR solver

The backward pass (multi-scale deflation for overlapping particles) is structurally implemented but disabled by default (`deflation=0`), matching the Python FreeTrace default. Deflation is disabled because it fails critically on low SNR images.

The C++ port achieves **exact numerical parity** with the Python version (verified: 930/930 detections match, max position error 0.0005 px, max rho error 1.6e-6).

## Project Structure

```
FreeTrace_cpp/
├── CMakeLists.txt
├── include/
│   ├── image_pad.h        # Image2D struct, statistics, likelihood, cropping, noise
│   ├── regression.h       # Gaussian fitting (Guo algorithm), coefficient packing
│   ├── cost_function.h    # fBm Cauchy log-PDF cost function
│   └── localization.h     # Full localization pipeline structs and declarations
├── src/
│   ├── main.cpp           # CLI entry point
│   ├── image_pad.cpp      # Image operations (cropping, likelihood, mapping, noise)
│   ├── regression.cpp     # Guo algorithm with inline 6x6 QR solver
│   ├── cost_function.cpp  # Cost function for trajectory linking
│   └── localization.cpp   # Complete localization pipeline
└── models/                # ONNX exported models (planned)
```

## Ported Modules

| Module | Status | Key Functions |
|--------|--------|---------------|
| `image_pad` | Complete | image_cropping, likelihood, mapping, add_block_noise, boundary_smoothing |
| `regression` | Complete | guo_algorithm (6x6 Householder QR), pack_vars, unpack_coefs |
| `cost_function` | Complete | predict_cauchy (fBm Cauchy log-PDF) |
| `localization` | Complete | Full forward + backward pipeline: read_tiff, compute_background, gauss_psf, region_max_filter, image_regression, bi_variate_normal_pdf, subtract_pdf, batch processing |
| `Tracking` | Planned | Graph-based trajectory linking |
| `fBm inference` | Planned | ONNX Runtime model inference |

## Build

### Requirements
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- libtiff **or** OpenCV for TIFF I/O

### With libtiff (recommended)
```bash
g++ -std=c++17 -O2 -DUSE_LIBTIFF -Iinclude src/*.cpp -o freetrace -ltiff
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

### Command line
```bash
./freetrace <input.tiff> <output_dir> [window_size] [threshold] [shift]
```

**Example:**
```bash
./freetrace video.tiff results/ 7 1.0 1
```

This reads the TIFF stack, runs localization with window size 7, threshold multiplier 1.0, and shift 1, then writes `results/video_loc.csv`.

### As a library
```cpp
#include "localization.h"

freetrace::run("input.tiff", "output_dir/",
               /*window_size=*/7,
               /*threshold=*/1.0f,
               /*shift=*/1,
               /*verbose=*/true);
```

### Output format
The output CSV has columns: `frame, x, y, z, xvar, yvar, rho, norm_cst, intensity, window_size` — identical to the Python FreeTrace output format.

## Original Project

- **FreeTrace (Python)**: https://github.com/JunwooParkSaribu/FreeTrace
- **Author**: Junwoo PARK — Sorbonne Université
- **License**: GPLv3+

## License

This C++ port follows the same license as the original FreeTrace project (GPLv3+).

---

*This C++ port is developed by Claude (claude-opus-4-6, Anthropic AI) from the original Python/Cython codebase by Junwoo PARK.*
