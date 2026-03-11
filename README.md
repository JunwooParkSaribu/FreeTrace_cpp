# FreeTrace C++

A high-performance C++ port of [FreeTrace](https://github.com/JunwooParkSaribu/FreeTrace), a single-molecule tracking software with fractional Brownian motion (fBm) inference.

This C++ implementation is being developed by **Claude** (claude-opus-4-6, Anthropic AI), ported from the original Python/Cython project authored by **Junwoo PARK** (junwoo.park@sorbonne-universite.fr).

## About

FreeTrace is a tool for localizing and tracking fluorescent particles in microscopy video data (TIFF stacks). It uses:

- **Sliding-window likelihood detection** to find candidate particle positions
- **Guo's iterative weighted least-squares algorithm** for sub-pixel 2D Gaussian fitting
- **Multi-scale deflation** to resolve overlapping particles
- **Fractional Brownian motion (fBm) inference** via Cauchy log-PDF cost for trajectory linking

The C++ port aims to deliver native performance without Python/Cython overhead, while maintaining algorithmic fidelity to the original.

## Project Structure

```
FreeTrace_cpp/
├── CMakeLists.txt
├── include/
│   ├── image_pad.h        # Image2D struct, statistics, likelihood, cropping, noise
│   ├── regression.h       # Gaussian fitting (Guo algorithm), coefficient packing
│   ├── cost_function.h    # fBm Cauchy log-PDF cost function
│   └── localization.h     # Full localization pipeline
├── src/
│   ├── main.cpp           # Entry point and tests
│   ├── image_pad.cpp      # Image operations implementation
│   ├── regression.cpp     # Regression implementation with 6x6 solver
│   ├── cost_function.cpp  # Cost function implementation
│   └── localization.cpp   # Localization pipeline (forward + backward)
└── models/                # ONNX exported models (planned)
```

## Ported Modules

| Module | Status | Description |
|--------|--------|-------------|
| `image_pad` | Done | Image2D, mean/std, overlap, boundary smoothing, image cropping, mapping, **likelihood**, **add_block_noise** |
| `regression` | Done | pack_vars, unpack_coefs, element-wise ops, matrix_pow_2d, **guo_algorithm** (with inline 6x6 solver) |
| `cost_function` | Done | predict_cauchy (fBm Cauchy log-PDF) |
| `localization` | Done | Full forward + backward pipeline: background estimation, Gaussian PSF, region max filter (single & multi-window), image regression, bivariate normal PDF, subtract PDF (deflation), CSV output |
| `Tracking` | Planned | Graph-based trajectory linking |
| `fBm inference` | Planned | ONNX Runtime model inference |

## Build

### Requirements
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Optional: OpenCV (for TIFF I/O)

### Quick build (no dependencies)
```bash
g++ -std=c++17 -O2 -Iinclude src/*.cpp -o freetrace
```

### With OpenCV (enables TIFF reading)
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

Currently the binary runs built-in tests. Full CLI usage with TIFF input is available when compiled with OpenCV:

```cpp
#include "localization.h"

// Run localization on a TIFF stack
freetrace::run("input.tiff", "output_dir/",
               /*window_size=*/7,
               /*threshold=*/1.0f,
               /*shift=*/1,
               /*verbose=*/true);
```

## Original Project

- **FreeTrace (Python)**: https://github.com/JunwooParkSaribu/FreeTrace
- **Author**: Junwoo PARK — Sorbonne Université
- **License**: GPLv3+

## License

This C++ port follows the same license as the original FreeTrace project (GPLv3+).

---

*This C++ port is developed by Claude (claude-opus-4-6, Anthropic AI) from the original Python/Cython codebase by Junwoo PARK.*
