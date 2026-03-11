#pragma once // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:30
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include "image_pad.h"
#include "regression.h"

namespace freetrace {

// Localized particle data per frame
struct LocalizationResult {
    // coords[frame] = vector of {x, y, z}
    std::vector<std::vector<std::array<float, 3>>> coords;
    // pdfs[frame] = vector of flattened pdf arrays
    std::vector<std::vector<std::vector<float>>> pdfs;
    // infos[frame] = vector of {x_var, y_var, rho, amp}
    std::vector<std::vector<std::array<float, 4>>> infos;
};

// Window size parameters
struct WinParams {
    int w, h; // width, height (always square in practice)
};

// Detection index: frame, row, col
struct DetIndex {
    int frame, row, col;
};

struct DetIndexWin : DetIndex {
    int win_size;
};

// --- Image I/O ---
// Read TIFF stack, returns flat float data + dimensions
std::vector<float> read_tiff(const std::string& path, int& nb_frames, int& height, int& width);

// Write localization CSV
void write_localization_csv(
    const std::string& output_path,
    const LocalizationResult& result
);

// --- Background estimation ---
// Returns per-window-size background arrays and threshold multipliers
struct BackgroundResult {
    std::map<int, std::vector<float>> bgs;  // key=win_size, value=flat bg array (nb_imgs * win_area)
    std::vector<float> thresholds;           // per-frame thresholds
};
BackgroundResult compute_background(
    const std::vector<float>& imgs, int nb_imgs, int rows, int cols,
    const std::vector<WinParams>& window_sizes, float alpha
);

// --- Gaussian PSF generation ---
Image2D gauss_psf(int win_w, int win_h, float radius);

// --- Region max filter (single-pass) ---
std::vector<DetIndex> region_max_filter2(
    std::vector<float>& maps, int nb_imgs, int rows, int cols,
    const WinParams& window_size, const std::vector<float>& thresholds,
    int detect_range = 0
);

// --- Core localization pipeline ---
LocalizationResult localize(
    const std::vector<float>& imgs, int nb_imgs, int rows, int cols,
    int window_size, float threshold_alpha, int shift = 1
);

// --- Top-level run function ---
bool run(const std::string& input_video_path,
         const std::string& output_path,
         int window_size = 7,
         float threshold = 1.0f,
         int shift = 1,
         bool verbose = false);

} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:30
