#pragma once // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <array>
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
std::vector<float> read_tiff(const std::string& path, int& nb_frames, int& height, int& width);
void write_localization_csv(const std::string& output_path, const LocalizationResult& result);

// --- Background estimation ---
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

// --- Region max filter (single-window, forward pass) ---
std::vector<DetIndex> region_max_filter2(
    std::vector<float>& maps, int nb_imgs, int rows, int cols,
    const WinParams& window_size, const std::vector<float>& thresholds,
    int detect_range = 0
);

// --- Region max filter (multi-window, backward pass) --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Returns per-frame list of [win_idx, row, col, hmap_val]
std::vector<DetIndexWin> region_max_filter_multi(
    std::vector<float>& maps,  // [nb_wins * nb_imgs * rows * cols]
    int nb_wins, int nb_imgs, int rows, int cols,
    const std::vector<WinParams>& window_sizes,
    const std::vector<float>& thresholds,  // [nb_imgs * nb_wins]
    int detect_range = 0
);

// --- Bivariate normal PDF evaluation ---
// Evaluates unnormalized 2D Gaussian PDF for each image
// Returns [nb_imgs * win_area]
std::vector<float> bi_variate_normal_pdf(
    int nb_imgs, int win_w, int win_h,
    const std::vector<float>& x_var, const std::vector<float>& y_var,
    const std::vector<float>& rho, const std::vector<float>& amp
);

// --- Image regression (Guo algorithm + unpack + PDF) ---
struct RegressionResult {
    std::vector<std::vector<float>> pdfs;  // [nb_imgs][win_area]
    std::vector<float> xs, ys;             // sub-pixel shifts
    std::vector<float> x_vars, y_vars;
    std::vector<float> amps, rhos;
};
RegressionResult image_regression(
    std::vector<float>& imgs,   // [nb_imgs * win_area], modified in-place
    const std::vector<float>& bgs, // [nb_imgs * win_area]
    int nb_imgs, int win_w, int win_h,
    const float* p0, int repeat = 5
);

// --- Subtract PDF (deflation) ---
void subtract_pdf(
    std::vector<float>& ext_imgs,
    int nb_imgs, int ext_rows, int ext_cols,
    const std::vector<std::vector<float>>& pdfs,
    const std::vector<std::array<int, 3>>& indices, // [n, r, c]
    int win_w, int win_h,
    const std::vector<float>& bg_means,
    int extend
);

// --- Core localization function (forward + backward) ---
LocalizationResult localize( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    const std::vector<float>& imgs, int nb_imgs, int rows, int cols,
    int window_size, float threshold_alpha, int shift = 1,
    int deflation_loop_backward = 0,
    NumpyRNG* external_rng = nullptr
);

// --- Core localization from pre-built extended images --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
LocalizationResult localize_from_ext(
    const std::vector<float>& ext_imgs,
    int nb_imgs, int rows, int cols, int ext_rows, int ext_cols,
    int window_size, float threshold_alpha, int shift,
    int extend
);

// --- Top-level run function ---
bool run(const std::string& input_video_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
         const std::string& output_path,
         int window_size = 7,
         float threshold = 1.0f,
         int shift = 1,
         bool verbose = false,
         const std::string& ext_imgs_path = ""); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

// --- 2D density image generation --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
void make_loc_depth_image(const std::string& output_path,
                          const LocalizationResult& result,
                          int multiplier = 4, int winsize = 7, int resolution = 2);

} // namespace freetrace
