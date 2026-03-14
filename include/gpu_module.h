#pragma once // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:40

#include <vector>
#include <string>

namespace freetrace {
namespace gpu {

// Returns true if a CUDA-capable GPU is available
bool is_available();

// Returns free GPU memory in GB
int get_gpu_mem_size();

// Returns free GPU memory in bytes // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
size_t get_gpu_free_mem_bytes();

// GPU-accelerated likelihood computation
// crop_imgs: [nb_imgs * nb_crops * surface_window]
// gauss_grid_data: [surface_window] (flattened)
// bg_squared_sums: [nb_imgs]
// bg_means: [nb_imgs]
// Returns: [nb_imgs * nb_crops]
std::vector<float> likelihood_gpu(
    const std::vector<float>& crop_imgs,
    const std::vector<float>& gauss_grid_data,
    float g_mean,
    const std::vector<float>& bg_squared_sums,
    const std::vector<float>& bg_means,
    int nb_imgs, int nb_crops,
    int surface_window
);

// GPU-accelerated image cropping
// extended_imgs: [nb_imgs * row_size * col_size]
// Returns: [nb_imgs * nb_crops * patch_size]
std::vector<float> image_cropping_gpu(
    const std::vector<float>& extended_imgs,
    int nb_imgs, int row_size, int col_size,
    int extend, int window_size0, int window_size1, int shift,
    int& out_nb_crops
);

// GPU-accelerated background estimation // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
// imgs: [nb_imgs * rows * cols] (already globally normalized)
// out_means, out_stds: [nb_imgs] output arrays (pre-allocated by caller)
void compute_background_gpu(
    const std::vector<float>& imgs,
    int nb_imgs, int rows, int cols,
    float* out_means, float* out_stds
); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

} // namespace gpu
} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:40
