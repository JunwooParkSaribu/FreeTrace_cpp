// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:40
#include "gpu_module.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace freetrace {
namespace gpu {

// ============================================================
// GPU availability check
// ============================================================

bool is_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

int get_gpu_mem_size() {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return static_cast<int>(free_mem / (1000ULL * 1000ULL * 1000ULL));
}

// ============================================================
// Likelihood CUDA kernel
// ============================================================
// Each thread handles one (img, crop) pair.
// Computes: min over window, dot product with g_bar, then log-likelihood.

__global__ void likelihood_kernel(
    const float* __restrict__ crop_imgs,   // [nb_imgs * nb_crops * sw]
    const float* __restrict__ g_bar,       // [sw]
    float g_squared_sum,
    const float* __restrict__ bg_squared_sums, // [nb_imgs]
    const float* __restrict__ bg_means,        // [nb_imgs]
    float* __restrict__ L,                     // [nb_imgs * nb_crops]
    int nb_imgs, int nb_crops, int surface_window
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nb_imgs * nb_crops;
    if (idx >= total) return;

    int n = idx / nb_crops;
    int c = idx % nb_crops;

    float bg_mean = bg_means[n];
    float denom = bg_squared_sums[n] - surface_window * bg_mean;
    if (fabsf(denom) < 1e-12f) denom = 1e-12f;

    int base = (n * nb_crops + c) * surface_window;

    // Compute min(crop - bg_mean)
    float i_local_min = 1e30f;
    for (int p = 0; p < surface_window; ++p) {
        float v = crop_imgs[base + p] - bg_mean;
        i_local_min = fminf(i_local_min, v);
    }
    float shift_val = fmaxf(0.0f, i_local_min);

    // Dot product: sum((crop - bg_mean - shift) * g_bar)
    float dot_val = 0.0f;
    for (int p = 0; p < surface_window; ++p) {
        float v = crop_imgs[base + p] - bg_mean - shift_val;
        dot_val += v * g_bar[p];
    }

    float i_hat_proj = dot_val / g_squared_sum;
    i_hat_proj = fmaxf(0.0f, i_hat_proj);

    // L = (N/2) * log(1 - i_hat^2 * g_sq / denom)
    float ratio = i_hat_proj * i_hat_proj * g_squared_sum / denom;
    if (ratio >= 1.0f) ratio = 1.0f - 1e-7f;
    L[n * nb_crops + c] = (surface_window / 2.0f) * logf(1.0f - ratio);
}

std::vector<float> likelihood_gpu(
    const std::vector<float>& crop_imgs,
    const std::vector<float>& gauss_grid_data,
    float g_mean,
    const std::vector<float>& bg_squared_sums,
    const std::vector<float>& bg_means,
    int nb_imgs, int nb_crops,
    int surface_window
) {
    // Compute g_bar on CPU
    std::vector<float> g_bar(surface_window);
    for (int i = 0; i < surface_window; ++i)
        g_bar[i] = gauss_grid_data[i] - g_mean;
    float g_squared_sum = 0.0f;
    for (int i = 0; i < surface_window; ++i)
        g_squared_sum += g_bar[i] * g_bar[i];

    int total = nb_imgs * nb_crops;

    // Allocate device memory
    float *d_crop, *d_gbar, *d_bgsq, *d_bgm, *d_L;
    size_t crop_size = (size_t)nb_imgs * nb_crops * surface_window * sizeof(float);
    size_t gbar_size = surface_window * sizeof(float);
    size_t img_size = nb_imgs * sizeof(float);
    size_t out_size = total * sizeof(float);

    cudaMalloc(&d_crop, crop_size);
    cudaMalloc(&d_gbar, gbar_size);
    cudaMalloc(&d_bgsq, img_size);
    cudaMalloc(&d_bgm, img_size);
    cudaMalloc(&d_L, out_size);

    // Copy to device
    cudaMemcpy(d_crop, crop_imgs.data(), crop_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gbar, g_bar.data(), gbar_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bgsq, bg_squared_sums.data(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bgm, bg_means.data(), img_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    likelihood_kernel<<<grid_size, block_size>>>(
        d_crop, d_gbar, g_squared_sum, d_bgsq, d_bgm, d_L,
        nb_imgs, nb_crops, surface_window
    );

    // Copy result back
    std::vector<float> L(total);
    cudaMemcpy(L.data(), d_L, out_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_crop);
    cudaFree(d_gbar);
    cudaFree(d_bgsq);
    cudaFree(d_bgm);
    cudaFree(d_L);

    return L;
}

// ============================================================
// Image cropping CUDA kernel
// ============================================================
// Each thread handles one (img, crop_index, pixel) triplet.

__global__ void image_cropping_kernel(
    const float* __restrict__ extended_imgs, // [nb_imgs * row_size * col_size]
    float* __restrict__ cropped,             // [nb_imgs * nb_crops * patch_size]
    const int* __restrict__ row_indices,     // [n_rows]
    const int* __restrict__ col_indices,     // [n_cols]
    int nb_imgs, int row_size, int col_size,
    int n_rows, int n_cols,
    int window_size0, int window_size1,
    int patch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nb_crops = n_rows * n_cols;
    int total = nb_imgs * nb_crops * patch_size;
    if (idx >= total) return;

    int n = idx / (nb_crops * patch_size);
    int remainder = idx % (nb_crops * patch_size);
    int crop_idx = remainder / patch_size;
    int pixel_idx = remainder % patch_size;

    int ri_idx = crop_idx / n_cols;
    int ci_idx = crop_idx % n_cols;
    int r = row_indices[ri_idx];
    int c = col_indices[ci_idx];

    int pi = pixel_idx / window_size0;  // row within patch
    int pj = pixel_idx % window_size0;  // col within patch

    int src = n * row_size * col_size + (r + pi) * col_size + (c + pj);
    cropped[idx] = extended_imgs[src];
}

std::vector<float> image_cropping_gpu(
    const std::vector<float>& extended_imgs,
    int nb_imgs, int row_size, int col_size,
    int extend, int window_size0, int window_size1, int shift,
    int& out_nb_crops
) {
    int start_row = extend / 2 - (window_size1 - 1) / 2;
    int end_row = row_size - window_size1 - start_row + 1;
    int start_col = extend / 2 - (window_size0 - 1) / 2;
    int end_col = col_size - window_size0 - start_col + 1;

    std::vector<int> row_indices, col_indices;
    for (int r = start_row; r < end_row; r += shift) row_indices.push_back(r);
    for (int c = start_col; c < end_col; c += shift) col_indices.push_back(c);

    int n_rows = row_indices.size();
    int n_cols = col_indices.size();
    int nb_crops = n_rows * n_cols;
    out_nb_crops = nb_crops;
    int patch_size = window_size0 * window_size1;
    int total = nb_imgs * nb_crops * patch_size;

    // Allocate device memory
    float *d_ext, *d_cropped;
    int *d_rows, *d_cols;
    size_t ext_size = (size_t)nb_imgs * row_size * col_size * sizeof(float);
    size_t crop_size = (size_t)total * sizeof(float);

    cudaMalloc(&d_ext, ext_size);
    cudaMalloc(&d_cropped, crop_size);
    cudaMalloc(&d_rows, n_rows * sizeof(int));
    cudaMalloc(&d_cols, n_cols * sizeof(int));

    cudaMemcpy(d_ext, extended_imgs.data(), ext_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, row_indices.data(), n_rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, col_indices.data(), n_cols * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    image_cropping_kernel<<<grid_size, block_size>>>(
        d_ext, d_cropped, d_rows, d_cols,
        nb_imgs, row_size, col_size,
        n_rows, n_cols,
        window_size0, window_size1, patch_size
    );

    std::vector<float> cropped(total);
    cudaMemcpy(cropped.data(), d_cropped, crop_size, cudaMemcpyDeviceToHost);

    cudaFree(d_ext);
    cudaFree(d_cropped);
    cudaFree(d_rows);
    cudaFree(d_cols);

    return cropped;
}

} // namespace gpu
} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:40
