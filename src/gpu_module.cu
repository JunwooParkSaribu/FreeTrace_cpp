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

size_t get_gpu_free_mem_bytes() { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

// ============================================================
// Likelihood CUDA kernel
// ============================================================
// Each thread handles one (img, crop) pair.
// Computes: min over window, dot product with g_bar, then log-likelihood.

__global__ void likelihood_kernel( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
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

    // Use double precision internally to match CPU arithmetic
    double bg_mean = (double)bg_means[n];
    double g_sq = (double)g_squared_sum;
    double denom = (double)bg_squared_sums[n] - surface_window * bg_mean;
    if (fabs(denom) < 1e-12) denom = 1e-12;

    int base = (n * nb_crops + c) * surface_window;

    // Compute min(crop - bg_mean)
    double i_local_min = 1e30;
    for (int p = 0; p < surface_window; ++p) {
        double v = (double)crop_imgs[base + p] - bg_mean;
        i_local_min = fmin(i_local_min, v);
    }
    double shift_val = fmax(0.0, i_local_min);

    // Dot product: sum((crop - bg_mean - shift) * g_bar)
    double dot_val = 0.0;
    for (int p = 0; p < surface_window; ++p) {
        double v = (double)crop_imgs[base + p] - bg_mean - shift_val;
        dot_val += v * (double)g_bar[p];
    }

    double i_hat_proj = dot_val / g_sq;
    i_hat_proj = fmax(0.0, i_hat_proj);

    // L = (N/2) * log(1 - i_hat^2 * g_sq / denom)
    double ratio = i_hat_proj * i_hat_proj * g_sq / denom;
    if (ratio >= 1.0) ratio = 1.0 - 1e-7;
    L[n * nb_crops + c] = (float)((surface_window / 2.0) * log(1.0 - ratio));
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

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

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
// Background estimation CUDA kernel
// ============================================================
// One block per frame. Each block:
//   1. Finds per-frame max, normalizes, quantizes to int bins 0-100
//   2. Iterates 3x: histogram → mode → std → update mask
//   3. Outputs final bg_mean and bg_std
// Shared memory: float reduce[BLOCK_SIZE] + int hist[101]

__global__ void background_kernel( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    const float* __restrict__ imgs,     // [nb_imgs * pixel_count]
    float* __restrict__ out_means,      // [nb_imgs]
    float* __restrict__ out_stds,       // [nb_imgs]
    int nb_imgs, int pixel_count,
    double frame0_std                   // std of frame 0 (used for iter 0 of ALL frames)
) {
    int frame = blockIdx.x;
    if (frame >= nb_imgs) return;

    extern __shared__ char smem[];
    // Use double for reductions to match CPU float64 precision
    double* s_reduce = (double*)smem;
    int* s_hist = (int*)(s_reduce + blockDim.x);

    const float* fdata = imgs + frame * pixel_count;
    int tid = threadIdx.x;
    int BS = blockDim.x;

    // --- Step 1: per-frame max (parallel reduction) ---
    double tmax = 0.0;
    for (int i = tid; i < pixel_count; i += BS)
        tmax = fmax(tmax, (double)fdata[i]);
    s_reduce[tid] = tmax;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] = fmax(s_reduce[tid], s_reduce[tid + s]);
        __syncthreads();
    }
    double fmax_val = s_reduce[0];
    if (fmax_val <= 0.0) fmax_val = 1.0;
    __syncthreads();

    // --- 3 iterations: histogram → mode → std → mask ---
    double lo = -1e30, hi = 1e30;
    double mode_val = 0.0;

    for (int iter = 0; iter < 3; iter++) {
        // Clear histogram
        for (int i = tid; i < 101; i += BS) s_hist[i] = 0;
        __syncthreads();

        // Build histogram + find max_ival among masked pixels // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        // Match Python: quantize in float32 to replicate (imgs * 100).astype(np.uint8)
        int t_max_ival = 0;
        for (int i = tid; i < pixel_count; i += BS) {
            float normed_f = fdata[i] / (float)fmax_val;
            int ival = (int)((unsigned char)(normed_f * 100.0f));
            double val = ival / 100.0;
            if (val > lo && val < hi) {
                // np.histogram bin assignment: correct for np.arange float64 edge
                int b = ival;
                if (b > 0 && val < b * 0.01)
                    b--;
                atomicAdd(&s_hist[b], 1);
                t_max_ival = max(t_max_ival, b);
            }
        }

        // Reduce max_ival (now max_bin after edge correction)
        s_reduce[tid] = (double)t_max_ival;
        __syncthreads();
        for (int s = BS / 2; s > 0; s >>= 1) {
            if (tid < s) s_reduce[tid] = fmax(s_reduce[tid], s_reduce[tid + s]);
            __syncthreads();
        }
        int max_bin = (int)(s_reduce[0] + 0.5);
        __syncthreads();

        // np.histogram: nb_bins = max_ival (not max_ival+1), last bin closed
        // Any values in bins >= nb_bins get clamped into nb_bins-1
        // Since we already applied edge correction, just use max_bin as nb_bins
        int nb_bins = max(1, max_bin);

        // Clamp overflow bins into last bin (np.histogram last-bin-closed rule)
        if (tid == 0) {
            for (int b = nb_bins; b <= 100; b++) {
                if (s_hist[b] > 0) {
                    s_hist[nb_bins - 1] += s_hist[b];
                    s_hist[b] = 0;
                }
            }
        }
        __syncthreads();

        // Find mode (thread 0)
        if (tid == 0) {
            int mode_bin = 0, mode_count = 0;
            for (int b = 0; b < nb_bins; b++) {
                if (s_hist[b] > mode_count) {
                    mode_count = s_hist[b];
                    mode_bin = b;
                }
            }
            s_reduce[0] = mode_bin * 0.01 + 0.005;
        }
        __syncthreads();
        mode_val = s_reduce[0];
        __syncthreads();

        // Compute std: iter 0 uses frame0_std for all frames (matching Python GPU) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        double mask_std = 0.0;
        if (iter == 0) {
            mask_std = frame0_std;
        } else {
            double tsum = 0.0, tsum2 = 0.0, tcount = 0.0;
            for (int i = tid; i < pixel_count; i += BS) {
                float normed_f = fdata[i] / (float)fmax_val;
                int ival = (int)((unsigned char)(normed_f * 100.0f));
                double val = ival / 100.0;
                if (val > lo && val < hi) {
                    tsum += val;
                    tsum2 += val * val;
                    tcount += 1.0;
                }
            }
            s_reduce[tid] = tsum;
            __syncthreads();
            for (int s = BS / 2; s > 0; s >>= 1) {
                if (tid < s) s_reduce[tid] += s_reduce[tid + s];
                __syncthreads();
            }
            double total_sum = s_reduce[0];
            __syncthreads();
            s_reduce[tid] = tsum2;
            __syncthreads();
            for (int s = BS / 2; s > 0; s >>= 1) {
                if (tid < s) s_reduce[tid] += s_reduce[tid + s];
                __syncthreads();
            }
            double total_sum2 = s_reduce[0];
            __syncthreads();
            s_reduce[tid] = tcount;
            __syncthreads();
            for (int s = BS / 2; s > 0; s >>= 1) {
                if (tid < s) s_reduce[tid] += s_reduce[tid + s];
                __syncthreads();
            }
            double total_count = s_reduce[0];
            __syncthreads();
            if (total_count > 0.5) {
                double mean_tmp = total_sum / total_count;
                mask_std = sqrt(fmax(0.0, total_sum2 / total_count - mean_tmp * mean_tmp));
            }
        }

        lo = mode_val - 3.0 * mask_std;
        hi = mode_val + 3.0 * mask_std; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    }

    // --- Final: compute mean/std from final mask --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    double tsum = 0.0, tsum2 = 0.0, tcount = 0.0;
    for (int i = tid; i < pixel_count; i += BS) {
        float normed_f = fdata[i] / (float)fmax_val;
        int ival = (int)((unsigned char)(normed_f * 100.0f));
        double val = ival / 100.0;
        if (val > lo && val < hi) {
            tsum += val;
            tsum2 += val * val;
            tcount += 1.0;
        }
    }

    s_reduce[tid] = tsum;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    double final_sum = s_reduce[0];
    __syncthreads();

    s_reduce[tid] = tsum2;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    double final_sum2 = s_reduce[0];
    __syncthreads();

    s_reduce[tid] = tcount;
    __syncthreads();
    for (int s = BS / 2; s > 0; s >>= 1) {
        if (tid < s) s_reduce[tid] += s_reduce[tid + s];
        __syncthreads();
    }
    double final_count = s_reduce[0];

    if (tid == 0) {
        if (final_count > 0.5) {
            double mean_d = final_sum / final_count;
            out_means[frame] = (float)mean_d;
            out_stds[frame] = (float)sqrt(fmax(0.0, final_sum2 / final_count - mean_d * mean_d));
        } else {
            out_means[frame] = 0.0f;
            out_stds[frame] = 0.0f;
        }
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

void compute_background_gpu(
    const std::vector<float>& imgs,
    int nb_imgs, int rows, int cols,
    float* out_means, float* out_stds
) {
    int pixel_count = rows * cols;

    // Allocate device memory
    float *d_imgs, *d_means, *d_stds;
    size_t imgs_size = (size_t)nb_imgs * pixel_count * sizeof(float);
    cudaMalloc(&d_imgs, imgs_size);
    cudaMalloc(&d_means, nb_imgs * sizeof(float));
    cudaMalloc(&d_stds, nb_imgs * sizeof(float));

    cudaMemcpy(d_imgs, imgs.data(), imgs_size, cudaMemcpyHostToDevice);

    // Pre-compute frame 0's std on CPU (matching Python GPU's flat cp.take behavior) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    // Frame 0 is already normalized to [0,1] with max=1.0
    double f0_sum = 0.0, f0_sum2 = 0.0;
    for (int i = 0; i < pixel_count; ++i) {
        float normed_f = imgs[i] / 1.0f; // frame 0, fmax=1.0
        int ival = (int)((unsigned char)(normed_f * 100.0f));
        double val = ival / 100.0;
        f0_sum += val;
        f0_sum2 += val * val;
    }
    double f0_mean = f0_sum / pixel_count;
    double frame0_std = std::sqrt(std::max(0.0, f0_sum2 / pixel_count - f0_mean * f0_mean));

    int block_size = 256;
    size_t smem_size = block_size * sizeof(double) + 101 * sizeof(int);
    background_kernel<<<nb_imgs, block_size, smem_size>>>(
        d_imgs, d_means, d_stds, nb_imgs, pixel_count, frame0_std
    ); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

    cudaMemcpy(out_means, d_means, nb_imgs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_stds, d_stds, nb_imgs * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_imgs);
    cudaFree(d_means);
    cudaFree(d_stds);
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

} // namespace gpu
} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:40
