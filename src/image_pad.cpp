#include "image_pad.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include <cmath>
#include <numeric>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_ACCELERATE // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
#include <Accelerate/Accelerate.h>
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

namespace freetrace {

float image_mean(const Image2D& img) {
    if (img.size() == 0) return 0.0f;
#ifdef USE_ACCELERATE // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    float result;
    vDSP_meanv(img.data.data(), 1, &result, img.size());
    return result;
#else
    float sum = 0.0f;
    for (int i = 0; i < img.size(); ++i) {
        sum += img.data[i];
    }
    return sum / img.size();
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
}

float image_mean(const float* data, int count) {
    if (count == 0) return 0.0f;
#ifdef USE_ACCELERATE // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    float result;
    vDSP_meanv(data, 1, &result, count);
    return result;
#else
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        sum += data[i];
    }
    return sum / count;
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
}

float image_std(const Image2D& img) {
    if (img.size() == 0) return 0.0f;
    float mean = image_mean(img);
#ifdef USE_ACCELERATE // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    // Subtract mean, square, then compute mean of squares
    std::vector<float> diff(img.size());
    float neg_mean = -mean;
    vDSP_vsadd(img.data.data(), 1, &neg_mean, diff.data(), 1, img.size());
    float sum_sq;
    vDSP_dotpr(diff.data(), 1, diff.data(), 1, &sum_sq, img.size());
    return std::sqrt(sum_sq / img.size());
#else
    float var = 0.0f;
    for (int i = 0; i < img.size(); ++i) {
        float diff = img.data[i] - mean;
        var += diff * diff;
    }
    return std::sqrt(var / img.size());
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
}

void image_overlap(Image2D& img1, const Image2D& img2, int div) {
    for (int r = 0; r < img1.rows; ++r) {
        for (int c = 0; c < img1.cols; ++c) {
            img1.at(r, c) = (img1.at(r, c) + img2.at(r, c)) / div;
        }
    }
}

void boundary_smoothing(Image2D& img, int row_min_idx, int row_max_idx, int col_min_idx, int col_max_idx) {
    const int border_max = 50;
    const int erase_space = 2;
    const int repeat_n = 2;
    int height = img.rows;
    int width = img.cols;

    // Collect border pixel coordinates
    std::vector<std::pair<int, int>> center_xy;
    center_xy.reserve(border_max * border_max);

    for (int border = 0; border < border_max; ++border) {
        int rmin = std::max(0, row_min_idx + border);
        int rmax = std::min(height - 1, row_max_idx - border);
        int cmin = std::max(0, col_min_idx + border);
        int cmax = std::min(width - 1, col_max_idx - border);
        for (int col = cmin; col <= cmax; ++col)
            center_xy.push_back({rmin, col});
        for (int row = rmin; row <= rmax; ++row)
            center_xy.push_back({row, cmax});
        for (int col = cmax; col >= cmin; --col)
            center_xy.push_back({rmax, col});
        for (int row = rmax; row >= rmin; --row)
            center_xy.push_back({row, cmin});
    }

    // Smooth border pixels by local mean
    for (int rep = 0; rep < repeat_n; ++rep) {
        for (auto& [r, c] : center_xy) {
            int r0 = std::max(0, r - erase_space);
            int r1 = std::min(height, r + erase_space + 1);
            int c0 = std::max(0, c - erase_space);
            int c1 = std::min(width, c + erase_space + 1);
            float sum = 0.0f;
            int count = 0;
            for (int ri = r0; ri < r1; ++ri) {
                for (int ci = c0; ci < c1; ++ci) {
                    sum += img.at(ri, ci);
                    count++;
                }
            }
            img.at(r, c) = (count > 0) ? sum / count : 0.0f;
        }
    }
}

std::vector<float> image_cropping(
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

    int nb_crops = row_indices.size() * col_indices.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    out_nb_crops = nb_crops;
    int patch_size = window_size0 * window_size1;
    int n_rows = row_indices.size();
    int n_cols = col_indices.size();
    std::vector<float> cropped(nb_imgs * nb_crops * patch_size, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < nb_imgs; ++n) {
        for (int ri_idx = 0; ri_idx < n_rows; ++ri_idx) {
            int r = row_indices[ri_idx];
            for (int ci_idx = 0; ci_idx < n_cols; ++ci_idx) {
                int c = col_indices[ci_idx];
                int crop_idx = ri_idx * n_cols + ci_idx;
                int flat_idx = 0;
                for (int ri = 0; ri < window_size1; ++ri) {
                    for (int ci = 0; ci < window_size0; ++ci) {
                        int src = n * row_size * col_size + (r + ri) * col_size + (c + ci);
                        int dst = n * nb_crops * patch_size + crop_idx * patch_size + flat_idx;
                        cropped[dst] = extended_imgs[src];
                        flat_idx++;
                    }
                }
            }
        }
    }
    return cropped; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
}

std::vector<float> likelihood( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    const std::vector<float>& crop_imgs,
    const Image2D& gauss_grid,
    const std::vector<float>& bg_squared_sums,
    const std::vector<float>& bg_means,
    int nb_imgs, int nb_crops,
    int window_size1, int window_size2
) {
    int surface_window = window_size1 * window_size2;

    // Compute g_bar = gauss_grid - mean(gauss_grid)
    float g_mean = image_mean(gauss_grid);
    std::vector<float> g_bar(surface_window);
    for (int i = 0; i < window_size1; ++i)
        for (int j = 0; j < window_size2; ++j)
            g_bar[i * window_size2 + j] = gauss_grid.data[i * gauss_grid.cols + j] - g_mean;

    // g_squared_sum = sum(g_bar^2) — use double for precision
    double g_squared_sum = 0.0;
    for (int i = 0; i < surface_window; ++i)
        g_squared_sum += (double)g_bar[i] * (double)g_bar[i];

    // Result: [nb_imgs * nb_crops]
    std::vector<float> L(nb_imgs * nb_crops, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < nb_imgs; ++n) {
        double bg_mean = (double)bg_means[n];
        double denom = (double)bg_squared_sums[n] - surface_window * bg_mean;
        if (std::abs(denom) < 1e-12) denom = 1e-12;

#ifdef USE_ACCELERATE // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        // vDSP-accelerated path (Apple Silicon)
        std::vector<float> i_hat(surface_window);
        float neg_bg_mean = -(float)bg_mean;

        for (int c = 0; c < nb_crops; ++c) {
            int base = n * nb_crops * surface_window + c * surface_window;

            // i_hat = crop - bg_mean (vDSP_vsadd: vector + scalar)
            vDSP_vsadd(crop_imgs.data() + base, 1, &neg_bg_mean, i_hat.data(), 1, surface_window);

            // Find min(i_hat)
            float local_min;
            vDSP_minv(i_hat.data(), 1, &local_min, surface_window);
            float shift_val = std::max(0.0f, local_min);

            // i_hat -= shift_val
            float neg_shift = -shift_val;
            vDSP_vsadd(i_hat.data(), 1, &neg_shift, i_hat.data(), 1, surface_window);

            // dot(i_hat, g_bar)
            float dot_f;
            vDSP_dotpr(i_hat.data(), 1, g_bar.data(), 1, &dot_f, surface_window);

            double i_hat_proj = (double)dot_f / g_squared_sum;
            i_hat_proj = std::max(0.0, i_hat_proj);

            double ratio = i_hat_proj * i_hat_proj * g_squared_sum / denom;
            if (ratio >= 1.0) ratio = 1.0 - 1e-7;
            L[n * nb_crops + c] = (float)((surface_window / 2.0) * std::log(1.0 - ratio));
        }
#else // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        // Scalar fallback path
        for (int c = 0; c < nb_crops; ++c) {
            int base = n * nb_crops * surface_window + c * surface_window;

            // i_hat = crop - bg_mean, then subtract max(0, min(i_hat))
            double i_local_min = 1e30;
            for (int p = 0; p < surface_window; ++p) {
                double v = (double)crop_imgs[base + p] - bg_mean;
                i_local_min = std::min(i_local_min, v);
            }
            double shift_val = std::max(0.0, i_local_min);

            // i_hat_proj = dot(i_hat - shift, g_bar) / g_squared_sum
            double dot_val = 0.0;
            for (int p = 0; p < surface_window; ++p) {
                double v = (double)crop_imgs[base + p] - bg_mean - shift_val;
                dot_val += v * (double)g_bar[p];
            }
            double i_hat_proj = dot_val / g_squared_sum;
            i_hat_proj = std::max(0.0, i_hat_proj);

            // L = (N/2) * log(1 - i_hat^2 * g_squared_sum / denom)
            double ratio = i_hat_proj * i_hat_proj * g_squared_sum / denom;
            if (ratio >= 1.0) ratio = 1.0 - 1e-7;
            L[n * nb_crops + c] = (float)((surface_window / 2.0) * std::log(1.0 - ratio));
        }
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    }
    return L;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

std::vector<float> mapping(
    const std::vector<float>& c_likelihood,
    int nb_img, int row_shape, int col_shape, int shift
) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> h_map(nb_img * row_shape * col_shape, 0.0f);
    if (shift == 1) {
        // c_likelihood has nb_img * row_shape * col_shape elements
        for (int i = 0; i < nb_img * row_shape * col_shape && i < static_cast<int>(c_likelihood.size()); ++i)
            h_map[i] = c_likelihood[i];
        return h_map;
    }
    for (int n = 0; n < nb_img; ++n) {
        int index = 0;
        for (int row = 0; row < row_shape; ++row) {
            for (int col = 0; col < col_shape; ++col) {
                if (row % shift == 0 && col % shift == 0) {
                    h_map[n * row_shape * col_shape + row * col_shape + col] =
                        c_likelihood[n * ((row_shape + shift - 1) / shift) * ((col_shape + shift - 1) / shift) + index];
                    index++;
                }
            }
        }
    }
    return h_map;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Compute mean and std of a sub-block (matching numpy's mean/std with ddof=0)
static void block_mean_std(
    const std::vector<float>& imgs, int base,
    int ext_cols, int r0, int r1, int c0, int c1,
    double& mean, double& sd
) {
    double sum = 0.0;
    int cnt = (r1 - r0) * (c1 - c0);
    if (cnt <= 0) { mean = 0; sd = 0; return; }
    for (int r = r0; r < r1; ++r)
        for (int c = c0; c < c1; ++c)
            sum += imgs[base + r * ext_cols + c];
    mean = sum / cnt;
    double sum2 = 0.0;
    for (int r = r0; r < r1; ++r)
        for (int c = c0; c < c1; ++c) {
            double d = imgs[base + r * ext_cols + c] - mean;
            sum2 += d * d;
        }
    sd = std::sqrt(sum2 / cnt);
}

void add_block_noise( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    std::vector<float>& ext_imgs,
    int nb_imgs, int ext_rows, int ext_cols,
    int extend,
    NumpyRNG* external_rng
) {
    int gap = extend / 2;
    NumpyRNG local_rng(42);
    NumpyRNG& rng = external_rng ? *external_rng : local_rng;

    // Build row_indice and col_indice matching Python's range(0, ext_rows, gap) etc.
    std::vector<int> row_indice, col_indice;
    for (int r = 0; r < ext_rows; r += gap) row_indice.push_back(r);
    for (int c = 0; c < ext_cols; c += gap) col_indice.push_back(c);

    // Helper: fill a block for ALL frames sequentially (matching Python's per-frame loop)
    // Python: for m, std in zip(crop_means, crop_stds):
    //             np.random.normal(loc=m, scale=std, size=(rows, cols))
    // This generates rows*cols normals per frame, in row-major order
    auto fill_block = [&](int dst_r0, int dst_r1, int dst_c0, int dst_c1,
                          int ref_r0, int ref_r1, int ref_c0, int ref_c1) {
        int dst_h = dst_r1 - dst_r0;
        int dst_w = dst_c1 - dst_c0;
        int ref_h = ref_r1 - ref_r0;
        int ref_w = ref_c1 - ref_c0;
        if (dst_h <= 0 || dst_w <= 0 || ref_h <= 0 || ref_w <= 0) return false;
        for (int n = 0; n < nb_imgs; ++n) {
            int base = n * ext_rows * ext_cols;
            double mean, sd;
            block_mean_std(ext_imgs, base, ext_cols, ref_r0, ref_r1, ref_c0, ref_c1, mean, sd);
            // Generate dst_h * dst_w normals in row-major order
            for (int r = dst_r0; r < dst_r1; ++r)
                for (int c = dst_c0; c < dst_c1; ++c)
                    ext_imgs[base + r * ext_cols + c] = rng.normal(mean, sd);
        }
        return true;
    };

    // Helper for smoothed blocks: reference is average of two sub-blocks
    auto fill_block_smoothed = [&](int dst_r0, int dst_r1, int dst_c0, int dst_c1,
                                   int refA_r0, int refA_r1, int refA_c0, int refA_c1,
                                   int refB_r0, int refB_r1, int refB_c0, int refB_c1) {
        int dst_h = dst_r1 - dst_r0;
        int dst_w = dst_c1 - dst_c0;
        if (dst_h <= 0 || dst_w <= 0) return;
        int refA_h = refA_r1 - refA_r0, refA_w = refA_c1 - refA_c0;
        int refB_h = refB_r1 - refB_r0, refB_w = refB_c1 - refB_c0;
        if (refA_h <= 0 || refA_w <= 0 || refB_h <= 0 || refB_w <= 0) return;
        for (int n = 0; n < nb_imgs; ++n) {
            int base = n * ext_rows * ext_cols;
            // Compute mean/std of averaged block (A+B)/2
            int cnt = refA_h * refA_w; // both blocks should be same size
            double sum = 0.0;
            for (int i = 0; i < refA_h; ++i)
                for (int j = 0; j < refA_w; ++j) {
                    double a = ext_imgs[base + (refA_r0 + i) * ext_cols + (refA_c0 + j)];
                    double b = ext_imgs[base + (refB_r0 + i) * ext_cols + (refB_c0 + j)];
                    sum += (a + b) / 2.0;
                }
            double mean = sum / cnt;
            double sum2 = 0.0;
            for (int i = 0; i < refA_h; ++i)
                for (int j = 0; j < refA_w; ++j) {
                    double a = ext_imgs[base + (refA_r0 + i) * ext_cols + (refA_c0 + j)];
                    double b = ext_imgs[base + (refB_r0 + i) * ext_cols + (refB_c0 + j)];
                    double d = (a + b) / 2.0 - mean;
                    sum2 += d * d;
                }
            double sd = std::sqrt(sum2 / cnt);
            for (int r = dst_r0; r < dst_r1; ++r)
                for (int c = dst_c0; c < dst_c1; ++c)
                    ext_imgs[base + r * ext_cols + c] = rng.normal(mean, sd);
        }
    };

    // ---- STEP 1: Top border ----
    // Python: for c in col_indice:
    //   ref = imgs[:, gap:2*gap, c:min(ext_cols-gap, c+gap)]
    //   dst = imgs[:, 0:gap, c:min(ext_cols-gap, c+gap)]
    for (int ci = 0; ci < (int)col_indice.size(); ++ci) {
        int c = col_indice[ci];
        int c1 = std::min(ext_cols - gap, c + gap);
        if (c1 <= c) break; // crop_img.shape[2] == 0
        fill_block(0, gap, c, c1,
                   gap, 2 * gap, c, c1);
    }

    // ---- STEP 2: Right border ----
    // Python: for r in row_indice:
    //   ref = imgs[:, r:min(ext_rows-gap, r+gap), ext_cols-2*gap:ext_cols-gap]
    //   dst = imgs[:, r:min(ext_rows-gap, r+gap), ext_cols-gap:ext_cols]
    for (int ri = 0; ri < (int)row_indice.size(); ++ri) {
        int r = row_indice[ri];
        int r1 = std::min(ext_rows - gap, r + gap);
        if (r1 <= r) break; // crop_img.shape[1] == 0
        fill_block(r, r1, ext_cols - gap, ext_cols,
                   r, r1, ext_cols - 2 * gap, ext_cols - gap);
    }

    // ---- STEP 3: Bottom border (reversed col order) ----
    // Python: for c in col_indice[::-1]:
    //   ref = imgs[:, ext_rows-2*gap:ext_rows-gap, c:min(ext_cols, c+gap)]
    //   dst = imgs[:, ext_rows-gap:ext_rows, c:min(ext_cols, c+gap)]
    for (int ci = (int)col_indice.size() - 1; ci >= 0; --ci) {
        int c = col_indice[ci];
        int c1 = std::min(ext_cols, c + gap);
        if (c1 <= c) continue; // crop_img.shape[2] == 0
        fill_block(ext_rows - gap, ext_rows, c, c1,
                   ext_rows - 2 * gap, ext_rows - gap, c, c1);
    }

    // ---- STEP 4: Left border ----
    // Python: for r in row_indice:
    //   ref = imgs[:, r:min(ext_rows, r+gap), gap:2*gap]
    //   dst = imgs[:, r:min(ext_rows, r+gap), 0:gap]
    for (int ri = 0; ri < (int)row_indice.size(); ++ri) {
        int r = row_indice[ri];
        int r1 = std::min(ext_rows, r + gap);
        fill_block(r, r1, 0, gap,
                   r, r1, gap, 2 * gap);
    }

    // ---- STEP 5: Smoothing - top border (col_indice[1:-1]) ----
    // Python: for c in col_indice[1:-1]:
    //   csize = min(ext_cols, c+2*gap) - c - gap
    //   refA = imgs[:, 0:gap, c-csize:c]
    //   refB = imgs[:, 0:gap, c+gap:c+gap+csize]
    //   ref = (refA + refB) / 2
    //   dst = imgs[:, 0:gap, c:min(ext_cols, c+gap)]
    for (int ci = 1; ci < (int)col_indice.size() - 1; ++ci) {
        int c = col_indice[ci];
        int csize = std::min(ext_cols, c + 2 * gap) - c - gap;
        int dc1 = std::min(ext_cols, c + gap);
        fill_block_smoothed(0, gap, c, dc1,
                            0, gap, c - csize, c,
                            0, gap, c + gap, c + gap + csize);
    }

    // ---- STEP 6: Smoothing - right border (row_indice[1:-1]) ----
    // Python: for r in row_indice[1:-1]:
    //   rsize = min(ext_rows, r+2*gap) - r - gap
    //   refA = imgs[:, r-rsize:r, ext_cols-2*gap:ext_cols-gap]
    //   refB = imgs[:, r+gap:r+gap+rsize, ext_cols-2*gap:ext_cols-gap]
    //   ref = (refA + refB) / 2
    //   dst = imgs[:, r:min(ext_rows, r+gap), ext_cols-gap:ext_cols]
    for (int ri = 1; ri < (int)row_indice.size() - 1; ++ri) {
        int r = row_indice[ri];
        int rsize = std::min(ext_rows, r + 2 * gap) - r - gap;
        int dr1 = std::min(ext_rows, r + gap);
        fill_block_smoothed(r, dr1, ext_cols - gap, ext_cols,
                            r - rsize, r, ext_cols - 2 * gap, ext_cols - gap,
                            r + gap, r + gap + rsize, ext_cols - 2 * gap, ext_cols - gap);
    }

    // ---- STEP 7: Smoothing - bottom border (col_indice[1:-1]) ----
    // Python: for c in col_indice[1:-1]:
    //   csize = min(ext_cols, c+2*gap) - c - gap
    //   refA = imgs[:, ext_rows-2*gap:ext_rows-gap, c-csize:c]
    //   refB = imgs[:, ext_rows-2*gap:ext_rows-gap, c+gap:c+gap+csize]
    //   ref = (refA + refB) / 2
    //   dst = imgs[:, ext_rows-gap:ext_rows, c:min(ext_cols, c+gap)]
    for (int ci = 1; ci < (int)col_indice.size() - 1; ++ci) {
        int c = col_indice[ci];
        int csize = std::min(ext_cols, c + 2 * gap) - c - gap;
        int dc1 = std::min(ext_cols, c + gap);
        fill_block_smoothed(ext_rows - gap, ext_rows, c, dc1,
                            ext_rows - 2 * gap, ext_rows - gap, c - csize, c,
                            ext_rows - 2 * gap, ext_rows - gap, c + gap, c + gap + csize);
    }

    // ---- STEP 8: Smoothing - left border (row_indice[1:-1]) ---- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    // Python: for r in row_indice[1:-1]:
    //   rsize = min(ext_rows, r+2*gap) - r - gap
    //   refA = imgs[:, r-rsize:r, col_indice[0]:col_indice[0]+gap]  (= 0:gap, left padding)
    //   refB = imgs[:, r+gap:r+gap+rsize, col_indice[0]:col_indice[0]+gap]
    //   ref = (refA + refB) / 2
    //   dst = imgs[:, r:min(ext_rows, r+gap), 0:gap]
    for (int ri = 1; ri < (int)row_indice.size() - 1; ++ri) {
        int r = row_indice[ri];
        int rsize = std::min(ext_rows, r + 2 * gap) - r - gap;
        int dr1 = std::min(ext_rows, r + gap);
        fill_block_smoothed(r, dr1, 0, gap,
                            r - rsize, r, 0, gap,
                            r + gap, r + gap + rsize, 0, gap);
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

} // namespace freetrace
