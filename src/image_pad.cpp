#include "image_pad.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include <cmath>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace freetrace {

float image_mean(const Image2D& img) {
    if (img.size() == 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < img.size(); ++i) {
        sum += img.data[i];
    }
    return sum / img.size();
}

float image_mean(const float* data, int count) {
    if (count == 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        sum += data[i];
    }
    return sum / count;
}

float image_std(const Image2D& img) {
    if (img.size() == 0) return 0.0f;
    float mean = image_mean(img);
    float var = 0.0f;
    for (int i = 0; i < img.size(); ++i) {
        float diff = img.data[i] - mean;
        var += diff * diff;
    }
    return std::sqrt(var / img.size());
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

std::vector<float> likelihood( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
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

    // g_squared_sum = sum(g_bar^2)
    float g_squared_sum = 0.0f;
    for (int i = 0; i < surface_window; ++i)
        g_squared_sum += g_bar[i] * g_bar[i];

    // Result: [nb_imgs * nb_crops] // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    std::vector<float> L(nb_imgs * nb_crops, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < nb_imgs; ++n) {
        float bg_mean = bg_means[n];
        float denom = bg_squared_sums[n] - surface_window * bg_mean;
        if (std::abs(denom) < 1e-12f) denom = 1e-12f;

        for (int c = 0; c < nb_crops; ++c) {
            int base = n * nb_crops * surface_window + c * surface_window;

            // i_hat = crop - bg_mean, then subtract max(0, min(i_hat))
            float i_local_min = 1e30f;
            for (int p = 0; p < surface_window; ++p) {
                float v = crop_imgs[base + p] - bg_mean;
                i_local_min = std::min(i_local_min, v);
            }
            float shift_val = std::max(0.0f, i_local_min);

            // i_hat_proj = dot(i_hat - shift, g_bar) / g_squared_sum
            float dot_val = 0.0f;
            for (int p = 0; p < surface_window; ++p) {
                float v = crop_imgs[base + p] - bg_mean - shift_val;
                dot_val += v * g_bar[p];
            }
            float i_hat_proj = dot_val / g_squared_sum;
            i_hat_proj = std::max(0.0f, i_hat_proj);

            // L = (N/2) * log(1 - i_hat^2 * g_squared_sum / denom)
            float ratio = i_hat_proj * i_hat_proj * g_squared_sum / denom;
            if (ratio >= 1.0f) ratio = 1.0f - 1e-7f;
            L[n * nb_crops + c] = (surface_window / 2.0f) * std::log(1.0f - ratio);
        }
    }
    return L;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

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

void add_block_noise( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float>& ext_imgs,
    int nb_imgs, int ext_rows, int ext_cols,
    int extend
) {
    int gap = extend / 2;
    std::mt19937 rng(42);

    // Helper lambda: fill a block with Gaussian noise matching stats of a reference block
    auto fill_noise = [&](int n, int dst_r0, int dst_r1, int dst_c0, int dst_c1,
                          int ref_r0, int ref_r1, int ref_c0, int ref_c1) {
        if (dst_r1 <= dst_r0 || dst_c1 <= dst_c0) return;
        if (ref_r1 <= ref_r0 || ref_c1 <= ref_c0) return;
        // Compute mean and std of reference
        float sum = 0.0f, sum2 = 0.0f;
        int cnt = 0;
        for (int r = ref_r0; r < ref_r1; ++r)
            for (int c = ref_c0; c < ref_c1; ++c) {
                float v = ext_imgs[n * ext_rows * ext_cols + r * ext_cols + c];
                sum += v; sum2 += v * v; cnt++;
            }
        float mean = (cnt > 0) ? sum / cnt : 0.0f;
        float var = (cnt > 1) ? sum2 / cnt - mean * mean : 0.0f;
        float sd = (var > 0) ? std::sqrt(var) : 0.0f;
        std::normal_distribution<float> dist(mean, sd);
        for (int r = dst_r0; r < dst_r1; ++r)
            for (int c = dst_c0; c < dst_c1; ++c)
                ext_imgs[n * ext_rows * ext_cols + r * ext_cols + c] = dist(rng);
    };

    for (int n = 0; n < nb_imgs; ++n) {
        // Top border
        fill_noise(n, 0, gap, gap, ext_cols - gap,
                      gap, 2 * gap, gap, ext_cols - gap);
        // Right border
        fill_noise(n, 0, ext_rows, ext_cols - gap, ext_cols,
                      0, ext_rows, ext_cols - 2 * gap, ext_cols - gap);
        // Bottom border
        fill_noise(n, ext_rows - gap, ext_rows, gap, ext_cols - gap,
                      ext_rows - 2 * gap, ext_rows - gap, gap, ext_cols - gap);
        // Left border
        fill_noise(n, 0, ext_rows, 0, gap,
                      0, ext_rows, gap, 2 * gap);
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

} // namespace freetrace
