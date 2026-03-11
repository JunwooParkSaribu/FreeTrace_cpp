#include "image_pad.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include <cmath>
#include <numeric>

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

    int nb_crops = row_indices.size() * col_indices.size();
    out_nb_crops = nb_crops;
    int patch_size = window_size0 * window_size1;
    std::vector<float> cropped(nb_imgs * nb_crops * patch_size, 0.0f);

    for (int n = 0; n < nb_imgs; ++n) {
        int crop_idx = 0;
        for (int r : row_indices) {
            for (int c : col_indices) {
                int flat_idx = 0;
                for (int ri = 0; ri < window_size1; ++ri) {
                    for (int ci = 0; ci < window_size0; ++ci) {
                        int src = n * row_size * col_size + (r + ri) * col_size + (c + ci);
                        int dst = n * nb_crops * patch_size + crop_idx * patch_size + flat_idx;
                        cropped[dst] = extended_imgs[src];
                        flat_idx++;
                    }
                }
                crop_idx++;
            }
        }
    }
    return cropped;
}

std::vector<float> mapping(
    const std::vector<float>& c_likelihood,
    int nb_img, int row_shape, int col_shape, int shift
) {
    std::vector<float> h_map(nb_img * row_shape * col_shape, 0.0f);
    if (shift == 1) {
        return std::vector<float>(c_likelihood.begin(),
                                  c_likelihood.begin() + nb_img * row_shape * col_shape);
    }
    for (int n = 0; n < nb_img; ++n) {
        int index = 0;
        for (int row = 0; row < row_shape; ++row) {
            for (int col = 0; col < col_shape; ++col) {
                if (row % shift == 0 && col % shift == 0) {
                    h_map[n * row_shape * col_shape + row * col_shape + col] =
                        c_likelihood[n * index + index];
                    index++;
                }
            }
        }
    }
    return h_map;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20

} // namespace freetrace
