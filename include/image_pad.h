#pragma once
#include <vector> // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include <cmath>
#include <algorithm>
#include <random>

namespace freetrace {

struct Image2D {
    std::vector<float> data;
    int rows = 0;
    int cols = 0;

    Image2D() = default;
    Image2D(int r, int c) : data(r * c, 0.0f), rows(r), cols(c) {}
    Image2D(int r, int c, float val) : data(r * c, val), rows(r), cols(c) {}

    float& at(int r, int c) { return data[r * cols + c]; }
    const float& at(int r, int c) const { return data[r * cols + c]; }
    int size() const { return rows * cols; }
};

// Image statistics
float image_mean(const Image2D& img);
float image_std(const Image2D& img);
float image_mean(const float* data, int count);

// Image operations
void image_overlap(Image2D& img1, const Image2D& img2, int div);
void boundary_smoothing(Image2D& img, int row_min_idx, int row_max_idx, int col_min_idx, int col_max_idx);

// Likelihood and cropping
// crop_imgs: [nb_imgs, nb_crops, win_area], gauss_grid: [win_h, win_w]
// Returns: [nb_imgs, nb_crops, 1]
std::vector<float> likelihood(
    const std::vector<float>& crop_imgs,
    const Image2D& gauss_grid,
    const std::vector<float>& bg_squared_sums,
    const std::vector<float>& bg_means,
    int nb_imgs, int nb_crops,
    int window_size1, int window_size2
); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

std::vector<float> image_cropping(
    const std::vector<float>& extended_imgs,
    int nb_imgs, int row_size, int col_size,
    int extend, int window_size0, int window_size1, int shift,
    int& out_nb_crops
);

std::vector<float> mapping(
    const std::vector<float>& c_likelihood,
    int nb_img, int row_shape, int col_shape, int shift
);

// Add block noise to extended image borders (Gaussian noise matching local statistics)
void add_block_noise(
    std::vector<float>& ext_imgs,
    int nb_imgs, int ext_rows, int ext_cols,
    int extend
); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

} // namespace freetrace
