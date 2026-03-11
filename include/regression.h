#pragma once // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include <vector>
#include <cmath>
#include <array>

namespace freetrace {

// Pack Gaussian parameters into coefficient matrix [a, b, c, d, e, f]
// vars = [x_var, x_mu, y_var, y_mu, rho]
std::vector<std::array<float, 6>> pack_vars(const float* vars, int len_img);

// Unpack fitted coefficients back to Gaussian parameters
// Returns: [x_var, x_mu, y_var, y_mu, rho, amp] and error indices
struct UnpackResult {
    std::vector<float> x_var, x_mu, y_var, y_mu, rho, amp;
    std::vector<int> err_indices;
};
UnpackResult unpack_coefs(const std::vector<std::array<float, 6>>& coefs, int win_w, int win_h);

// Element-wise operations on flattened 2D arrays
std::vector<float> element_wise_subtraction_2d(
    std::vector<float>& array1, const std::vector<float>& array2,
    int nb_imgs, int img_size
);

void element_wise_maximum_2d(
    std::vector<float>& array1, const std::vector<float>& local_bgs,
    int nb_imgs, int img_size
);

void matrix_pow_2d(std::vector<float>& array, int rows, int cols, int power);

} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
