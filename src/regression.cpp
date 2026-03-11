#include "regression.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include <cmath>
#include <algorithm>

namespace freetrace {

std::vector<std::array<float, 6>> pack_vars(const float* vars, int len_img) {
    // vars: [x_var(0), x_mu(1), y_var(2), y_mu(3), rho(4)]
    float rho = vars[4];
    float k = 1.0f - rho * rho;
    float sx = std::sqrt(vars[0]);
    float sy = std::sqrt(vars[2]);

    float a = -1.0f / (2.0f * vars[0] * k);
    float b = vars[1] / (k * vars[0]) - (rho * vars[3]) / (k * sx * sy);
    float c = -1.0f / (2.0f * vars[2] * k);
    float d = vars[3] / (k * vars[2]) - (rho * vars[1]) / (k * sx * sy);
    float e = rho / (k * sx * sy);
    float f = -(vars[1] * vars[1]) / (2.0f * k * vars[0])
              - (vars[3] * vars[3]) / (2.0f * k * vars[2])
              + (rho * vars[1] * vars[3]) / (k * sx * sy)
              + std::log(1.0f / (2.0f * M_PI * sx * sy * std::sqrt(k)));

    std::vector<std::array<float, 6>> result(len_img);
    for (int i = 0; i < len_img; ++i) {
        result[i] = {a, b, c, d, e, f};
    }
    return result;
}

UnpackResult unpack_coefs(const std::vector<std::array<float, 6>>& coefs, int win_w, int win_h) {
    int n = coefs.size();
    UnpackResult res;
    res.x_var.resize(n);
    res.x_mu.resize(n);
    res.y_var.resize(n);
    res.y_mu.resize(n);
    res.rho.resize(n);
    res.amp.resize(n);

    for (int i = 0; i < n; ++i) {
        float rho_i = coefs[i][4] * std::sqrt(1.0f / (4.0f * std::abs(coefs[i][0]) * std::abs(coefs[i][2])));
        float k = 1.0f - rho_i * rho_i;
        float xv = std::abs(1.0f / (-2.0f * coefs[i][0] * k));
        float yv = std::abs(1.0f / (-2.0f * coefs[i][2] * k));

        res.rho[i] = rho_i;
        res.x_var[i] = xv;
        res.y_var[i] = yv;

        // Validate
        if (xv < 0 || yv < 0 || xv > 3 * win_w || yv > 3 * win_h ||
            rho_i < -1 || rho_i > 1 || std::isnan(rho_i)) {
            res.err_indices.push_back(i);
            res.x_mu[i] = 0.0f;
            res.y_mu[i] = 0.0f;
            res.amp[i] = 0.0f;
            continue;
        }

        // Solve 2x2 system for mu
        float sx = std::sqrt(xv);
        float sy = std::sqrt(yv);
        float a00 = -rho_i * sy / sx;
        float a01 = 1.0f;
        float a10 = 1.0f;
        float a11 = -rho_i * sx / sy;
        float b0 = coefs[i][3] * k * yv;
        float b1 = coefs[i][1] * k * xv;

        float det = a00 * a11 - a01 * a10;
        if (std::abs(det) < 1e-12f) {
            res.err_indices.push_back(i);
            res.x_mu[i] = 0.0f;
            res.y_mu[i] = 0.0f;
            res.amp[i] = 0.0f;
            continue;
        }
        res.x_mu[i] = (a11 * b0 - a01 * b1) / det;
        res.y_mu[i] = (a00 * b1 - a10 * b0) / det;

        float xm = res.x_mu[i], ym = res.y_mu[i];
        res.amp[i] = std::exp(coefs[i][5]
            + (xm * xm) / (2.0f * k * xv)
            + (ym * ym) / (2.0f * k * yv)
            - (rho_i * xm * ym) / (k * sx * sy));
    }
    return res;
}

std::vector<float> element_wise_subtraction_2d(
    std::vector<float>& array1, const std::vector<float>& array2,
    int nb_imgs, int img_size
) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
    std::vector<float> local_bgs(nb_imgs);
    for (int i = 0; i < nb_imgs; ++i) {
        float local_bg = 9999.0f;
        for (int j = 0; j < img_size; ++j) {
            int idx = i * img_size + j;
            array1[idx] -= array2[idx];
            local_bg = std::min(local_bg, array1[idx]);
        }
        local_bgs[i] = local_bg;
    }
    return local_bgs;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20

void element_wise_maximum_2d(
    std::vector<float>& array1, const std::vector<float>& local_bgs,
    int nb_imgs, int img_size
) {
    for (int i = 0; i < nb_imgs; ++i) {
        for (int j = 0; j < img_size; ++j) {
            array1[i * img_size + j] = array1[i * img_size + j] - local_bgs[i] + 1e-2f;
        }
    }
}

void matrix_pow_2d(std::vector<float>& array, int rows, int cols, int power) {
    if (power == 0) {
        std::fill(array.begin(), array.end(), 1.0f);
        return;
    }
    if (power >= 2) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int idx = i * cols + j;
                float tmp = array[idx];
                for (int k = 1; k < power; ++k) {
                    array[idx] *= tmp;
                }
            }
        }
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20

} // namespace freetrace
