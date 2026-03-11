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
}

// --- 6x6 least-squares solver via Householder QR --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Solves min ||A*x - b||^2 where A is 6x6, b is 6x1
// Uses Householder QR decomposition (matches scipy lstsq behavior)
static bool solve_6x6_lstsq(const float A[6][6], const float b[6], float x[6]) {
    // Work on copies: augmented [A | b] stored as Q*R
    double R[6][6], rhs[6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) R[i][j] = A[i][j];
        rhs[i] = b[i];
    }

    // Householder QR with column pivoting
    int perm[6] = {0, 1, 2, 3, 4, 5};
    double col_norms[6];
    for (int j = 0; j < 6; ++j) {
        col_norms[j] = 0;
        for (int i = 0; i < 6; ++i) col_norms[j] += R[i][j] * R[i][j];
    }

    int rank = 0;
    for (int k = 0; k < 6; ++k) {
        // Column pivoting: find column with largest remaining norm
        int pivot = k;
        double max_norm = col_norms[k];
        for (int j = k + 1; j < 6; ++j) {
            if (col_norms[j] > max_norm) {
                max_norm = col_norms[j];
                pivot = j;
            }
        }
        if (max_norm < 1e-20) break; // Remaining columns are zero
        rank++;

        // Swap columns k and pivot
        if (pivot != k) {
            std::swap(perm[k], perm[pivot]);
            std::swap(col_norms[k], col_norms[pivot]);
            for (int i = 0; i < 6; ++i) std::swap(R[i][k], R[i][pivot]);
        }

        // Compute Householder vector for column k
        double sigma = 0;
        for (int i = k + 1; i < 6; ++i) sigma += R[i][k] * R[i][k];
        double norm_x = std::sqrt(R[k][k] * R[k][k] + sigma);
        if (norm_x < 1e-30) continue;

        double v0 = (R[k][k] >= 0) ? R[k][k] + norm_x : R[k][k] - norm_x;
        double scale = 1.0 / v0;
        double v[6] = {};
        v[k] = 1.0;
        for (int i = k + 1; i < 6; ++i) v[i] = R[i][k] * scale;
        double tau = 2.0 / (1.0 + sigma * scale * scale);

        // Apply Householder: R = (I - tau*v*v^T) * R
        for (int j = k; j < 6; ++j) {
            double dot = 0;
            for (int i = k; i < 6; ++i) dot += v[i] * R[i][j];
            for (int i = k; i < 6; ++i) R[i][j] -= tau * v[i] * dot;
        }
        // Apply to rhs
        double dot = 0;
        for (int i = k; i < 6; ++i) dot += v[i] * rhs[i];
        for (int i = k; i < 6; ++i) rhs[i] -= tau * v[i] * dot;

        // Update column norms for remaining columns
        for (int j = k + 1; j < 6; ++j)
            col_norms[j] -= R[k][j] * R[k][j];
    }

    if (rank == 0) return false;

    // Back substitution: R[0:rank, 0:rank] * x_perm[0:rank] = rhs[0:rank]
    double x_perm[6] = {};
    for (int i = rank - 1; i >= 0; --i) {
        double s = rhs[i];
        for (int j = i + 1; j < rank; ++j) s -= R[i][j] * x_perm[j];
        if (std::abs(R[i][i]) < 1e-30) return false;
        x_perm[i] = s / R[i][i];
    }

    // Un-permute
    for (int i = 0; i < 6; ++i) x[perm[i]] = static_cast<float>(x_perm[i]);
    return true;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

std::vector<std::array<float, 6>> guo_algorithm( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float>& imgs,
    const std::vector<float>& bgs,
    const float* p0,
    int nb_imgs, int win_w, int win_h,
    int repeat
) {
    int win_area = win_w * win_h;

    // Build x_grid and y_grid: pixel coordinates centered on window
    std::vector<float> xgrid(win_area), ygrid(win_area);
    for (int r = 0; r < win_h; ++r) {
        for (int c = 0; c < win_w; ++c) {
            xgrid[r * win_w + c] = static_cast<float>(c - win_w / 2);
            ygrid[r * win_w + c] = static_cast<float>(r - win_h / 2);
        }
    }

    // Initialize coefficients from p0
    float p0_arr[5] = {p0[0], p0[1], p0[2], p0[3], p0[4]};
    auto coef_vals = pack_vars(p0_arr, nb_imgs);

    // Subtract background and shift: imgs = max(0, imgs - bgs) + 1e-2
    auto local_bgs = element_wise_subtraction_2d(imgs, bgs, nb_imgs, win_area);
    element_wise_maximum_2d(imgs, local_bgs, nb_imgs, win_area);

    // Pre-compute log(imgs) — clamp to avoid log(0)
    std::vector<float> log_imgs(nb_imgs * win_area);
    for (int i = 0; i < nb_imgs * win_area; ++i)
        log_imgs[i] = std::log(std::max(1e-10f, imgs[i]));

    // yk_2 starts as imgs copy
    std::vector<float> yk_2(imgs.begin(), imgs.end());

    for (int k = 0; k < repeat; ++k) {
        if (k != 0) {
            // yk_2 = exp(a*x^2 + b*x + c*y^2 + d*y + e*xy + f)
            for (int n = 0; n < nb_imgs; ++n) {
                auto& cf = coef_vals[n];
                for (int p = 0; p < win_area; ++p) {
                    float x = xgrid[p], y = ygrid[p];
                    yk_2[n * win_area + p] = std::exp(
                        cf[0] * x * x + cf[1] * x +
                        cf[2] * y * y + cf[3] * y +
                        cf[4] * x * y + cf[5]);
                }
            }
        }
        // Square yk_2
        for (auto& v : yk_2) v = v * v;

        // Build and solve 6x6 system for each image
        bool converged = true;
        std::vector<std::array<float, 6>> new_coefs(nb_imgs);

        for (int n = 0; n < nb_imgs; ++n) {
            // Accumulate 6x6 coef_matrix and 6x1 ans_matrix
            float A[6][6] = {};
            float b[6] = {};

            for (int p = 0; p < win_area; ++p) {
                float w = yk_2[n * win_area + p];
                float x = xgrid[p], y = ygrid[p];
                float lg = log_imgs[n * win_area + p];

                float x2 = x * x, y2 = y * y, xy = x * y;
                // Basis vector: [x^2, x, y^2, y, x*y, 1]
                float basis[6] = {x2, x, y2, y, xy, 1.0f};

                for (int i = 0; i < 6; ++i) {
                    b[i] += w * basis[i] * lg;
                    for (int j = 0; j < 6; ++j)
                        A[i][j] += w * basis[i] * basis[j];
                }
            }

            float sol[6];
            if (solve_6x6_lstsq(A, b, sol)) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                for (int i = 0; i < 6; ++i) new_coefs[n][i] = sol[i];
            } else {
                new_coefs[n] = coef_vals[n]; // keep previous
            }

            // Check convergence
            for (int i = 0; i < 6; ++i) {
                if (std::abs(new_coefs[n][i] - coef_vals[n][i]) > 1e-7f * std::abs(coef_vals[n][i]) + 1e-12f)
                    converged = false;
            }
        }

        coef_vals = new_coefs;
        if (converged) break;
    }
    return coef_vals;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

} // namespace freetrace
