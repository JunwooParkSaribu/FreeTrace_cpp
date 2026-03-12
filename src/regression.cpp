#include "regression.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include <cmath>
#include <algorithm>

#ifdef USE_LAPACK // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00
extern "C" {
    void dgelsy_(int* m, int* n, int* nrhs, double* a, int* lda,
                 double* b, int* ldb, int* jpvt, double* rcond,
                 int* rank, double* work, int* lwork, int* info);
}
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00

namespace freetrace {

std::vector<std::array<double, 6>> pack_vars(const float* vars, int len_img) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00
    // vars: [x_var(0), x_mu(1), y_var(2), y_mu(3), rho(4)]
    // Use double precision to match Python (scipy lstsq returns float64)
    double rho = vars[4];
    double k = 1.0 - rho * rho;
    double sx = std::sqrt(static_cast<double>(vars[0]));
    double sy = std::sqrt(static_cast<double>(vars[2]));

    double a = -1.0 / (2.0 * vars[0] * k);
    double b = vars[1] / (k * vars[0]) - (rho * vars[3]) / (k * sx * sy);
    double c = -1.0 / (2.0 * vars[2] * k);
    double d = vars[3] / (k * vars[2]) - (rho * vars[1]) / (k * sx * sy);
    double e = rho / (k * sx * sy);
    double f = -(vars[1] * vars[1]) / (2.0 * k * vars[0])
              - (vars[3] * vars[3]) / (2.0 * k * vars[2])
              + (rho * vars[1] * vars[3]) / (k * sx * sy)
              + std::log(1.0 / (2.0 * M_PI * sx * sy * std::sqrt(k)));

    std::vector<std::array<double, 6>> result(len_img);
    for (int i = 0; i < len_img; ++i) {
        result[i] = {a, b, c, d, e, f};
    }
    return result;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00

UnpackResult unpack_coefs(const std::vector<std::array<double, 6>>& coefs, int win_w, int win_h) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00
    int n = coefs.size();
    UnpackResult res;
    res.x_var.resize(n);
    res.x_mu.resize(n);
    res.y_var.resize(n);
    res.y_mu.resize(n);
    res.rho.resize(n);
    res.amp.resize(n);

    for (int i = 0; i < n; ++i) {
        // All intermediate math in double to match Python (numpy float64)
        double rho_i = coefs[i][4] * std::sqrt(1.0 / (4.0 * std::abs(coefs[i][0]) * std::abs(coefs[i][2])));
        double k = 1.0 - rho_i * rho_i;
        double xv = std::abs(1.0 / (-2.0 * coefs[i][0] * k));
        double yv = std::abs(1.0 / (-2.0 * coefs[i][2] * k));

        res.rho[i] = static_cast<float>(rho_i);
        res.x_var[i] = static_cast<float>(xv);
        res.y_var[i] = static_cast<float>(yv);

        // Validate
        if (xv < 0 || yv < 0 || xv > 3 * win_w || yv > 3 * win_h ||
            rho_i < -1 || rho_i > 1 || std::isnan(rho_i)) {
            res.err_indices.push_back(i);
            res.x_mu[i] = 0.0f;
            res.y_mu[i] = 0.0f;
            res.amp[i] = 0.0f;
            continue;
        }

        // Solve 2x2 system for mu using double (matches np.linalg.lstsq float64)
        double sx = std::sqrt(xv);
        double sy = std::sqrt(yv);
        double a00 = -rho_i * sy / sx;
        double a01 = 1.0;
        double a10 = 1.0;
        double a11 = -rho_i * sx / sy;
        double b0 = coefs[i][3] * k * yv;
        double b1 = coefs[i][1] * k * xv;

        double det = a00 * a11 - a01 * a10;
        if (std::abs(det) < 1e-12) {
            res.err_indices.push_back(i);
            res.x_mu[i] = 0.0f;
            res.y_mu[i] = 0.0f;
            res.amp[i] = 0.0f;
            continue;
        }
        double xm = (a11 * b0 - a01 * b1) / det;
        double ym = (a00 * b1 - a10 * b0) / det;
        res.x_mu[i] = static_cast<float>(xm);
        res.y_mu[i] = static_cast<float>(ym);

        res.amp[i] = static_cast<float>(std::exp(coefs[i][5]
            + (xm * xm) / (2.0 * k * xv)
            + (ym * ym) / (2.0 * k * yv)
            - (rho_i * xm * ym) / (k * sx * sy)));
    }
    return res;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00

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

// --- 6x6 least-squares solver --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00
// Solves min ||A*x - b||^2 where A is 6x6, b is 6x1
// A is double (coef_matrix: float64 in Python from iter 1+)
// b is float (ans_matrix: forced float32 in Python via dtype=np.float32)
#ifdef USE_LAPACK
// LAPACK dgelsy — matches Python's scipy.linalg.lstsq(lapack_driver='gelsy')
static bool solve_6x6_lstsq(const double A[6][6], const float b[6], double x[6]) {
    // dgelsy uses column-major order
    double A_col[36]; // 6x6 column-major
    double rhs[6];
    for (int i = 0; i < 6; ++i) {
        rhs[i] = static_cast<double>(b[i]);
        for (int j = 0; j < 6; ++j)
            A_col[j * 6 + i] = A[i][j]; // row-major → column-major
    }
    int m = 6, n = 6, nrhs = 1, lda = 6, ldb = 6;
    int jpvt[6] = {0, 0, 0, 0, 0, 0};
    double rcond = -1.0;
    int rank, info;
    double work[128];
    int lwork = 128;
    dgelsy_(&m, &n, &nrhs, A_col, &lda, rhs, &ldb, jpvt, &rcond, &rank, work, &lwork, &info);
    if (info != 0 || rank == 0) return false;
    for (int i = 0; i < 6; ++i) x[i] = rhs[i];
    return true;
}
#else
// Fallback: Householder QR with column pivoting
static bool solve_6x6_lstsq(const double A[6][6], const float b[6], double x[6]) {
    double R[6][6], rhs[6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) R[i][j] = A[i][j];
        rhs[i] = b[i];
    }
    int perm[6] = {0, 1, 2, 3, 4, 5};
    double col_norms[6];
    for (int j = 0; j < 6; ++j) {
        col_norms[j] = 0;
        for (int i = 0; i < 6; ++i) col_norms[j] += R[i][j] * R[i][j];
    }
    int rank = 0;
    for (int k = 0; k < 6; ++k) {
        int pivot = k;
        double max_norm = col_norms[k];
        for (int j = k + 1; j < 6; ++j) {
            if (col_norms[j] > max_norm) { max_norm = col_norms[j]; pivot = j; }
        }
        if (max_norm < 1e-20) break;
        rank++;
        if (pivot != k) {
            std::swap(perm[k], perm[pivot]);
            std::swap(col_norms[k], col_norms[pivot]);
            for (int i = 0; i < 6; ++i) std::swap(R[i][k], R[i][pivot]);
        }
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
        for (int j = k; j < 6; ++j) {
            double dot = 0;
            for (int i = k; i < 6; ++i) dot += v[i] * R[i][j];
            for (int i = k; i < 6; ++i) R[i][j] -= tau * v[i] * dot;
        }
        double dot = 0;
        for (int i = k; i < 6; ++i) dot += v[i] * rhs[i];
        for (int i = k; i < 6; ++i) rhs[i] -= tau * v[i] * dot;
        for (int j = k + 1; j < 6; ++j) col_norms[j] -= R[k][j] * R[k][j];
    }
    if (rank == 0) return false;
    double x_perm[6] = {};
    for (int i = rank - 1; i >= 0; --i) {
        double s = rhs[i];
        for (int j = i + 1; j < rank; ++j) s -= R[i][j] * x_perm[j];
        if (std::abs(R[i][i]) < 1e-30) return false;
        x_perm[i] = s / R[i][i];
    }
    for (int i = 0; i < 6; ++i) x[perm[i]] = x_perm[i];
    return true;
}
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00

std::vector<std::array<double, 6>> guo_algorithm( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:30
    std::vector<float>& imgs,
    const std::vector<float>& bgs,
    const float* p0,
    int nb_imgs, int win_w, int win_h,
    int repeat
) {
    int win_area = win_w * win_h;

    // Build x_grid and y_grid: pixel coordinates centered on window (float32 like Python)
    std::vector<float> xgrid(win_area), ygrid(win_area);
    for (int r = 0; r < win_h; ++r) {
        for (int c = 0; c < win_w; ++c) {
            xgrid[r * win_w + c] = static_cast<float>(c - win_w / 2);
            ygrid[r * win_w + c] = static_cast<float>(r - win_h / 2);
        }
    }

    // Initialize coefficients from p0 (double precision)
    float p0_arr[5] = {p0[0], p0[1], p0[2], p0[3], p0[4]};
    auto coef_vals = pack_vars(p0_arr, nb_imgs);

    // Subtract background and shift: imgs = max(0, imgs - bgs) + 1e-2
    auto local_bgs = element_wise_subtraction_2d(imgs, bgs, nb_imgs, win_area);
    element_wise_maximum_2d(imgs, local_bgs, nb_imgs, win_area);

    // Pre-compute log(imgs) — clamp to avoid log(0)
    std::vector<float> log_imgs(nb_imgs * win_area);
    for (int i = 0; i < nb_imgs * win_area; ++i)
        log_imgs[i] = std::log(std::max(1e-10f, imgs[i]));

    // yk_2 as double — matches Python where yk_2 becomes float64 from iteration 1+
    // (because coef_vals is float64 from scipy, promotes all numpy ops to float64)
    std::vector<double> yk_2d(nb_imgs * win_area);
    // Iteration 0: yk_2 = imgs (float32 → double)
    for (int i = 0; i < nb_imgs * win_area; ++i)
        yk_2d[i] = imgs[i];

    for (int k = 0; k < repeat; ++k) {
        if (k != 0) {
            // yk_2 = exp(a*x^2 + b*x + c*y^2 + d*y + e*xy + f)
            // coef_vals is double, xgrid is float → promoted to double (matches Python)
            for (int n = 0; n < nb_imgs; ++n) {
                auto& cf = coef_vals[n];
                for (int p = 0; p < win_area; ++p) {
                    double x = xgrid[p], y = ygrid[p];
                    yk_2d[n * win_area + p] = std::exp(
                        cf[0] * x * x + cf[1] * x +
                        cf[2] * y * y + cf[3] * y +
                        cf[4] * x * y + cf[5]);
                }
            }
        }
        // Square yk_2
        for (auto& v : yk_2d) v = v * v;

        // Build and solve 6x6 system for each image
        bool converged = true;
        std::vector<std::array<double, 6>> new_coefs(nb_imgs);

        for (int n = 0; n < nb_imgs; ++n) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00
            // Python precision behavior per iteration:
            //   iter 0: yk_2 is float32 (from imgs.copy()), so coef_matrix elements are float32,
            //           accumulated via np.sum in float32, then assigned to float64 a_mat
            //   iter 1+: yk_2 is float64 (from exp(double_coefs * grid)), so coef_matrix is float64,
            //            accumulated in float64
            // b vector (ans_matrix): always forced to float32 via dtype=np.float32
            double Ad[6][6] = {};
            float bf[6] = {};

            if (k == 0) {
                // Iteration 0: accumulate A in float32, then upcast to double (matching Python)
                float Af[6][6] = {};
                for (int p = 0; p < win_area; ++p) {
                    float w = static_cast<float>(yk_2d[n * win_area + p]);
                    float xf = xgrid[p], yf = ygrid[p];
                    float lg = log_imgs[n * win_area + p];
                    float xf2 = xf * xf, yf2 = yf * yf, xyf = xf * yf;
                    float fbasis[6] = {xf2, xf, yf2, yf, xyf, 1.0f};
                    for (int i = 0; i < 6; ++i) {
                        for (int j = 0; j < 6; ++j)
                            Af[i][j] += w * fbasis[i] * fbasis[j];
                        bf[i] += w * fbasis[i] * lg;
                    }
                }
                // Upcast to double for solve (matches Python: float32 values in float64 a_mat)
                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 6; ++j)
                        Ad[i][j] = Af[i][j];
            } else {
                // Iteration 1+: accumulate A in double (yk_2 is float64 in Python)
                for (int p = 0; p < win_area; ++p) {
                    double w = yk_2d[n * win_area + p];
                    float xf = xgrid[p], yf = ygrid[p];
                    float lg = log_imgs[n * win_area + p];
                    double x = xf, y = yf;
                    double x2 = x * x, y2 = y * y, xy = x * y;
                    double dbasis[6] = {x2, x, y2, y, xy, 1.0};
                    for (int i = 0; i < 6; ++i) {
                        for (int j = 0; j < 6; ++j)
                            Ad[i][j] += w * dbasis[i] * dbasis[j];
                    }
                    // b: float32 accumulation
                    float wf = static_cast<float>(w);
                    float xf2 = xf * xf, yf2 = yf * yf, xyf = xf * yf;
                    float fbasis[6] = {xf2, xf, yf2, yf, xyf, 1.0f};
                    for (int i = 0; i < 6; ++i)
                        bf[i] += wf * fbasis[i] * lg;
                }
            } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:00

            double sol[6];
            if (solve_6x6_lstsq(Ad, bf, sol)) {
                for (int i = 0; i < 6; ++i) new_coefs[n][i] = sol[i];
            } else {
                new_coefs[n] = coef_vals[n];
            }

            // Check convergence (double precision, matching Python np.allclose rtol=1e-7)
            for (int i = 0; i < 6; ++i) {
                if (std::abs(new_coefs[n][i] - coef_vals[n][i]) > 1e-7 * std::abs(coef_vals[n][i]) + 1e-12)
                    converged = false;
            }
        }

        coef_vals = new_coefs;
        if (converged) break;
    }
    return coef_vals;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 10:30

} // namespace freetrace
