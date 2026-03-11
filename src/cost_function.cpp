#include "cost_function.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include <cmath>

namespace freetrace {

std::pair<double, int> predict_cauchy(
    const double* next_vec, const double* prev_vec,
    float alpha, int lag, float precision, int dimension
) {
    double log_pdf = 0.0;
    int abnormal = 0;
    int delta_t = lag + 1;

    for (int dim_i = 0; dim_i < dimension; ++dim_i) {
        double vec1 = next_vec[dim_i];
        double vec2 = prev_vec[dim_i];

        if (vec1 < 0) vec1 -= precision;
        else          vec1 += precision;
        if (vec2 < 0) vec2 -= precision;
        else          vec2 += precision;

        double coord_ratio = vec1 / vec2;

        if (alpha > 0.95 && alpha < 1.05) {
            if (std::abs(coord_ratio) > 8)
                abnormal = 1;
            log_pdf += std::log(1.0 / M_PI * 1.0 / (coord_ratio * coord_ratio + 1.0));
        } else {
            double rho = 0.5 * (std::pow(delta_t - 1, alpha) - 2.0 * std::pow(delta_t, alpha) + std::pow(delta_t + 1, alpha));
            double relativ_cov = 0.5 * (std::pow(delta_t + 1, alpha) - std::pow(delta_t, alpha) - std::pow(1, alpha));
            double scale = std::sqrt(std::abs(1.0 - rho * rho));

            if (std::abs(coord_ratio - rho) > 8.0 * scale)
                abnormal = 1;

            double z = (coord_ratio - relativ_cov) / scale;
            log_pdf += std::log(1.0 / (M_PI * scale) * 1.0 / (z * z * (rho / relativ_cov) + (relativ_cov / rho)));
        }
    }

    return {log_pdf, abnormal};
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20

} // namespace freetrace
