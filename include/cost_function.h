#pragma once // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include <cmath>
#include <tuple>

namespace freetrace {

// Returns (log_pdf, abnormal_flag)
// Predicts the Cauchy-distributed cost for linking particles under fBm
std::pair<double, int> predict_cauchy(
    const double* next_vec, const double* prev_vec,
    float alpha, int lag, float precision, int dimension
);

} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
