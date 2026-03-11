#include <iostream> // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
#include "image_pad.h"
#include "regression.h"
#include "cost_function.h"

int main(int argc, char* argv[]) {
    std::cout << "FreeTrace C++ v0.1.0" << std::endl;

    // Test image_pad: create a small image and compute mean/std
    freetrace::Image2D img(3, 3);
    img.at(0, 0) = 1.0f; img.at(0, 1) = 2.0f; img.at(0, 2) = 3.0f;
    img.at(1, 0) = 4.0f; img.at(1, 1) = 5.0f; img.at(1, 2) = 6.0f;
    img.at(2, 0) = 7.0f; img.at(2, 1) = 8.0f; img.at(2, 2) = 9.0f;

    std::cout << "Image mean: " << freetrace::image_mean(img) << " (expected 5.0)" << std::endl;
    std::cout << "Image std:  " << freetrace::image_std(img) << " (expected ~2.58)" << std::endl;

    // Test cost_function: predict_cauchy with BM (alpha=1)
    double next_vec[] = {1.5, 2.0};
    double prev_vec[] = {1.0, 1.0};
    auto [log_pdf, abnormal] = freetrace::predict_cauchy(next_vec, prev_vec, 1.0f, 0, 1e-6f, 2);
    std::cout << "Cauchy log_pdf: " << log_pdf << ", abnormal: " << abnormal << std::endl;

    // Test pack_vars
    float vars[] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    auto packed = freetrace::pack_vars(vars, 1);
    std::cout << "Packed coefs: [" << packed[0][0] << ", " << packed[0][1] << ", "
              << packed[0][2] << ", " << packed[0][3] << ", "
              << packed[0][4] << ", " << packed[0][5] << "]" << std::endl;

    std::cout << "All tests passed." << std::endl;
    return 0;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
