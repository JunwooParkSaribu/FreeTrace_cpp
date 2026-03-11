#include <iostream> // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <string>
#include "image_pad.h"
#include "regression.h"
#include "cost_function.h"
#include "localization.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

int main(int argc, char* argv[]) {
    if (argc >= 3) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        // Localization mode: freetrace <input.tiff> <output_dir> [window_size] [threshold] [shift]
        std::string input = argv[1];
        std::string output = argv[2];
        int win_size = (argc > 3) ? std::stoi(argv[3]) : 7;
        float threshold = (argc > 4) ? std::stof(argv[4]) : 1.0f;
        int shift = (argc > 5) ? std::stoi(argv[5]) : 1;
        std::string ext_imgs = (argc > 6) ? argv[6] : ""; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        std::cout << "FreeTrace C++ — Localization" << std::endl;
        std::cout << "  Input:  " << input << std::endl;
        std::cout << "  Output: " << output << std::endl;
        std::cout << "  Window: " << win_size << ", Threshold: " << threshold
                  << ", Shift: " << shift << std::endl;
        if (!ext_imgs.empty()) std::cout << "  Ext imgs: " << ext_imgs << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        bool ok = freetrace::run(input, output, win_size, threshold, shift, /*verbose=*/true, ext_imgs); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        return ok ? 0 : 1;
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Default: unit tests
    std::cout << "FreeTrace C++ v0.1.0" << std::endl;

    freetrace::Image2D img(3, 3);
    img.at(0, 0) = 1.0f; img.at(0, 1) = 2.0f; img.at(0, 2) = 3.0f;
    img.at(1, 0) = 4.0f; img.at(1, 1) = 5.0f; img.at(1, 2) = 6.0f;
    img.at(2, 0) = 7.0f; img.at(2, 1) = 8.0f; img.at(2, 2) = 9.0f;

    std::cout << "Image mean: " << freetrace::image_mean(img) << " (expected 5.0)" << std::endl;
    std::cout << "Image std:  " << freetrace::image_std(img) << " (expected ~2.58)" << std::endl;

    double next_vec[] = {1.5, 2.0};
    double prev_vec[] = {1.0, 1.0};
    auto [log_pdf, abnormal] = freetrace::predict_cauchy(next_vec, prev_vec, 1.0f, 0, 1e-6f, 2);
    std::cout << "Cauchy log_pdf: " << log_pdf << ", abnormal: " << abnormal << std::endl;

    float vars[] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    auto packed = freetrace::pack_vars(vars, 1);
    std::cout << "Packed coefs: [" << packed[0][0] << ", " << packed[0][1] << ", "
              << packed[0][2] << ", " << packed[0][3] << ", "
              << packed[0][4] << ", " << packed[0][5] << "]" << std::endl;

    std::cout << "All tests passed." << std::endl;
    return 0;
}
