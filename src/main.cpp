#include <iostream> // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <string>
#include <cstring>
#include "image_pad.h"
#include "regression.h"
#include "cost_function.h"
#include "localization.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include "tracking.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

int main(int argc, char* argv[]) {
    // Tracking mode: freetrace track <loc.csv> <output_dir> <nb_frames> [options] // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (argc >= 5 && std::string(argv[1]) == "track") {
        std::string loc_csv = argv[2]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::string output = argv[3]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int nb_frames = std::stoi(argv[4]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        freetrace::TrackingConfig config; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        config.verbose = true;
        std::string tiff_path; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        // Parse optional args // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int i = 5; i < argc; i++) {
            if (std::strcmp(argv[i], "--depth") == 0 && i + 1 < argc) config.graph_depth = std::stoi(argv[++i]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            else if (std::strcmp(argv[i], "--cutoff") == 0 && i + 1 < argc) config.cutoff = std::stoi(argv[++i]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            else if (std::strcmp(argv[i], "--jump") == 0 && i + 1 < argc) config.jump_threshold = std::stof(argv[++i]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            else if (std::strcmp(argv[i], "--tiff") == 0 && i + 1 < argc) tiff_path = argv[++i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            else if (std::strcmp(argv[i], "--postprocess") == 0) config.post_process = true; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            else if (std::strcmp(argv[i], "--nn") == 0) config.use_nn = true; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }

        // Read TIFF dimensions if provided // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (!tiff_path.empty()) {
            int tiff_frames, tiff_h, tiff_w;
            freetrace::read_tiff(tiff_path, tiff_frames, tiff_h, tiff_w); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            config.img_rows = tiff_h;
            config.img_cols = tiff_w;
        }

        std::cout << "FreeTrace C++ — Tracking" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::cout << "  Loc CSV:   " << loc_csv << std::endl;
        std::cout << "  Output:    " << output << std::endl;
        std::cout << "  Nb frames: " << nb_frames << std::endl;
        std::cout << "  Depth:     " << config.graph_depth << ", Cutoff: " << config.cutoff << std::endl;
        if (config.jump_threshold > 0) std::cout << "  Jump threshold: " << config.jump_threshold << " px" << std::endl;
        if (config.post_process) std::cout << "  Post-processing: enabled" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (config.use_nn) std::cout << "  NN inference: enabled" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (config.img_rows > 0) std::cout << "  Image: " << config.img_cols << "x" << config.img_rows << " (from TIFF)" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        bool ok = freetrace::run_tracking(loc_csv, output, nb_frames, config); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        return ok ? 0 : 1;
    }

    if (argc >= 3 && std::string(argv[1]) != "track") { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
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
