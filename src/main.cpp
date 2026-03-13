#include <iostream> // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
#include <string>
#include <cstring>
#include "image_pad.h"
#include "regression.h"
#include "cost_function.h"
#include "localization.h"
#include "tracking.h"
#include "gpu_module.h"

static void print_usage() { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    std::cout << "FreeTrace C++\n"
              << "Usage:\n"
              << "  Tracking:\n"
              << "    freetrace track <loc.csv> <output_dir> <nb_frames> [options]\n"
              << "  Options:\n"
              << "    --depth N        Graph depth (default: 3)\n"
              << "    --cutoff N       Cutoff (default: 3)\n"
              << "    --jump F         Maximum jump distance in px (default: auto)\n"
              << "    --tiff PATH      TIFF file for image dimensions and output naming\n"
              << "    --no-fbm         Disable fBm mode (no NN, fixed alpha/K, no H-K output)\n"
              << "    --postprocess    Enable post-processing\n"
              << "    --quiet          Suppress status messages\n"
              << "\n"
              << "  Localization:\n"
              << "    freetrace <input.tiff> <output_dir> [window_size] [threshold] [shift]\n";
}

int main(int argc, char* argv[]) {
    // Tracking mode: freetrace track <loc.csv> <output_dir> <nb_frames> [options]
    if (argc >= 5 && std::string(argv[1]) == "track") {
        std::string loc_csv = argv[2];
        std::string output = argv[3];
        int nb_frames = std::stoi(argv[4]);

        freetrace::TrackingConfig config;
        config.verbose = true;
        // Parse optional args // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        for (int i = 5; i < argc; i++) {
            if (std::strcmp(argv[i], "--depth") == 0 && i + 1 < argc) config.graph_depth = std::stoi(argv[++i]);
            else if (std::strcmp(argv[i], "--cutoff") == 0 && i + 1 < argc) config.cutoff = std::stoi(argv[++i]);
            else if (std::strcmp(argv[i], "--jump") == 0 && i + 1 < argc) config.jump_threshold = std::stof(argv[++i]);
            else if (std::strcmp(argv[i], "--tiff") == 0 && i + 1 < argc) config.tiff_path = argv[++i];
            else if (std::strcmp(argv[i], "--no-fbm") == 0) { config.fbm_mode = false; config.use_nn = false; config.hk_output = false; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
            else if (std::strcmp(argv[i], "--postprocess") == 0) config.post_process = true;
            else if (std::strcmp(argv[i], "--quiet") == 0) config.verbose = false;
            else if (std::strcmp(argv[i], "--init-k") == 0 && i + 1 < argc) config.init_k = std::stof(argv[++i]);
            else if (std::strcmp(argv[i], "--init-alpha") == 0 && i + 1 < argc) config.init_alpha = std::stof(argv[++i]);
            else { std::cerr << "Unknown option: " << argv[i] << std::endl; print_usage(); return 1; }
        }

        // Read TIFF dimensions if provided
        if (!config.tiff_path.empty()) {
            int tiff_frames, tiff_h, tiff_w;
            freetrace::read_tiff(config.tiff_path, tiff_frames, tiff_h, tiff_w);
            config.img_rows = tiff_h;
            config.img_cols = tiff_w;
        }

        // GPU detection: use GPU if available, otherwise notify user // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        bool gpu_avail = freetrace::gpu::is_available();
        if (!gpu_avail && config.verbose) {
            if (config.fbm_mode) {
                std::cout << "\n  ** Note: No GPU detected. FreeTrace is running on CPU with NN inference — this will be slow. **\n" << std::endl;
            } else {
                std::cout << "\n  ** Note: No GPU detected. FreeTrace is running on CPU. **\n" << std::endl;
            }
        } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

        // Banner
        std::cout << "FreeTrace C++ — Tracking" << std::endl;
        std::cout << "  Loc CSV:   " << loc_csv << std::endl;
        std::cout << "  Output:    " << output << std::endl;
        std::cout << "  Nb frames: " << nb_frames << std::endl;
        std::cout << "  Depth:     " << config.graph_depth << ", Cutoff: " << config.cutoff << std::endl;
        if (config.jump_threshold > 0)
            std::cout << "  Jump threshold: " << config.jump_threshold << " px" << std::endl;
        else
            std::cout << "  Jump threshold: auto (inferred from data)" << std::endl;
        std::cout << "  fBm mode: " << (config.fbm_mode ? "ON (NN inference + H-K output)" : "OFF (fixed alpha/K)") << std::endl;
        if (config.post_process) std::cout << "  Post-processing: enabled" << std::endl;
        if (gpu_avail) std::cout << "  GPU: available" << std::endl;
        if (config.img_rows > 0) std::cout << "  Image: " << config.img_cols << "x" << config.img_rows << " (from TIFF)" << std::endl;

        bool ok = freetrace::run_tracking(loc_csv, output, nb_frames, config);
        return ok ? 0 : 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    }

    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage();
        return 0;
    }

    if (argc >= 3 && std::string(argv[1]) != "track") {
        // Localization mode: freetrace <input.tiff> <output_dir> [window_size] [threshold] [shift]
        std::string input = argv[1];
        std::string output = argv[2];
        int win_size = (argc > 3) ? std::stoi(argv[3]) : 7;
        float threshold = (argc > 4) ? std::stof(argv[4]) : 1.0f;
        int shift = (argc > 5) ? std::stoi(argv[5]) : 1;
        std::string ext_imgs = (argc > 6) ? argv[6] : "";

        std::cout << "FreeTrace C++ — Localization" << std::endl;
        std::cout << "  Input:  " << input << std::endl;
        std::cout << "  Output: " << output << std::endl;
        std::cout << "  Window: " << win_size << ", Threshold: " << threshold
                  << ", Shift: " << shift << std::endl;
        if (!ext_imgs.empty()) std::cout << "  Ext imgs: " << ext_imgs << std::endl;

        bool ok = freetrace::run(input, output, win_size, threshold, shift, /*verbose=*/true, ext_imgs);
        return ok ? 0 : 1;
    }

    // No arguments: print usage
    print_usage();
    return 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
}
