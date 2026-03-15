#include <iostream> // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
#include <string>
#include <cstring>
#include "image_pad.h"
#include "regression.h"
#include "cost_function.h"
#include "localization.h"
#include "tracking.h"
#include "gpu_module.h"

// Shared tracking options (used by both "track" and full-pipeline modes)
struct TrackingOpts { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    freetrace::TrackingConfig config;
    bool quiet = false;
};

static TrackingOpts parse_tracking_opts(int argc, char* argv[], int start_idx) {
    TrackingOpts opts;
    opts.config.verbose = true;
    for (int i = start_idx; i < argc; i++) {
        if (std::strcmp(argv[i], "--depth") == 0 && i + 1 < argc) opts.config.graph_depth = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cutoff") == 0 && i + 1 < argc) opts.config.cutoff = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--jump") == 0 && i + 1 < argc) opts.config.jump_threshold = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--tiff") == 0 && i + 1 < argc) opts.config.tiff_path = argv[++i];
        else if (std::strcmp(argv[i], "--no-fbm") == 0) { opts.config.fbm_mode = false; opts.config.use_nn = false; opts.config.hk_output = false; }
        else if (std::strcmp(argv[i], "--postprocess") == 0) opts.config.post_process = true;
        else if (std::strcmp(argv[i], "--quiet") == 0) { opts.config.verbose = false; opts.quiet = true; }
        else if (std::strcmp(argv[i], "--init-k") == 0 && i + 1 < argc) opts.config.init_k = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--init-alpha") == 0 && i + 1 < argc) opts.config.init_alpha = std::stof(argv[++i]);
        // Localization options (ignored here, parsed separately)
        else if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc) ++i;
        else if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) ++i;
        else if (std::strcmp(argv[i], "--shift") == 0 && i + 1 < argc) ++i;
        else if (std::strcmp(argv[i], "--cpu") == 0) {} // parsed by parse_loc_opts // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        else { std::cerr << "Unknown option: " << argv[i] << std::endl; return opts; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    }
    return opts;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// GPU notice is now printed by nn_inference.cpp after ONNX Runtime loads // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

static void print_tracking_banner(const std::string& loc_csv, const std::string& output, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                                  int nb_frames, const freetrace::TrackingConfig& config) {
    std::cout << "FreeTrace C++ - Tracking" << std::endl;
    std::cout << "  Loc CSV:   " << loc_csv << std::endl;
    std::cout << "  Output:    " << output << std::endl;
    std::cout << "  Nb frames: " << nb_frames << std::endl;
    std::cout << "  Depth:     " << config.graph_depth << ", Cutoff: " << config.cutoff << std::endl;
    if (config.jump_threshold > 0)
        std::cout << "  Jump threshold: " << config.jump_threshold << " px" << std::endl;
    else
        std::cout << "  Jump threshold: auto (inferred from data)" << std::endl;
    std::cout << "  fBm mode: " << (config.fbm_mode ? "ON (NN inference + H-K output)" : "OFF (fixed alpha/K)") << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    if (config.post_process) std::cout << "  Post-processing: enabled" << std::endl;
    if (config.img_rows > 0) std::cout << "  Image: " << config.img_cols << "x" << config.img_rows << " (from TIFF)" << std::endl;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

static void print_usage() {
    std::cout << "FreeTrace C++\n"
              << "Usage:\n"
              << "  Full pipeline (localization + tracking):\n"
              << "    freetrace <input.tiff> <output_dir> [options]\n"
              << "\n"
              << "  Localization only:\n"
              << "    freetrace localize <input.tiff> <output_dir> [options]\n"
              << "\n"
              << "  Tracking only:\n"
              << "    freetrace track <loc.csv> <output_dir> <nb_frames> [options]\n"
              << "\n"
              << "  Localization options:\n"
              << "    --window N       Window size (default: 7)\n"
              << "    --threshold F    Detection threshold (default: 1.0)\n"
              << "    --shift N        Shift (default: 1)\n"
              << "    (GPU is used automatically if available)\n"
              << "\n"
              << "  Tracking options:\n"
              << "    --depth N        Graph depth (default: 3)\n"
              << "    --cutoff N       Cutoff (default: 3)\n"
              << "    --jump F         Maximum jump distance in px (default: auto)\n"
              << "    --tiff PATH      TIFF file for image dimensions (tracking-only mode)\n"
              << "    --no-fbm         Disable fBm mode (no NN, fixed alpha/K, no H-K output)\n"
              << "    --postprocess    Enable post-processing\n"
              << "    --quiet          Suppress status messages\n";
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// Build loc CSV path from tiff path and output dir (matches localization.cpp naming)
static std::string build_loc_csv_path(const std::string& tiff_path, const std::string& output_path) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    auto last_sep = tiff_path.find_last_of("/\\");
    std::string fname = (last_sep != std::string::npos) ? tiff_path.substr(last_sep + 1) : tiff_path;
    auto tif_pos = fname.find(".tif");
    if (tif_pos != std::string::npos) fname = fname.substr(0, tif_pos);
#ifdef _WIN32
    return output_path + "\\" + fname + "_loc.csv";
#else
    return output_path + "/" + fname + "_loc.csv";
#endif
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// Parse localization options from argv // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
static void parse_loc_opts(int argc, char* argv[], int start_idx, int& window, float& threshold, int& shift) {
    for (int i = start_idx; i < argc; i++) {
        if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc) window = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) threshold = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--shift") == 0 && i + 1 < argc) shift = std::stoi(argv[++i]);
        // Skip tracking options (parsed by parse_tracking_opts)
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

int main(int argc, char* argv[]) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage();
        return 0;
    }

    // ---- Mode 1: Tracking only ----
    // freetrace track <loc.csv> <output_dir> <nb_frames> [options]
    if (argc >= 5 && std::string(argv[1]) == "track") {
        std::string loc_csv = argv[2];
        std::string output = argv[3];
        int nb_frames = std::stoi(argv[4]);

        auto opts = parse_tracking_opts(argc, argv, 5);
        auto& config = opts.config;

        if (!config.tiff_path.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
            int tiff_frames, tiff_h, tiff_w;
            freetrace::read_tiff(config.tiff_path, tiff_frames, tiff_h, tiff_w);
            config.img_rows = tiff_h;
            config.img_cols = tiff_w;
            nb_frames = tiff_frames;
        }

        print_tracking_banner(loc_csv, output, nb_frames, config); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

        return freetrace::run_tracking(loc_csv, output, nb_frames, config) ? 0 : 1;
    }

    // ---- Mode 2: Localization only ---- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    // freetrace localize <input.tiff> <output_dir> [options]
    if (argc >= 4 && std::string(argv[1]) == "localize") {
        std::string input = argv[2];
        std::string output = argv[3];
        int window = 7; float threshold = 1.0f; int shift = 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        parse_loc_opts(argc, argv, 4, window, threshold, shift);

        std::cout << "FreeTrace C++ - Localization" << std::endl;
        std::cout << "  Input:  " << input << std::endl;
        std::cout << "  Output: " << output << std::endl;
        std::cout << "  Window: " << window << ", Threshold: " << threshold
                  << ", Shift: " << shift << std::endl;

        return freetrace::run(input, output, window, threshold, shift, /*verbose=*/true) ? 0 : 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

    // ---- Mode 3: Full pipeline (localization + tracking) ---- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    // freetrace <input.tiff> <output_dir> [options]
    if (argc >= 3 && std::string(argv[1]) != "track" && std::string(argv[1]) != "localize") {
        std::string input = argv[1];
        std::string output = argv[2];

        // Parse localization options
        int window = 7; float threshold = 1.0f; int shift = 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        parse_loc_opts(argc, argv, 3, window, threshold, shift);

        // Parse tracking options
        auto opts = parse_tracking_opts(argc, argv, 3);
        auto& config = opts.config;
        bool verbose = config.verbose;

        // --- Step 1: Localization ---
        std::cout << "FreeTrace C++ - Full Pipeline" << std::endl;
        std::cout << "  Input:  " << input << std::endl;
        std::cout << "  Output: " << output << std::endl;
        std::cout << "\n=== Step 1: Localization ===" << std::endl;
        std::cout << "  Window: " << window << ", Threshold: " << threshold
                  << ", Shift: " << shift << std::endl;

        bool lok = freetrace::run(input, output, window, threshold, shift, verbose); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        if (!lok) {
            std::cerr << "Localization failed." << std::endl;
            return 1;
        }

        // --- Step 2: Tracking ---
        // Get nb_frames and image dimensions from the TIFF
        int nb_frames, tiff_h, tiff_w;
        freetrace::read_tiff(input, nb_frames, tiff_h, tiff_w);
        config.tiff_path = input;
        config.img_rows = tiff_h;
        config.img_cols = tiff_w;

        // Build loc CSV path (matches localization output naming)
        std::string loc_csv = build_loc_csv_path(input, output);

        std::cout << "\n=== Step 2: Tracking ===" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        print_tracking_banner(loc_csv, output, nb_frames, config);

        return freetrace::run_tracking(loc_csv, output, nb_frames, config) ? 0 : 1;
    }

    // No valid arguments
    print_usage();
    return 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
}
