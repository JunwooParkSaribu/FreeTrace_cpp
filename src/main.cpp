#include <iostream> // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream> // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "image_pad.h"
#include "regression.h"
#include "cost_function.h"
#include "localization.h"
#include "tracking.h"
#include "gpu_module.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

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
        else if (std::strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) ++i; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        else if (std::strcmp(argv[i], "--cpu") == 0) {} // parsed by parse_loc_opts // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        else { std::cerr << "Unknown option: " << argv[i] << std::endl; return opts; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    }
    return opts;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// GPU notice is now printed by nn_inference.cpp after ONNX Runtime loads // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

static void print_tracking_banner(const std::string& loc_csv, const std::string& output, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                                  int nb_frames, const freetrace::TrackingConfig& config) {
    std::cout << "FreeTrace - Tracking" << std::endl;
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

static void print_usage() { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    std::cout << "FreeTrace v1.6.3.0\n" // Modified by Claude (claude-opus-4-7, Anthropic AI) - 2026-04-28
              << "Usage:\n"
              << "  Full pipeline (localization + tracking):\n"
              << "    freetrace <input> <output_dir> [options]\n"
              << "\n"
              << "  Batch mode (process all files in a folder):\n"
              << "    freetrace batch <input_folder> <output_dir> [options]\n"
              << "\n"
              << "  Localization only:\n"
              << "    freetrace localize <input> <output_dir> [options]\n"
              << "\n"
              << "  Tracking only:\n"
              << "    freetrace track <loc.csv> <output_dir> <nb_frames> [options]\n"
              << "\n"
              << "  Supported input formats: .tif, .tiff, .nd2\n"
              << "  Note: Old ND2 files (JPEG2000 format) are not supported.\n"
              << "        Use the Python version of FreeTrace for old ND2 files.\n"
              << "\n"
              << "  Localization options:\n"
              << "    --window N       Window size (default: 7)\n"
              << "    --threshold F    Detection threshold (default: 1.0)\n"
              << "    --shift N        Shift (default: 1)\n"
              << "    --batch-size N   Batch size (default: auto, computed from available memory)\n" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
              << "    (GPU is used automatically if available)\n"
              << "\n"
              << "  Tracking options:\n"
              << "    --depth N        Graph depth (default: 3)\n"
              << "    --cutoff N       Cutoff (default: 3)\n"
              << "    --jump F         Maximum jump distance in px (default: auto)\n"
              << "    --tiff PATH      TIFF/ND2 file for image dimensions (tracking-only mode)\n"
              << "    --no-fbm         Disable fBm mode (no NN, fixed alpha/K, no H-K output)\n"
              << "    --postprocess    Enable post-processing\n"
              << "    --quiet          Suppress status messages\n";
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

// Build loc CSV path from input path and output dir (matches localization.cpp naming)
static std::string build_loc_csv_path(const std::string& input_path, const std::string& output_path) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    auto last_sep = input_path.find_last_of("/\\");
    std::string fname = (last_sep != std::string::npos) ? input_path.substr(last_sep + 1) : input_path;
    // Strip known extensions: .tif, .tiff, .nd2
    auto tif_pos = fname.find(".tif");
    if (tif_pos != std::string::npos) fname = fname.substr(0, tif_pos);
    auto nd2_pos = fname.find(".nd2");
    if (nd2_pos != std::string::npos) fname = fname.substr(0, nd2_pos);
#ifdef _WIN32
    return output_path + "\\" + fname + "_loc.csv";
#else
    return output_path + "/" + fname + "_loc.csv";
#endif
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

// Parse localization options from argv // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
static void parse_loc_opts(int argc, char* argv[], int start_idx, int& window, float& threshold, int& shift, int& batch_size) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    for (int i = start_idx; i < argc; i++) {
        if (std::strcmp(argv[i], "--window") == 0 && i + 1 < argc) window = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) threshold = std::stof(argv[++i]);
        else if (std::strcmp(argv[i], "--shift") == 0 && i + 1 < argc) shift = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        // Skip tracking options (parsed by parse_tracking_opts)
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

int main(int argc, char* argv[]) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    // Force line-by-line flushing so GUI receives progress updates in real time  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
#ifndef _WIN32
    // setvbuf with _IOLBF + size 0 crashes on MSVC (allocates 0-byte buffer).
    // On Windows, std::unitbuf alone is sufficient.
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IOLBF, 0);
#endif

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
            freetrace::read_image(config.tiff_path, tiff_frames, tiff_h, tiff_w);
            config.img_rows = tiff_h;
            config.img_cols = tiff_w;
            nb_frames = tiff_frames;
        }

        print_tracking_banner(loc_csv, output, nb_frames, config); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

        return freetrace::run_tracking(loc_csv, output, nb_frames, config) ? 0 : 1;
    }

    // ---- Mode 2: Batch mode ---- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    // freetrace batch <input_folder> <output_dir> [options]
    if (argc >= 4 && std::string(argv[1]) == "batch") {
        std::string input_folder = argv[2];
        std::string output = argv[3];

        // Parse options
        int window = 7; float threshold = 1.0f; int shift = 1; int batch_sz = 100; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        parse_loc_opts(argc, argv, 4, window, threshold, shift, batch_sz);
        auto opts = parse_tracking_opts(argc, argv, 4);
        auto& config = opts.config;
        bool verbose = config.verbose;

        // Collect supported files from input folder
        std::vector<std::string> files;
        if (!std::filesystem::is_directory(input_folder)) {
            std::cerr << "Not a directory: " << input_folder << std::endl;
            return 1;
        }
        for (auto& entry : std::filesystem::directory_iterator(input_folder)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            if (ext == ".tif" || ext == ".tiff" || ext == ".nd2") {
                files.push_back(entry.path().string());
            } else {
                std::cerr << "  Skipping unsupported file: " << entry.path().filename().string() << std::endl;
            }
        }
        std::sort(files.begin(), files.end());

        if (files.empty()) {
            std::cerr << "No supported files (.tif, .tiff, .nd2) found in: " << input_folder << std::endl;
            return 1;
        }

        try { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-23
            std::filesystem::create_directories(output);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "ERROR: Cannot create output directory '" << output << "': " << e.what() << std::endl;
            return 1;
        } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-23
        std::cout << "FreeTrace - Batch Mode" << std::endl;
        std::cout << "  Input folder: " << input_folder << std::endl;
        std::cout << "  Output:       " << output << std::endl;
        std::cout << "  Files found:  " << files.size() << std::endl;
        std::cout << "  Window: " << window << ", Threshold: " << threshold
                  << ", Shift: " << shift << std::endl;
        std::cout << std::endl;

        // Error log for batch mode  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
        std::string error_log_path = (std::filesystem::path(output) / "error_log.txt").string();
        std::ofstream error_log;
        auto log_error = [&](const std::string& fname, const std::string& stage, const std::string& msg) {
            if (!error_log.is_open()) {
                error_log.open(error_log_path, std::ios::app);
                error_log << "FreeTrace Batch — Error Log" << std::endl;
                error_log << "Input folder: " << input_folder << std::endl;
                error_log << "Output:       " << output << std::endl;
                error_log << std::string(60, '-') << std::endl;
            }
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            std::ostringstream ts;
            ts << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
            error_log << "[" << ts.str() << "] " << fname << " — " << stage << ": " << msg << std::endl;
            error_log.flush();
        };  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19

        int success_count = 0;
        for (size_t idx = 0; idx < files.size(); ++idx) {
            const auto& input = files[idx];
            auto fname = std::filesystem::path(input).filename().string();
            std::cerr << "PROGRESS_BATCH:" << idx << ":" << files.size() << ":" << fname << std::endl;  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
            std::cout << "=== [" << (idx + 1) << "/" << files.size() << "] " << fname << " ===" << std::endl;

            try {  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
                // Step 1: Localization
                bool lok = freetrace::run(input, output, window, threshold, shift, verbose, "", batch_sz);
                if (!lok) {
                    std::cerr << "  Localization FAILED for " << fname << ", skipping." << std::endl;
                    log_error(fname, "Localization", "returned false");
                    std::cout << std::endl;
                    continue;
                }

                // Step 2: Tracking
                int nb_frames, img_h, img_w;
                freetrace::read_image(input, nb_frames, img_h, img_w);
                auto track_config = config;
                track_config.tiff_path = input;
                track_config.img_rows = img_h;
                track_config.img_cols = img_w;

                std::string loc_csv = build_loc_csv_path(input, output);
                bool tok = freetrace::run_tracking(loc_csv, output, nb_frames, track_config);
                if (!tok) {
                    std::cerr << "  Tracking FAILED for " << fname << "." << std::endl;
                    log_error(fname, "Tracking", "returned false");
                } else {
                    success_count++;
                }
            } catch (const std::exception& e) {
                std::cerr << "  ERROR for " << fname << ": " << e.what() << std::endl;
                log_error(fname, "Exception", e.what());
            } catch (...) {
                std::cerr << "  UNKNOWN ERROR for " << fname << std::endl;
                log_error(fname, "Exception", "unknown error (non-std::exception)");
            }  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-19
            std::cout << std::endl;
        }

        if (error_log.is_open()) {
            error_log.close();
            std::cout << "Error log written to: " << error_log_path << std::endl;
        }
        std::cout << "=== Batch complete: " << success_count << "/" << files.size() << " succeeded ===" << std::endl;
        return (success_count == static_cast<int>(files.size())) ? 0 : 1;
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

    // ---- Mode 3: Localization only ---- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    // freetrace localize <input> <output_dir> [options]
    if (argc >= 4 && std::string(argv[1]) == "localize") {
        std::string input = argv[2];
        std::string output = argv[3];
        int window = 7; float threshold = 1.0f; int shift = 1; int batch_sz = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        parse_loc_opts(argc, argv, 4, window, threshold, shift, batch_sz);

        std::cout << "FreeTrace - Localization" << std::endl;
        std::cout << "  Input:  " << input << std::endl;
        std::cout << "  Output: " << output << std::endl;
        std::cout << "  Window: " << window << ", Threshold: " << threshold
                  << ", Shift: " << shift << std::endl;

        return freetrace::run(input, output, window, threshold, shift, /*verbose=*/true, "", batch_sz) ? 0 : 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

    // ---- Mode 3: Full pipeline (localization + tracking) ---- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    // freetrace <input.tiff> <output_dir> [options]
    if (argc >= 3 && std::string(argv[1]) != "track" && std::string(argv[1]) != "localize" && std::string(argv[1]) != "batch") { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        std::string input = argv[1];
        std::string output = argv[2];

        // Parse localization options
        int window = 7; float threshold = 1.0f; int shift = 1; int batch_sz = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        parse_loc_opts(argc, argv, 3, window, threshold, shift, batch_sz);

        // Parse tracking options
        auto opts = parse_tracking_opts(argc, argv, 3);
        auto& config = opts.config;
        bool verbose = config.verbose;

        // --- Step 1: Localization ---
        std::cout << "FreeTrace - Full Pipeline" << std::endl;
        std::cout << "  Input:  " << input << std::endl;
        std::cout << "  Output: " << output << std::endl;
        std::cout << "\n=== Step 1: Localization ===" << std::endl;
        std::cout << "  Window: " << window << ", Threshold: " << threshold
                  << ", Shift: " << shift << std::endl;

        bool lok = freetrace::run(input, output, window, threshold, shift, verbose, "", batch_sz); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        if (!lok) {
            std::cerr << "Localization failed." << std::endl;
            return 1;
        }

        // --- Step 2: Tracking ---
        // Get nb_frames and image dimensions from the TIFF
        int nb_frames, tiff_h, tiff_w;
        freetrace::read_image(input, nb_frames, tiff_h, tiff_w);
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
