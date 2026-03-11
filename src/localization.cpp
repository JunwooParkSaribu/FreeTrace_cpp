#include "localization.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:30
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <random>

// OpenCV for TIFF I/O (optional — can be replaced with libtiff)
#ifdef USE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#endif

namespace freetrace {

// ============================================================
// Image I/O
// ============================================================

#ifdef USE_OPENCV
std::vector<float> read_tiff(const std::string& path, int& nb_frames, int& height, int& width) {
    std::vector<cv::Mat> pages;
    cv::imreadmulti(path, pages, cv::IMREAD_UNCHANGED);
    nb_frames = static_cast<int>(pages.size());
    if (nb_frames == 0) return {};
    height = pages[0].rows;
    width = pages[0].cols;

    std::vector<float> data(nb_frames * height * width);
    float global_max = 0.0f;
    for (int n = 0; n < nb_frames; ++n) {
        cv::Mat f;
        pages[n].convertTo(f, CV_32F);
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c) {
                float v = f.at<float>(r, c);
                data[n * height * width + r * width + c] = v;
                global_max = std::max(global_max, v);
            }
    }
    // Normalize to [0, 1]
    if (global_max > 0.0f)
        for (auto& v : data) v /= global_max;
    return data;
}
#else
std::vector<float> read_tiff(const std::string& path, int& nb_frames, int& height, int& width) {
    std::cerr << "TIFF reading requires OpenCV (compile with -DUSE_OPENCV). "
              << "Returning empty." << std::endl;
    nb_frames = height = width = 0;
    return {};
}
#endif

// ============================================================
// CSV output
// ============================================================

void write_localization_csv(
    const std::string& output_path,
    const LocalizationResult& result
) {
    std::ofstream ofs(output_path + "_loc.csv");
    ofs << "frame,x,y,z,xvar,yvar,rho,norm_cst,intensity,window_size\n";
    for (int frame = 0; frame < static_cast<int>(result.coords.size()); ++frame) {
        for (int p = 0; p < static_cast<int>(result.coords[frame].size()); ++p) {
            auto& pos = result.coords[frame][p];
            auto& info = result.infos[frame][p];
            auto& pdf = result.pdfs[frame][p];
            int ws = static_cast<int>(std::sqrt(static_cast<float>(pdf.size())));
            float intensity = (pdf.size() > 0) ? pdf[pdf.size() / 2] : 0.0f;
            // Note: x/y swap as in Python version
            ofs << (frame + 1) << ","
                << pos[1] << "," << pos[0] << "," << pos[2] << ","
                << info[0] << "," << info[1] << "," << info[2] << ","
                << info[3] << "," << intensity << "," << ws << "\n";
        }
    }
}

// ============================================================
// Background estimation
// ============================================================

BackgroundResult compute_background(
    const std::vector<float>& imgs, int nb_imgs, int rows, int cols,
    const std::vector<WinParams>& window_sizes, float alpha
) {
    // Normalize images per frame
    std::vector<float> norm_imgs(imgs.size());
    for (int n = 0; n < nb_imgs; ++n) {
        float fmax = 0.0f;
        int base = n * rows * cols;
        for (int i = 0; i < rows * cols; ++i)
            fmax = std::max(fmax, imgs[base + i]);
        if (fmax > 0.0f)
            for (int i = 0; i < rows * cols; ++i)
                norm_imgs[base + i] = imgs[base + i] / fmax;
    }

    const float bins = 0.01f;
    std::vector<float> bg_means(nb_imgs, 0.0f);
    std::vector<float> bg_stds(nb_imgs, 0.0f);

    for (int n = 0; n < nb_imgs; ++n) {
        int base = n * rows * cols;
        int pixel_count = rows * cols;

        // Quantize to uint8/100
        std::vector<float> bg_int(pixel_count);
        for (int i = 0; i < pixel_count; ++i)
            bg_int[i] = static_cast<int>(norm_imgs[base + i] * 100) / 100.0f;

        std::vector<int> args(pixel_count);
        std::iota(args.begin(), args.end(), 0);
        std::vector<int> post_mask = args;

        // Iterative background estimation (3 iterations)
        float mode_val = 0.0f, mask_std = 0.0f;
        for (int iter = 0; iter < 3; ++iter) {
            if (post_mask.empty()) break;
            // Find max value in masked data
            float max_val = 0.0f;
            for (int idx : post_mask)
                max_val = std::max(max_val, bg_int[idx]);

            int nb_bins = static_cast<int>((max_val + bins) / bins);
            if (nb_bins < 1) break;
            std::vector<int> hist(nb_bins, 0);
            for (int idx : post_mask) {
                int bin = std::min(static_cast<int>(bg_int[idx] / bins), nb_bins - 1);
                hist[bin]++;
            }
            int mode_bin = std::distance(hist.begin(), std::max_element(hist.begin(), hist.end()));
            mode_val = mode_bin * bins + bins / 2.0f;

            // Compute std
            float sum = 0.0f, sum2 = 0.0f;
            for (int idx : post_mask) { sum += bg_int[idx]; sum2 += bg_int[idx] * bg_int[idx]; }
            float mean_tmp = sum / post_mask.size();
            mask_std = std::sqrt(sum2 / post_mask.size() - mean_tmp * mean_tmp);

            // Filter
            std::vector<int> new_mask;
            float lo = mode_val - 3.0f * mask_std;
            float hi = mode_val + 3.0f * mask_std;
            for (int idx : args)
                if (bg_int[idx] > lo && bg_int[idx] < hi)
                    new_mask.push_back(idx);
            post_mask = new_mask;
        }

        // Compute final mean/std
        if (!post_mask.empty()) {
            float sum = 0.0f, sum2 = 0.0f;
            for (int idx : post_mask) { sum += bg_int[idx]; sum2 += bg_int[idx] * bg_int[idx]; }
            bg_means[n] = sum / post_mask.size();
            bg_stds[n] = std::sqrt(sum2 / post_mask.size() - bg_means[n] * bg_means[n]);
        }
    }

    // Build background arrays per window size
    BackgroundResult result;
    for (auto& ws : window_sizes) {
        int area = ws.w * ws.h;
        std::vector<float> bg(nb_imgs * area);
        for (int n = 0; n < nb_imgs; ++n)
            for (int i = 0; i < area; ++i)
                bg[n * area + i] = bg_means[n];
        result.bgs[ws.w] = std::move(bg);
    }

    // Thresholds
    result.thresholds.resize(nb_imgs);
    for (int n = 0; n < nb_imgs; ++n) {
        float t = (bg_stds[n] > 0) ? 1.0f / (bg_means[n] * bg_means[n] / (bg_stds[n] * bg_stds[n])) * 2.0f : 1.0f;
        t = std::max(t, 1.0f);
        if (std::isnan(t)) t = 1.0f;
        result.thresholds[n] = t * alpha;
    }

    return result;
}

// ============================================================
// Gaussian PSF
// ============================================================

Image2D gauss_psf(int win_w, int win_h, float radius) {
    Image2D grid(win_h, win_w);
    float cx = win_w / 2.0f;
    float cy = win_h / 2.0f;
    float norm = std::sqrt(M_PI) * radius;

    for (int r = 0; r < win_h; ++r) {
        for (int c = 0; c < win_w; ++c) {
            float dx = (c + 0.5f) - cx;
            float dy = (r + 0.5f) - cy;
            grid.at(r, c) = std::exp(-(dx * dx + dy * dy) / (2.0f * radius * radius)) / norm;
        }
    }
    return grid;
}

// ============================================================
// Region max filter (simplified single-window version)
// ============================================================

std::vector<DetIndex> region_max_filter2(
    std::vector<float>& maps, int nb_imgs, int rows, int cols,
    const WinParams& window_size, const std::vector<float>& thresholds,
    int detect_range
) {
    std::vector<DetIndex> indices;
    int r_start = (detect_range == 0) ? window_size.h / 2 : detect_range;
    int c_start = (detect_range == 0) ? window_size.w / 2 : detect_range;

    for (int pass = 0; pass < 2; ++pass) {
        for (int n = 0; n < nb_imgs; ++n) {
            float thresh = thresholds[n];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    int idx = n * rows * cols + r * cols + c;
                    if (maps[idx] <= thresh) { maps[idx] = 0.0f; continue; }

                    // Check if local maximum
                    int r0 = std::max(0, r - r_start);
                    int r1 = std::min(rows, r + r_start + 1);
                    int c0 = std::max(0, c - c_start);
                    int c1 = std::min(cols, c + c_start + 1);

                    float local_max = 0.0f;
                    for (int ri = r0; ri < r1; ++ri)
                        for (int ci = c0; ci < c1; ++ci)
                            local_max = std::max(local_max, maps[n * rows * cols + ri * cols + ci]);

                    if (maps[idx] == local_max && maps[idx] != 0.0f) {
                        indices.push_back({n, r, c});
                        // Zero out neighborhood
                        for (int ri = r0; ri < r1; ++ri)
                            for (int ci = c0; ci < c1; ++ci)
                                maps[n * rows * cols + ri * cols + ci] = 0.0f;
                    }
                }
            }
        }
    }
    return indices;
}

// ============================================================
// Parameter generation (same as Python params_gen)
// ============================================================

static void params_gen(int win_s,
    std::vector<WinParams>& single_ws, std::vector<float>& single_rad,
    std::vector<WinParams>& multi_ws, std::vector<float>& multi_rad
) {
    if (win_s < 5) win_s = 5;
    if (win_s % 2 == 0) win_s += 1;

    single_ws = {{win_s, win_s}};
    multi_ws = {{win_s - 2, win_s - 2}, {win_s, win_s}, {win_s + 2, win_s + 2}};
    single_rad.clear();
    for (auto& ws : single_ws) single_rad.push_back((ws.w / 2) / 2.0f);
    multi_rad.clear();
    for (auto& ws : multi_ws) multi_rad.push_back((ws.w / 2) / 2.0f);
}

// ============================================================
// Top-level localization run
// ============================================================

bool run(const std::string& input_video_path,
         const std::string& output_path,
         int window_size,
         float threshold,
         int shift,
         bool verbose)
{
    int nb_frames, height, width;
    auto images = read_tiff(input_video_path, nb_frames, height, width);
    if (images.empty()) {
        std::cerr << "Failed to read: " << input_video_path << std::endl;
        return false;
    }

    if (verbose)
        std::cout << "Loaded " << nb_frames << " frames (" << height << "x" << width << ")" << std::endl;

    // Generate window params
    std::vector<WinParams> single_ws, multi_ws;
    std::vector<float> single_rad, multi_rad;
    params_gen(window_size, single_ws, single_rad, multi_ws, multi_rad);

    // Compute background
    std::vector<WinParams> all_ws = single_ws;
    all_ws.insert(all_ws.end(), multi_ws.begin(), multi_ws.end());
    auto bg_result = compute_background(images, nb_frames, height, width, all_ws, threshold);

    if (verbose)
        std::cout << "Background computed. Starting localization..." << std::endl;

    // Generate Gaussian PSFs
    std::vector<Image2D> forward_grids, backward_grids;
    for (size_t i = 0; i < single_ws.size(); ++i)
        forward_grids.push_back(gauss_psf(single_ws[i].w, single_ws[i].h, single_rad[i]));
    for (size_t i = 0; i < multi_ws.size(); ++i)
        backward_grids.push_back(gauss_psf(multi_ws[i].w, multi_ws[i].h, multi_rad[i]));

    // --- Simplified localization pipeline ---
    // Full pipeline (forward detection + backward refinement) would follow
    // the Python Localization.localization() function.
    // For now: forward pass only with single window size.

    int extend = multi_ws.back().w * 4;
    int ext_rows = height + extend;
    int ext_cols = width + extend;
    int half_ext = extend / 2;

    // Create extended images with zero padding
    std::vector<float> ext_imgs(nb_frames * ext_rows * ext_cols, 0.0f);
    for (int n = 0; n < nb_frames; ++n)
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c)
                ext_imgs[n * ext_rows * ext_cols + (r + half_ext) * ext_cols + (c + half_ext)] =
                    images[n * height * width + r * width + c];

    // Likelihood computation for each window size
    auto& ws0 = single_ws[0];
    auto& grid0 = forward_grids[0];
    auto& bg_means_vec = bg_result.bgs[ws0.w];

    // Extract per-frame bg means
    std::vector<float> bg_means(nb_frames);
    for (int n = 0; n < nb_frames; ++n)
        bg_means[n] = bg_means_vec[n * ws0.w * ws0.h]; // all values same per frame

    // Crop images
    int nb_crops = 0;
    auto crop_imgs = image_cropping(ext_imgs, nb_frames, ext_rows, ext_cols,
                                     extend, ws0.w, ws0.h, shift, nb_crops);

    // Compute bg_squared_sums
    std::vector<float> bg_sq_sums(nb_frames);
    for (int n = 0; n < nb_frames; ++n)
        bg_sq_sums[n] = ws0.w * ws0.h * bg_means[n] * bg_means[n];

    // Likelihood (simplified — uses the already-ported image_pad functions)
    // For full implementation, the likelihood() function from image_pad needs to be ported
    // This is a placeholder for the detection pipeline

    if (verbose)
        std::cout << "Localization pipeline (C++ skeleton) completed." << std::endl;

    // Placeholder result
    LocalizationResult result;
    result.coords.resize(nb_frames);
    result.pdfs.resize(nb_frames);
    result.infos.resize(nb_frames);

    // Write output
    std::string loc_output = output_path + "/" +
        input_video_path.substr(input_video_path.find_last_of('/') + 1);
    loc_output = loc_output.substr(0, loc_output.find(".tif"));
    write_localization_csv(loc_output, result);

    if (verbose)
        std::cout << "Localization written to " << loc_output << "_loc.csv" << std::endl;

    return true;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:30

} // namespace freetrace
