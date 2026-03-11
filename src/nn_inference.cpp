// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include "nn_inference.h"

#ifdef USE_ONNXRUNTIME // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <onnxruntime_cxx_api.h>
#endif

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace freetrace { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Preprocessing helpers (matching Python's RegModel methods)
// ============================================================

// displacement: step distances between consecutive points // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static std::vector<float> displacement(const std::vector<float>& xs, const std::vector<float>& ys) {
    std::vector<float> disps; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 1; i < (int)xs.size(); i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float dx = xs[i] - xs[i-1], dy = ys[i] - ys[i-1]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        disps.push_back(std::sqrt(dx*dx + dy*dy)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
    return disps; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// radius: distance from first point for each position // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static std::vector<float> radius(const std::vector<float>& xs, const std::vector<float>& ys) {
    std::vector<float> rads = {0.0f}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 1; i < (int)xs.size(); i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float dx = xs[i] - xs[0], dy = ys[i] - ys[0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        rads.push_back(std::sqrt(dx*dx + dy*dy)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
    return rads; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// abs_subtraction: |x[i] - x[i-1]| // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static std::vector<float> abs_subtraction(const std::vector<float>& xs) {
    std::vector<float> result = {0.0f}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 1; i < (int)xs.size(); i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        result.push_back(std::abs(xs[i] - xs[i-1])); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
    return result; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

static float vec_std(const std::vector<float>& v) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float mean = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (float x : v) mean += x; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    mean /= v.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float var = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (float x : v) var += (x - mean) * (x - mean); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return std::sqrt(var / v.size()); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

static float vec_mean(const std::vector<float>& v) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float s = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (float x : v) s += x; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return s / v.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// IQR mean: mean of values between Q1 and Q3 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static float iqr_mean(std::vector<float> v) {
    if (v.size() <= 4) return vec_mean(v); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::sort(v.begin(), v.end()); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int n = (int)v.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // numpy quantile with 'normal_unbiased' method // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float q25_idx = 0.25f * (n - 1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float q75_idx = 0.75f * (n - 1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int lo = (int)q25_idx, hi = (int)q75_idx; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float q25 = v[lo] + (q25_idx - lo) * (v[lo+1] - v[lo]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float q75 = v[hi] + (q75_idx - hi) * (v[std::min(hi+1, n-1)] - v[hi]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return (q25 + q75) / 2.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// cvt_2_signal: convert x,y window to 3-feature signal // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Returns feature vector of shape (model_num, 1, 3) flattened
static void cvt_2_signal(const std::vector<float>& x_in, const std::vector<float>& y_in, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                          std::vector<float>& out_x_signal, std::vector<float>& out_y_signal) {
    int n = (int)x_in.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto rads = radius(x_in, y_in); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto disps = displacement(x_in, y_in); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float mean_disp = vec_mean(disps); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (mean_disp < 1e-10f) mean_disp = 1e-10f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Normalized radius and displacement // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> rad_norm(n), xs_raw(n), ys_raw(n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        rad_norm[i] = rads[i] / mean_disp / n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        xs_raw[i] = (x_in[i] - x_in[0]) / mean_disp / n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        ys_raw[i] = (y_in[i] - y_in[0]) / mean_disp / n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }

    // Normalized cumulative abs subtraction for x // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float std_x = vec_std(x_in); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (std_x < 1e-10f) std_x = 1e-10f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> x_norm(n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i < n; i++) x_norm[i] = x_in[i] / std_x; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto abs_x = abs_subtraction(x_norm); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> cumx(n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    cumx[0] = abs_x[0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 1; i < n; i++) cumx[i] = cumx[i-1] + abs_x[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i < n; i++) cumx[i] /= n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Same for y // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float std_y = vec_std(y_in); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (std_y < 1e-10f) std_y = 1e-10f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> y_norm(n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i < n; i++) y_norm[i] = y_in[i] / std_y; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto abs_y = abs_subtraction(y_norm); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> cumy(n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    cumy[0] = abs_y[0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 1; i < n; i++) cumy[i] = cumy[i-1] + abs_y[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i < n; i++) cumy[i] /= n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Output: x_signal = transpose([cumx, rad_norm, xs_raw]) -> shape (n, 3) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    out_x_signal.resize(n * 3); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    out_y_signal.resize(n * 3); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        out_x_signal[i * 3 + 0] = cumx[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        out_x_signal[i * 3 + 1] = rad_norm[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        out_x_signal[i * 3 + 2] = xs_raw[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        out_y_signal[i * 3 + 0] = cumy[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        out_y_signal[i * 3 + 1] = rad_norm[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        out_y_signal[i * 3 + 2] = ys_raw[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
}

// model_selection: choose which alpha model to use based on trajectory length // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static int model_selection(const NNModels& models, int length) {
    for (int i = 0; i < (int)models.crits.size() - 1; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (models.crits[i] <= length && length < models.crits[i+1]) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            return models.reg_model_nums[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
    }
    // Fallback: second-to-last model // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return models.reg_model_nums[std::max(0, (int)models.reg_model_nums.size() - 2)]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// ONNX Runtime implementation
// ============================================================

#ifdef USE_ONNXRUNTIME // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

bool load_nn_models(NNModels& models, const std::string& models_dir) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    try {
        auto* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "freetrace"); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        models.env = env; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        Ort::SessionOptions opts; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        opts.SetIntraOpNumThreads(1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        models.reg_model_nums = {3, 5, 8}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        models.crits = {3, 5, 8, 8192}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Load alpha models // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int n : models.reg_model_nums) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            std::string path = models_dir + "/reg_model_" + std::to_string(n) + ".onnx"; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto* session = new Ort::Session(*env, path.c_str(), opts); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            models.alpha_sessions[n] = session; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }

        // Load k model // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::string k_path = models_dir + "/reg_k_model.onnx"; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        models.k_session = new Ort::Session(*env, k_path.c_str(), opts); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        models.loaded = true; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        return true; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    } catch (const Ort::Exception& e) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::cerr << "ONNX load error: " << e.what() << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        return false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
}

void free_nn_models(NNModels& models) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (auto& [n, session] : models.alpha_sessions) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        delete static_cast<Ort::Session*>(session); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
    models.alpha_sessions.clear(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (models.k_session) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        delete static_cast<Ort::Session*>(models.k_session); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        models.k_session = nullptr; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
    if (models.env) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        delete static_cast<Ort::Env*>(models.env); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        models.env = nullptr; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
    models.loaded = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

float predict_alpha_nn(const NNModels& models, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                       const std::vector<float>& xs,
                       const std::vector<float>& ys) {
    if (!models.loaded || xs.size() < 3) return 1.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    int n = (int)xs.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int model_num = model_selection(models, n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    auto it = models.alpha_sessions.find(model_num); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (it == models.alpha_sessions.end()) return 1.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto* session = static_cast<Ort::Session*>(it->second); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Recoupe trajectory into sliding windows of size model_num // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<std::vector<float>> windows_x, windows_y; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i + model_num <= n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        windows_x.push_back(std::vector<float>(xs.begin() + i, xs.begin() + i + model_num)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        windows_y.push_back(std::vector<float>(ys.begin() + i, ys.begin() + i + model_num)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
    if (windows_x.empty()) return 1.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Convert each window to signals and build input tensor // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // Each window produces 2 signals (x and y), shape (model_num, 1, 3) each
    // Total batch: 2 * windows_x.size(), shape: (batch, model_num, 1, 3)
    int batch_size = 2 * (int)windows_x.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> input_data(batch_size * model_num * 1 * 3); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    for (int w = 0; w < (int)windows_x.size(); w++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<float> x_sig, y_sig; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        cvt_2_signal(windows_x[w], windows_y[w], x_sig, y_sig); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // x signal -> batch index 2*w, shape (model_num, 1, 3) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int offset_x = (2 * w) * model_num * 3; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int i = 0; i < model_num * 3; i++) input_data[offset_x + i] = x_sig[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // y signal -> batch index 2*w+1 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int offset_y = (2 * w + 1) * model_num * 3; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int i = 0; i < model_num * 3; i++) input_data[offset_y + i] = y_sig[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }

    try { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<int64_t> shape = {batch_size, model_num, 1, 3}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                                                   input_data.size(), shape.data(), shape.size());

        // Get actual I/O names from session // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Ort::AllocatorWithDefaultOptions alloc; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto in_name = session->GetInputNameAllocated(0, alloc); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto out_name = session->GetOutputNameAllocated(0, alloc); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        const char* input_names[] = {in_name.get()}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        const char* output_names[] = {out_name.get()}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        float* pred_data = outputs[0].GetTensorMutableData<float>(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<float> preds(pred_data, pred_data + batch_size); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // IQR mean if batch > 4, else simple mean (matching Python) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        return iqr_mean(preds); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    } catch (const Ort::Exception& e) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::cerr << "Alpha predict error: " << e.what() << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        return 1.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
}

float predict_k_nn(const NNModels& models, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                   const std::vector<float>& xs,
                   const std::vector<float>& ys) {
    if (!models.loaded || xs.size() < 2) return 0.5f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    auto* session = static_cast<Ort::Session*>(models.k_session); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (!session) return 0.5f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Compute log10(mean displacement) or log10(IQR mean displacement) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto disps = displacement(xs, ys); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (disps.empty()) return 0.5f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    float log_disp; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if ((int)xs.size() < 10) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        log_disp = std::log10(vec_mean(disps)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    } else {
        log_disp = std::log10(iqr_mean(disps)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
    if (std::isnan(log_disp) || std::isinf(log_disp)) return 0.5f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    try { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<float> input_data = {log_disp}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<int64_t> shape = {1, 1}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), 1, shape.data(), 2); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Get actual I/O names from session // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Ort::AllocatorWithDefaultOptions alloc; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto in_name = session->GetInputNameAllocated(0, alloc); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto out_name = session->GetOutputNameAllocated(0, alloc); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        const char* input_names[] = {in_name.get()}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        const char* output_names[] = {out_name.get()}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        float k = outputs[0].GetTensorMutableData<float>()[0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (std::isnan(k)) return 1.0f; // NaN -> 1.0 (matching Python) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        return k; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    } catch (const Ort::Exception& e) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::cerr << "K predict error: " << e.what() << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        return 0.5f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
}

#else // No ONNX Runtime // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

bool load_nn_models(NNModels& /*models*/, const std::string& /*models_dir*/) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

void free_nn_models(NNModels& /*models*/) {} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

float predict_alpha_nn(const NNModels& /*models*/, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                       const std::vector<float>& /*xs*/,
                       const std::vector<float>& /*ys*/) {
    return 1.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

float predict_k_nn(const NNModels& /*models*/, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                   const std::vector<float>& /*xs*/,
                   const std::vector<float>& /*ys*/) {
    return 0.5f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

#endif // USE_ONNXRUNTIME // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
