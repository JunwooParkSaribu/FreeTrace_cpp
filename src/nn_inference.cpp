// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
#include "nn_inference.h"

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace freetrace {

// ============================================================
// Preprocessing helpers — double precision to match Python/NumPy
// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

static std::vector<double> displacement_d(const std::vector<double>& xs, const std::vector<double>& ys) {
    std::vector<double> disps;
    for (int i = 1; i < (int)xs.size(); i++) {
        double dx = xs[i] - xs[i-1], dy = ys[i] - ys[i-1];
        disps.push_back(std::sqrt(dx*dx + dy*dy));
    }
    return disps;
}

static std::vector<double> radius_d(const std::vector<double>& xs, const std::vector<double>& ys) {
    std::vector<double> rads = {0.0};
    for (int i = 1; i < (int)xs.size(); i++) {
        double dx = xs[i] - xs[0], dy = ys[i] - ys[0];
        rads.push_back(std::sqrt(dx*dx + dy*dy));
    }
    return rads;
}

static std::vector<double> abs_subtraction_d(const std::vector<double>& xs) {
    std::vector<double> result = {0.0};
    for (int i = 1; i < (int)xs.size(); i++) {
        result.push_back(std::abs(xs[i] - xs[i-1]));
    }
    return result;
}

static double vec_std_d(const std::vector<double>& v) {
    double mean = 0;
    for (double x : v) mean += x;
    mean /= v.size();
    double var = 0;
    for (double x : v) var += (x - mean) * (x - mean);
    return std::sqrt(var / v.size());
}

static double vec_mean_d(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s / v.size();
}

// numpy quantile with 'normal_unbiased' method (alpha=3/8, beta=3/8)
static double quantile_normal_unbiased_d(const std::vector<double>& sorted_v, double q) {
    int n = (int)sorted_v.size();
    double idx = 0.375 + q * ((double)n + 0.25) - 1.0;
    if (idx <= 0.0) return sorted_v[0];
    if (idx >= n - 1.0) return sorted_v[n - 1];
    int lo = (int)idx;
    double frac = idx - lo;
    return sorted_v[lo] + frac * (sorted_v[lo + 1] - sorted_v[lo]);
}

// IQR mean: mean of Q25 and Q75 (matching Python np.mean(np.quantile(..., q=[0.25, 0.75])))
static double iqr_mean_d(std::vector<double> v) {
    if (v.size() <= 4) return vec_mean_d(v);
    std::sort(v.begin(), v.end());
    double q25 = quantile_normal_unbiased_d(v, 0.25);
    double q75 = quantile_normal_unbiased_d(v, 0.75);
    return (q25 + q75) / 2.0;
}

// float versions that delegate to double
static std::vector<float> displacement(const std::vector<float>& xs, const std::vector<float>& ys) {
    std::vector<double> xd(xs.begin(), xs.end()), yd(ys.begin(), ys.end());
    auto dd = displacement_d(xd, yd);
    return std::vector<float>(dd.begin(), dd.end());
}

static float vec_mean(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += x;
    return (float)(s / v.size());
}

static float iqr_mean(std::vector<float> v) {
    std::vector<double> vd(v.begin(), v.end());
    return (float)iqr_mean_d(vd);
}

// cvt_2_signal: double-precision preprocessing, output as float for model input
static void cvt_2_signal(const std::vector<double>& x_in, const std::vector<double>& y_in,
                          std::vector<float>& out_x_signal, std::vector<float>& out_y_signal) {
    int n = (int)x_in.size();
    auto rads = radius_d(x_in, y_in);
    auto disps = displacement_d(x_in, y_in);
    double mean_disp = vec_mean_d(disps);
    if (mean_disp < 1e-10) mean_disp = 1e-10;

    // Normalized radius and position delta
    std::vector<double> rad_norm(n), xs_raw(n), ys_raw(n);
    for (int i = 0; i < n; i++) {
        rad_norm[i] = rads[i] / mean_disp / n;
        xs_raw[i] = (x_in[i] - x_in[0]) / mean_disp / n;
        ys_raw[i] = (y_in[i] - y_in[0]) / mean_disp / n;
    }

    // Normalized cumulative abs subtraction for x
    double std_x = vec_std_d(x_in);
    if (std_x < 1e-10) std_x = 1e-10;
    std::vector<double> x_norm(n);
    for (int i = 0; i < n; i++) x_norm[i] = x_in[i] / std_x;
    auto abs_x = abs_subtraction_d(x_norm);
    std::vector<double> cumx(n);
    cumx[0] = abs_x[0];
    for (int i = 1; i < n; i++) cumx[i] = cumx[i-1] + abs_x[i];
    for (int i = 0; i < n; i++) cumx[i] /= n;

    // Same for y
    double std_y = vec_std_d(y_in);
    if (std_y < 1e-10) std_y = 1e-10;
    std::vector<double> y_norm(n);
    for (int i = 0; i < n; i++) y_norm[i] = y_in[i] / std_y;
    auto abs_y = abs_subtraction_d(y_norm);
    std::vector<double> cumy(n);
    cumy[0] = abs_y[0];
    for (int i = 1; i < n; i++) cumy[i] = cumy[i-1] + abs_y[i];
    for (int i = 0; i < n; i++) cumy[i] /= n;

    // Output as float: x_signal = transpose([cumx, rad_norm, xs_raw]) -> shape (n, 3)
    out_x_signal.resize(n * 3);
    out_y_signal.resize(n * 3);
    for (int i = 0; i < n; i++) {
        out_x_signal[i * 3 + 0] = (float)cumx[i];
        out_x_signal[i * 3 + 1] = (float)rad_norm[i];
        out_x_signal[i * 3 + 2] = (float)xs_raw[i];
        out_y_signal[i * 3 + 0] = (float)cumy[i];
        out_y_signal[i * 3 + 1] = (float)rad_norm[i];
        out_y_signal[i * 3 + 2] = (float)ys_raw[i];
    }
}

static int model_selection(const NNModels& models, int length) {
    for (int i = 0; i < (int)models.crits.size() - 1; i++) {
        if (models.crits[i] <= length && length < models.crits[i+1]) {
            return models.reg_model_nums[i];
        }
    }
    return models.reg_model_nums[std::max(0, (int)models.reg_model_nums.size() - 2)];
}

// ============================================================
// ONNX Runtime implementation
// ============================================================

#ifdef USE_ONNXRUNTIME

bool load_nn_models(NNModels& models, const std::string& models_dir) {
    try {
        auto* env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "freetrace");
        models.env = env;

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);

        bool gpu_enabled = false;
        try {
            OrtCUDAProviderOptions cuda_opts;
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
            gpu_enabled = true;
        } catch (...) {
        }

        models.reg_model_nums = {3, 5, 8};
        models.crits = {3, 5, 8, 8192};

        for (int n : models.reg_model_nums) {
            std::string path = models_dir + "/reg_model_" + std::to_string(n) + ".onnx";
            auto* session = new Ort::Session(*env, path.c_str(), opts);
            models.alpha_sessions[n] = session;
        }

        std::string k_path = models_dir + "/reg_k_model.onnx";
        models.k_session = new Ort::Session(*env, k_path.c_str(), opts);

        models.loaded = true;
        if (gpu_enabled) {
            std::cout << "NN inference: GPU (CUDA) enabled" << std::endl;
        } else {
            std::cout << "NN inference: CPU mode" << std::endl;
        }
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX load error: " << e.what() << std::endl;
        return false;
    }
}

void free_nn_models(NNModels& models) {
    for (auto& [n, session] : models.alpha_sessions) {
        delete static_cast<Ort::Session*>(session);
    }
    models.alpha_sessions.clear();
    if (models.k_session) {
        delete static_cast<Ort::Session*>(models.k_session);
        models.k_session = nullptr;
    }
    if (models.env) {
        delete static_cast<Ort::Env*>(models.env);
        models.env = nullptr;
    }
    models.loaded = false;
}

float predict_alpha_nn(const NNModels& models,
                       const std::vector<float>& xs,
                       const std::vector<float>& ys) {
    if (!models.loaded || xs.size() < 3) return 1.0f;

    int n = (int)xs.size();
    int model_num = model_selection(models, n);

    auto it = models.alpha_sessions.find(model_num);
    if (it == models.alpha_sessions.end()) return 1.0f;
    auto* session = static_cast<Ort::Session*>(it->second);

    // Recoupe trajectory into sliding windows of size model_num
    // Use double for preprocessing, matching Python/NumPy float64
    std::vector<std::vector<double>> windows_x, windows_y;
    for (int i = 0; i + model_num <= n; i++) {
        windows_x.push_back(std::vector<double>(xs.begin() + i, xs.begin() + i + model_num));
        windows_y.push_back(std::vector<double>(ys.begin() + i, ys.begin() + i + model_num));
    }
    if (windows_x.empty()) return 1.0f;

    int batch_size = 2 * (int)windows_x.size();
    std::vector<float> input_data(batch_size * model_num * 1 * 3);

    for (int w = 0; w < (int)windows_x.size(); w++) {
        std::vector<float> x_sig, y_sig;
        cvt_2_signal(windows_x[w], windows_y[w], x_sig, y_sig);

        int offset_x = (2 * w) * model_num * 3;
        for (int i = 0; i < model_num * 3; i++) input_data[offset_x + i] = x_sig[i];

        int offset_y = (2 * w + 1) * model_num * 3;
        for (int i = 0; i < model_num * 3; i++) input_data[offset_y + i] = y_sig[i];
    }

    try {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> shape = {batch_size, model_num, 1, 3};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(),
                                                                   input_data.size(), shape.data(), shape.size());

        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name = session->GetInputNameAllocated(0, alloc);
        auto out_name = session->GetOutputNameAllocated(0, alloc);
        const char* input_names[] = {in_name.get()};
        const char* output_names[] = {out_name.get()};

        auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        float* pred_data = outputs[0].GetTensorMutableData<float>();
        // Use double for postprocessing (IQR mean)
        std::vector<double> preds(batch_size);
        for (int i = 0; i < batch_size; i++) preds[i] = (double)pred_data[i];

        if (preds.size() <= 4) return (float)vec_mean_d(preds);
        return (float)iqr_mean_d(preds);
    } catch (const Ort::Exception& e) {
        std::cerr << "Alpha predict error: " << e.what() << std::endl;
        return 1.0f;
    }
}

float predict_k_nn(const NNModels& models,
                   const std::vector<float>& xs,
                   const std::vector<float>& ys) {
    if (!models.loaded || xs.size() < 2) return 0.5f;

    auto* session = static_cast<Ort::Session*>(models.k_session);
    if (!session) return 0.5f;

    // Compute displacements in double precision
    std::vector<double> xd(xs.begin(), xs.end()), yd(ys.begin(), ys.end());
    auto disps = displacement_d(xd, yd);
    if (disps.empty()) return 0.5f;

    double log_disp;
    if ((int)xs.size() < 10) {
        log_disp = std::log10(vec_mean_d(disps));
    } else {
        log_disp = std::log10(iqr_mean_d(disps));
    }
    if (std::isnan(log_disp) || std::isinf(log_disp)) return 0.5f;

    try {
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<float> input_data = {(float)log_disp};
        std::vector<int64_t> shape = {1, 1};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), 1, shape.data(), 2);

        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name = session->GetInputNameAllocated(0, alloc);
        auto out_name = session->GetOutputNameAllocated(0, alloc);
        const char* input_names[] = {in_name.get()};
        const char* output_names[] = {out_name.get()};

        auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        float k = outputs[0].GetTensorMutableData<float>()[0];
        if (std::isnan(k)) return 1.0f;
        return k;
    } catch (const Ort::Exception& e) {
        std::cerr << "K predict error: " << e.what() << std::endl;
        return 0.5f;
    }
}

#else // No ONNX Runtime

bool load_nn_models(NNModels& /*models*/, const std::string& /*models_dir*/) {
    return false;
}

void free_nn_models(NNModels& /*models*/) {}

float predict_alpha_nn(const NNModels& /*models*/,
                       const std::vector<float>& /*xs*/,
                       const std::vector<float>& /*ys*/) {
    return 1.0f;
}

float predict_k_nn(const NNModels& /*models*/,
                   const std::vector<float>& /*xs*/,
                   const std::vector<float>& /*ys*/) {
    return 0.5f;
}

#endif // USE_ONNXRUNTIME

} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
