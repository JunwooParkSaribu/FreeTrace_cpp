// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
#include "nn_inference.h"

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#ifdef USE_COREML
#include "nn_coreml.h"
#endif

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <thread> // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

namespace freetrace {

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
// Direct k model: load weights and compute without ONNX Runtime
// Architecture: input(1) → dense(256) → dense(128) → dense(1), no activations
// ============================================================

static bool load_k_direct(KModelDirect& km, const std::string& models_dir) {
    std::string path = models_dir + "/k_model_weights.bin";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    int dims[6];
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));
    // dims: {1, 256, 256, 128, 128, 1}
    int w1_size = dims[0] * dims[1]; // 256
    int b1_size = dims[1];           // 256
    int w2_size = dims[2] * dims[3]; // 32768
    int b2_size = dims[3];           // 128
    int w3_size = dims[4] * dims[5]; // 128
    int b3_size = dims[5];           // 1

    km.W1.resize(w1_size); f.read(reinterpret_cast<char*>(km.W1.data()), w1_size * sizeof(float));
    km.b1.resize(b1_size); f.read(reinterpret_cast<char*>(km.b1.data()), b1_size * sizeof(float));
    km.W2.resize(w2_size); f.read(reinterpret_cast<char*>(km.W2.data()), w2_size * sizeof(float));
    km.b2.resize(b2_size); f.read(reinterpret_cast<char*>(km.b2.data()), b2_size * sizeof(float));
    km.W3.resize(w3_size); f.read(reinterpret_cast<char*>(km.W3.data()), w3_size * sizeof(float));
    float b3_val; f.read(reinterpret_cast<char*>(&b3_val), sizeof(float));
    km.b3 = b3_val;
    km.loaded = f.good();
    return km.loaded;
}

// Direct computation: x -> W1*x+b1 -> W2*h+b2 -> W3*h+b3
static float k_direct_predict(const KModelDirect& km, float input) {
    // Layer 1: h1 = W1 * input + b1  (1 → 256)
    // W1 is (1, 256) so h1[i] = W1[i] * input + b1[i]
    float h1[256];
    for (int i = 0; i < 256; i++)
        h1[i] = km.W1[i] * input + km.b1[i];

    // Layer 2: h2 = W2^T * h1 + b2  (256 → 128)
    // W2 is (256, 128), h2[j] = sum_i(W2[i*128+j] * h1[i]) + b2[j]
    float h2[128];
    for (int j = 0; j < 128; j++) {
        float sum = km.b2[j];
        for (int i = 0; i < 256; i++)
            sum += km.W2[i * 128 + j] * h1[i];
        h2[j] = sum;
    }

    // Layer 3: out = W3^T * h2 + b3  (128 → 1)
    float out = km.b3;
    for (int i = 0; i < 128; i++)
        out += km.W3[i] * h2[i];

    return out;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

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

// Helper: create session options with thread config // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-24
static Ort::SessionOptions make_session_opts(int num_threads, bool try_cuda, bool& gpu_enabled) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(num_threads);
    opts.SetInterOpNumThreads(num_threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    gpu_enabled = false;
#if !defined(__APPLE__)
    if (try_cuda) {
        try {
            OrtCUDAProviderOptions cuda_opts;
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
            gpu_enabled = true;
        } catch (...) {
        }
    }
#endif
    return opts;
}

// Helper: load all ONNX sessions with the given options
static bool load_sessions(NNModels& models, Ort::Env* env,
                           const std::string& models_dir,
                           Ort::SessionOptions& opts) {
    Ort::AllocatorWithDefaultOptions alloc;

    for (int n : models.reg_model_nums) {
        std::string path = models_dir + "/reg_model_" + std::to_string(n) + ".onnx";
#ifdef _WIN32
        std::wstring wpath(path.begin(), path.end());
        auto* session = new Ort::Session(*env, wpath.c_str(), opts);
#else
        auto* session = new Ort::Session(*env, path.c_str(), opts);
#endif
        models.alpha_sessions[n] = session;
        models.alpha_input_names[n] = std::string(session->GetInputNameAllocated(0, alloc).get());
        models.alpha_output_names[n] = std::string(session->GetOutputNameAllocated(0, alloc).get());
    }

    std::string k_path = models_dir + "/reg_k_model.onnx";
#ifdef _WIN32
    std::wstring wk_path(k_path.begin(), k_path.end());
    auto* k_sess = new Ort::Session(*env, wk_path.c_str(), opts);
#else
    auto* k_sess = new Ort::Session(*env, k_path.c_str(), opts);
#endif
    models.k_session = k_sess;
    models.k_input_name = std::string(k_sess->GetInputNameAllocated(0, alloc).get());
    models.k_output_name = std::string(k_sess->GetOutputNameAllocated(0, alloc).get());
    return true;
}

bool load_nn_models(NNModels& models, const std::string& models_dir) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
    try {
        models.reg_model_nums = {3, 5, 8};
        models.crits = {3, 5, 8, 8192};

        // Load direct k model weights (fast path, bypasses ONNX for k predictions)
        bool has_k_direct = load_k_direct(models.k_direct, models_dir);
        if (has_k_direct) {
            std::cout << "Loaded direct k model weights (fast path)" << std::endl;
        }

#ifdef USE_COREML
        // macOS: CoreML only — no ONNX at all  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
        if (!coreml_load_alpha_models(models_dir.c_str(),
                                      models.reg_model_nums.data(),
                                      (int)models.reg_model_nums.size())) {
            return false;  // models not in this directory, caller tries next
        }
        models.use_coreml = true;
        models.loaded = true;
        std::cout << "\n  NN inference: CoreML (GPU / Apple Neural Engine)\n" << std::endl;
#endif
#ifndef USE_COREML
        {
            // Non-Apple: use ONNX Runtime
            auto* env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "freetrace");
            models.env = env;
            int num_threads = std::max(1, (int)std::thread::hardware_concurrency());
            models.run_options = new Ort::RunOptions();
            models.mem_info = new Ort::MemoryInfo(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

            bool gpu_enabled = false;
            auto opts = make_session_opts(num_threads, true, gpu_enabled);

            bool sessions_ok = false;
            if (gpu_enabled) {
                try {
                    sessions_ok = load_sessions(models, env, models_dir, opts);
                } catch (...) {
                    // CUDA EP registered but session creation failed (no GPU driver, etc.)
                    for (auto& [n, session] : models.alpha_sessions)
                        delete static_cast<Ort::Session*>(session);
                    models.alpha_sessions.clear();
                    if (models.k_session) {
                        delete static_cast<Ort::Session*>(models.k_session);
                        models.k_session = nullptr;
                    }
                    gpu_enabled = false;
                    opts = make_session_opts(num_threads, false, gpu_enabled);
                    std::cout << "\n  [WARNING] fBm mode is enabled, but no GPU is detected for neural network inference.\n"
                              << "  Loading NN models on CPU - this may take a moment.\n"
                              << "  Note: tracking will be significantly slower due to CPU-based neural network inference.\n" << std::endl;
                    sessions_ok = load_sessions(models, env, models_dir, opts);
                }
            } else {
                std::cout << "\n  [WARNING] fBm mode is enabled, but no GPU is detected for neural network inference.\n"
                          << "  Loading NN models on CPU - this may take a moment.\n"
                          << "  Note: tracking will be significantly slower due to CPU-based neural network inference.\n" << std::endl;
                sessions_ok = load_sessions(models, env, models_dir, opts);
            }

            if (!sessions_ok) return false;
            models.loaded = true;

            if (gpu_enabled) {
                std::cout << "\n  NN inference: GPU (CUDA)\n" << std::endl;
            } else {
                std::cout << "\n  NN inference: CPU (" << num_threads << " threads)\n" << std::endl;
            }
        }  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26
#endif // !USE_COREML
        return true;
    } catch (const std::exception& e) {
        std::cerr << "  NN model loading failed: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "  NN model loading failed (unknown error)." << std::endl;
        return false;
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-25

void free_nn_models(NNModels& models) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    for (auto& [n, session] : models.alpha_sessions) {
        delete static_cast<Ort::Session*>(session);
    }
    models.alpha_sessions.clear();
    if (models.k_session) {
        delete static_cast<Ort::Session*>(models.k_session);
        models.k_session = nullptr;
    }
    if (models.run_options) {
        delete static_cast<Ort::RunOptions*>(models.run_options);
        models.run_options = nullptr;
    }
    if (models.mem_info) {
        delete static_cast<Ort::MemoryInfo*>(models.mem_info);
        models.mem_info = nullptr;
    }
    if (models.env) {
        delete static_cast<Ort::Env*>(models.env);
        models.env = nullptr;
    }
    models.loaded = false;
#ifdef USE_COREML // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    if (models.use_coreml) coreml_free_models();
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13


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

#ifdef USE_COREML // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    if (models.use_coreml) {
        std::vector<float> pred_data(batch_size);
        if (coreml_predict_alpha(model_num, input_data.data(), batch_size, model_num, pred_data.data())) {
            std::vector<double> preds(batch_size);
            for (int i = 0; i < batch_size; i++) preds[i] = (double)pred_data[i];
            if (preds.size() <= 4) return (float)vec_mean_d(preds);
            return (float)iqr_mean_d(preds);
        }
    }
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

    try {
        auto& mem_info = *static_cast<Ort::MemoryInfo*>(models.mem_info);
        std::vector<int64_t> shape = {batch_size, model_num, 1, 3};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(),
                                                                   input_data.size(), shape.data(), shape.size());

        const char* input_names[] = {models.alpha_input_names.at(model_num).c_str()};
        const char* output_names[] = {models.alpha_output_names.at(model_num).c_str()};

        auto outputs = session->Run(*static_cast<Ort::RunOptions*>(models.run_options),
                                    input_names, &input_tensor, 1, output_names, 1);

        float* pred_data = outputs[0].GetTensorMutableData<float>();
        std::vector<double> preds(batch_size);
        for (int i = 0; i < batch_size; i++) preds[i] = (double)pred_data[i];

        if (preds.size() <= 4) return (float)vec_mean_d(preds);
        return (float)iqr_mean_d(preds);
    } catch (const Ort::Exception& e) {
        std::cerr << "Alpha predict error: " << e.what() << std::endl;
        return 1.0f;
    }
}

std::vector<float> predict_alpha_nn_batch(const NNModels& models, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                                          const std::vector<std::vector<float>>& xs_batch,
                                          const std::vector<std::vector<float>>& ys_batch) {

    int N = (int)xs_batch.size();
    std::vector<float> results(N, 1.0f);
    if (!models.loaded || N == 0) return results;

    // Group trajectories by model_num for batched inference
    // model_num -> list of (index_in_batch, preprocessed_windows_count)
    struct TrajectoryInfo {
        int orig_idx;
        int num_windows;  // 2 * actual windows (x + y)
    };
    std::map<int, std::vector<TrajectoryInfo>> groups;
    std::map<int, std::vector<float>> group_input_data;

    for (int idx = 0; idx < N; idx++) {
        if (xs_batch[idx].size() < 3) continue;
        int n = (int)xs_batch[idx].size();
        int model_num = model_selection(models, n);
        if (models.alpha_sessions.find(model_num) == models.alpha_sessions.end()) continue;

        // Recoupe into windows
        std::vector<std::vector<double>> windows_x, windows_y;
        for (int i = 0; i + model_num <= n; i++) {
            windows_x.push_back(std::vector<double>(xs_batch[idx].begin() + i,
                                                     xs_batch[idx].begin() + i + model_num));
            windows_y.push_back(std::vector<double>(ys_batch[idx].begin() + i,
                                                     ys_batch[idx].begin() + i + model_num));
        }
        if (windows_x.empty()) continue;

        int batch_size = 2 * (int)windows_x.size();
        groups[model_num].push_back({idx, batch_size});

        // Preprocess and append to group input data
        auto& data = group_input_data[model_num];
        for (int w = 0; w < (int)windows_x.size(); w++) {
            std::vector<float> x_sig, y_sig;
            cvt_2_signal(windows_x[w], windows_y[w], x_sig, y_sig);
            data.insert(data.end(), x_sig.begin(), x_sig.end());
            data.insert(data.end(), y_sig.begin(), y_sig.end());
        }
    }

    // Run one ONNX call per model_num (typically 1-3 calls instead of N)
    for (auto& [model_num, infos] : groups) {
        auto it = models.alpha_sessions.find(model_num);
        if (it == models.alpha_sessions.end()) continue;
        auto* session = static_cast<Ort::Session*>(it->second);
        auto& data = group_input_data[model_num];

        int total_batch = 0;
        for (auto& info : infos) total_batch += info.num_windows;
        if (total_batch == 0) continue;

        std::vector<float> pred_buf(total_batch); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
        bool predicted = false;

#ifdef USE_COREML
        if (models.use_coreml) {
            predicted = coreml_predict_alpha(model_num, data.data(),
                                            total_batch, model_num, pred_buf.data());
        }
#endif
        if (!predicted) {
            try {
                auto& mem_info = *static_cast<Ort::MemoryInfo*>(models.mem_info);
                std::vector<int64_t> shape = {total_batch, model_num, 1, 3};
                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    mem_info, data.data(), data.size(), shape.data(), shape.size());
                const char* input_names[] = {models.alpha_input_names.at(model_num).c_str()};
                const char* output_names[] = {models.alpha_output_names.at(model_num).c_str()};

                auto outputs = session->Run(*static_cast<Ort::RunOptions*>(models.run_options),
                                            input_names, &input_tensor, 1, output_names, 1);

                float* p = outputs[0].GetTensorMutableData<float>();
                for (int i = 0; i < total_batch; i++) pred_buf[i] = p[i];
                predicted = true;
            } catch (const Ort::Exception& e) {
                std::cerr << "Alpha batch predict error: " << e.what() << std::endl;
            }
        }

        if (predicted) {
            int offset = 0;
            for (auto& info : infos) {
                std::vector<double> preds(info.num_windows);
                for (int i = 0; i < info.num_windows; i++)
                    preds[i] = (double)pred_buf[offset + i];
                offset += info.num_windows;

                if (preds.size() <= 4)
                    results[info.orig_idx] = (float)vec_mean_d(preds);
                else
                    results[info.orig_idx] = (float)iqr_mean_d(preds);
            }
        } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    }
    return results;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

float predict_k_nn(const NNModels& models, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                   const std::vector<float>& xs,
                   const std::vector<float>& ys) {

    if (!models.loaded || xs.size() < 2) return 0.5f;

    // Compute log displacement (shared preprocessing)
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

    // Fast path: direct computation (no ONNX overhead)
    if (models.k_direct.loaded) {
        float k = k_direct_predict(models.k_direct, (float)log_disp);
        return std::isnan(k) ? 1.0f : k;
    }

    // Fallback: ONNX Runtime
    auto* session = static_cast<Ort::Session*>(models.k_session);
    if (!session) return 0.5f;
    try {
        auto& mem_info = *static_cast<Ort::MemoryInfo*>(models.mem_info);
        float input_val = (float)log_disp;
        std::vector<int64_t> shape = {1, 1};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, &input_val, 1, shape.data(), 2);
        const char* input_names[] = {models.k_input_name.c_str()};
        const char* output_names[] = {models.k_output_name.c_str()};
        auto outputs = session->Run(*static_cast<Ort::RunOptions*>(models.run_options),
                                    input_names, &input_tensor, 1, output_names, 1);
        float k = outputs[0].GetTensorMutableData<float>()[0];
        return std::isnan(k) ? 1.0f : k;
    } catch (const Ort::Exception& e) {
        std::cerr << "K predict error: " << e.what() << std::endl;
        return 0.5f;
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

std::vector<float> predict_k_nn_batch(const NNModels& models, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                                      const std::vector<std::vector<float>>& xs_batch,
                                      const std::vector<std::vector<float>>& ys_batch) {

    int N = (int)xs_batch.size();
    std::vector<float> results(N, 0.5f);
    if (!models.loaded || N == 0) return results;

    auto* session = static_cast<Ort::Session*>(models.k_session);
    if (!session) return results;

    // Compute all log_displacements
    std::vector<float> input_data;
    std::vector<int> valid_indices;
    input_data.reserve(N);

    for (int i = 0; i < N; i++) {
        if (xs_batch[i].size() < 2) continue;
        std::vector<double> xd(xs_batch[i].begin(), xs_batch[i].end());
        std::vector<double> yd(ys_batch[i].begin(), ys_batch[i].end());
        auto disps = displacement_d(xd, yd);
        if (disps.empty()) continue;

        double log_disp;
        if ((int)xs_batch[i].size() < 10) {
            log_disp = std::log10(vec_mean_d(disps));
        } else {
            log_disp = std::log10(iqr_mean_d(disps));
        }
        if (std::isnan(log_disp) || std::isinf(log_disp)) continue;
        input_data.push_back((float)log_disp);
        valid_indices.push_back(i);
    }

    if (input_data.empty()) return results;

    // Fast path: direct computation // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    if (models.k_direct.loaded) {
        for (int i = 0; i < (int)input_data.size(); i++) {
            float k = k_direct_predict(models.k_direct, input_data[i]);
            results[valid_indices[i]] = std::isnan(k) ? 1.0f : k;
        }
        return results;
    }

    // Fallback: ONNX Runtime batched
    try {
        auto& mem_info = *static_cast<Ort::MemoryInfo*>(models.mem_info);
        int batch = (int)input_data.size();
        std::vector<int64_t> shape = {batch, 1};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(), shape.data(), 2);

        const char* input_names[] = {models.k_input_name.c_str()};
        const char* output_names[] = {models.k_output_name.c_str()};

        auto outputs = session->Run(*static_cast<Ort::RunOptions*>(models.run_options),
                                    input_names, &input_tensor, 1, output_names, 1);

        float* pred_data = outputs[0].GetTensorMutableData<float>();
        for (int i = 0; i < batch; i++) {
            float k = pred_data[i];
            results[valid_indices[i]] = std::isnan(k) ? 1.0f : k;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "K batch predict error: " << e.what() << std::endl;
    }
    return results; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

#else // No ONNX Runtime  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

bool load_nn_models(NNModels& models, const std::string& models_dir) {
#ifdef USE_COREML
    try {
        models.reg_model_nums = {3, 5, 8};
        models.crits = {3, 5, 8, 8192};
        load_k_direct(models.k_direct, models_dir);
        if (!coreml_load_alpha_models(models_dir.c_str(),
                                      models.reg_model_nums.data(),
                                      (int)models.reg_model_nums.size())) {
            return false;
        }
        models.use_coreml = true;
        models.loaded = true;
        std::cout << "\n  NN inference: CoreML (GPU / Apple Neural Engine)\n" << std::endl;
        return true;
    } catch (...) {
        return false;
    }
#else
    (void)models; (void)models_dir;
    return false;
#endif
}

void free_nn_models(NNModels& models) {
#ifdef USE_COREML
    coreml_free_models();
#endif
    models.loaded = false;
}

float predict_alpha_nn(const NNModels& models,
                       const std::vector<float>& xs,
                       const std::vector<float>& ys) {
#ifdef USE_COREML
    if (!models.loaded || xs.size() < 3) return 1.0f;
    int n = (int)xs.size();
    int model_num = model_selection(models, n);

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

    std::vector<float> pred_data(batch_size);
    if (coreml_predict_alpha(model_num, input_data.data(), batch_size, model_num, pred_data.data())) {
        std::vector<double> preds(batch_size);
        for (int i = 0; i < batch_size; i++) preds[i] = (double)pred_data[i];
        if (preds.size() <= 4) return (float)vec_mean_d(preds);
        return (float)iqr_mean_d(preds);
    }
#else
    (void)models; (void)xs; (void)ys;
#endif
    return 1.0f;
}

float predict_k_nn(const NNModels& models,
                   const std::vector<float>& xs,
                   const std::vector<float>& ys) {
    if (!models.loaded || xs.size() < 2) return 0.5f;
    std::vector<double> xd(xs.begin(), xs.end()), yd(ys.begin(), ys.end());
    auto disps = displacement_d(xd, yd);
    if (disps.empty()) return 0.5f;
    double log_disp;
    if ((int)xs.size() < 10) log_disp = std::log10(vec_mean_d(disps));
    else log_disp = std::log10(iqr_mean_d(disps));
    if (std::isnan(log_disp) || std::isinf(log_disp)) return 0.5f;
    if (models.k_direct.loaded) {
        float k = k_direct_predict(models.k_direct, (float)log_disp);
        return std::isnan(k) ? 1.0f : k;
    }
    return 0.5f;
}

std::vector<float> predict_alpha_nn_batch(const NNModels& models,
                                          const std::vector<std::vector<float>>& xs_batch,
                                          const std::vector<std::vector<float>>& ys_batch) {
    int N = (int)xs_batch.size();
    std::vector<float> results(N, 1.0f);
#ifdef USE_COREML
    if (!models.loaded || N == 0) return results;

    std::map<int, std::vector<std::pair<int,int>>> groups; // model_num -> [(orig_idx, num_windows)]
    std::map<int, std::vector<float>> group_data;

    for (int idx = 0; idx < N; idx++) {
        if (xs_batch[idx].size() < 3) continue;
        int n = (int)xs_batch[idx].size();
        int model_num = model_selection(models, n);
        std::vector<std::vector<double>> wx, wy;
        for (int i = 0; i + model_num <= n; i++) {
            wx.push_back(std::vector<double>(xs_batch[idx].begin() + i, xs_batch[idx].begin() + i + model_num));
            wy.push_back(std::vector<double>(ys_batch[idx].begin() + i, ys_batch[idx].begin() + i + model_num));
        }
        if (wx.empty()) continue;
        int bsz = 2 * (int)wx.size();
        groups[model_num].push_back({idx, bsz});
        auto& data = group_data[model_num];
        for (int w = 0; w < (int)wx.size(); w++) {
            std::vector<float> x_sig, y_sig;
            cvt_2_signal(wx[w], wy[w], x_sig, y_sig);
            data.insert(data.end(), x_sig.begin(), x_sig.end());
            data.insert(data.end(), y_sig.begin(), y_sig.end());
        }
    }

    for (auto& [model_num, infos] : groups) {
        auto& data = group_data[model_num];
        int total_batch = 0;
        for (auto& [idx, nw] : infos) total_batch += nw;
        if (total_batch == 0) continue;
        std::vector<float> pred_buf(total_batch);
        if (coreml_predict_alpha(model_num, data.data(), total_batch, model_num, pred_buf.data())) {
            int offset = 0;
            for (auto& [orig_idx, nw] : infos) {
                std::vector<double> preds(nw);
                for (int i = 0; i < nw; i++) preds[i] = (double)pred_buf[offset + i];
                offset += nw;
                if (preds.size() <= 4) results[orig_idx] = (float)vec_mean_d(preds);
                else results[orig_idx] = (float)iqr_mean_d(preds);
            }
        }
    }
#else
    (void)models; (void)ys_batch;
#endif
    return results;
}

std::vector<float> predict_k_nn_batch(const NNModels& models,
                                      const std::vector<std::vector<float>>& xs_batch,
                                      const std::vector<std::vector<float>>& ys_batch) {
    int N = (int)xs_batch.size();
    std::vector<float> results(N, 0.5f);
    if (!models.loaded || N == 0) return results;
    for (int i = 0; i < N; i++) {
        if (xs_batch[i].size() < 2) continue;
        std::vector<double> xd(xs_batch[i].begin(), xs_batch[i].end());
        std::vector<double> yd(ys_batch[i].begin(), ys_batch[i].end());
        auto disps = displacement_d(xd, yd);
        if (disps.empty()) continue;
        double log_disp;
        if ((int)xs_batch[i].size() < 10) log_disp = std::log10(vec_mean_d(disps));
        else log_disp = std::log10(iqr_mean_d(disps));
        if (std::isnan(log_disp) || std::isinf(log_disp)) continue;
        if (models.k_direct.loaded) {
            float k = k_direct_predict(models.k_direct, (float)log_disp);
            results[i] = std::isnan(k) ? 1.0f : k;
        }
    }
    return results;
}

#endif // USE_ONNXRUNTIME  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-26

} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
