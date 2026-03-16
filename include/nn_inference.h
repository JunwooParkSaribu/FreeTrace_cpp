#pragma once // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <vector>
#include <string>
#include <array>
#include <map>

namespace freetrace { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Neural network inference for alpha/k prediction using ONNX Runtime // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Direct k model: 3-layer linear network (1→256→128→1), no activations // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
struct KModelDirect {
    bool loaded = false;
    std::vector<float> W1;  // (1, 256) flattened
    std::vector<float> b1;  // (256,)
    std::vector<float> W2;  // (256, 128) flattened
    std::vector<float> b2;  // (128,)
    std::vector<float> W3;  // (128, 1) flattened
    float b3 = 0.0f;
};

struct NNModels { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
    bool loaded = false;
    bool use_coreml = false;        // CoreML native inference (macOS GPU/ANE)
    void* env = nullptr;            // Ort::Env*
    void* run_options = nullptr;    // Ort::RunOptions* (cached, reused)
    void* mem_info = nullptr;       // Ort::MemoryInfo* (cached, reused)
    std::map<int, void*> alpha_sessions;  // model_num -> Ort::Session*
    void* k_session = nullptr;            // Ort::Session* (kept for batch)
    KModelDirect k_direct;                // Direct computation (fast path)
    // Cached input/output names (avoid per-call heap allocation)
    std::map<int, std::string> alpha_input_names;   // model_num -> name
    std::map<int, std::string> alpha_output_names;  // model_num -> name
    std::string k_input_name;
    std::string k_output_name;
    std::vector<int> reg_model_nums; // [3, 5, 8]
    std::vector<int> crits;          // [3, 5, 8, 8192]
}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// Load ONNX models from directory // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
bool load_nn_models(NNModels& models, const std::string& models_dir); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Free ONNX models // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
void free_nn_models(NNModels& models); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Predict alpha from x,y trajectory coordinates // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Returns alpha value (default 1.0 if models not loaded or trajectory too short)
float predict_alpha_nn(const NNModels& models, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                       const std::vector<float>& xs,
                       const std::vector<float>& ys);

// Predict k from x,y trajectory coordinates
// Returns k value (default 0.5 if models not loaded)
float predict_k_nn(const NNModels& models,
                   const std::vector<float>& xs,
                   const std::vector<float>& ys);

// Batched predictions — single ONNX call for multiple trajectories // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
std::vector<float> predict_alpha_nn_batch(const NNModels& models,
                                          const std::vector<std::vector<float>>& xs_batch,
                                          const std::vector<std::vector<float>>& ys_batch);
std::vector<float> predict_k_nn_batch(const NNModels& models,
                                      const std::vector<std::vector<float>>& xs_batch,
                                      const std::vector<std::vector<float>>& ys_batch);

} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
