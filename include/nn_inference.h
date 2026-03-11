#pragma once // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <vector>
#include <string>
#include <array>
#include <map>

namespace freetrace { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Neural network inference for alpha/k prediction using ONNX Runtime // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

struct NNModels { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool loaded = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    void* env = nullptr; // Ort::Env* // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::map<int, void*> alpha_sessions; // model_num -> Ort::Session* // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    void* k_session = nullptr; // Ort::Session* // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<int> reg_model_nums; // [3, 5, 8] // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<int> crits; // [3, 5, 8, 8192] // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
};

// Load ONNX models from directory // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
bool load_nn_models(NNModels& models, const std::string& models_dir); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Free ONNX models // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
void free_nn_models(NNModels& models); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Predict alpha from x,y trajectory coordinates // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Returns alpha value (default 1.0 if models not loaded or trajectory too short)
float predict_alpha_nn(const NNModels& models, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                       const std::vector<float>& xs,
                       const std::vector<float>& ys);

// Predict k from x,y trajectory coordinates // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Returns k value (default 0.5 if models not loaded)
float predict_k_nn(const NNModels& models, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                   const std::vector<float>& xs,
                   const std::vector<float>& ys);

} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
