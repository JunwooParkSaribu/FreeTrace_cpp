// Stub NN inference for builds without ONNX Runtime // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
#include "nn_inference.h"

namespace freetrace {

bool load_nn_models(NNModels& models, const std::string&) {
    models.loaded = false;
    return false;
}

void free_nn_models(NNModels& models) {
    models.loaded = false;
}

float predict_alpha_nn(const NNModels&, const std::vector<float>&, const std::vector<float>&) {
    return 1.0f;
}

float predict_k_nn(const NNModels&, const std::vector<float>&, const std::vector<float>&) {
    return 0.5f;
}

} // namespace freetrace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
