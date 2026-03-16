// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
// CoreML native inference for alpha models on macOS
// Uses Apple's CoreML framework to dispatch LSTM layers to GPU/ANE
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Load .mlpackage alpha models from directory
// model_nums: array of model numbers to load (e.g., {3, 5, 8})
// Returns true if at least one model was loaded
int coreml_load_alpha_models(const char* models_dir, const int* model_nums, int count);

// Check if CoreML models are available
int coreml_is_available(void);

// Predict alpha for a batch
// input_data: float array of shape [batch_size * seq_len * 1 * 3] (row-major)
// output_data: pre-allocated float array of size [batch_size] for results
// Returns 1 on success, 0 on failure
int coreml_predict_alpha(int model_num, const float* input_data,
                         int batch_size, int seq_len,
                         float* output_data);

// Free all loaded CoreML models
void coreml_free_models(void);

#ifdef __cplusplus
}
#endif
// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
