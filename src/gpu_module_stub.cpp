// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:40
// Stub implementation when CUDA is not available.
// Only compiled when USE_CUDA is NOT defined.
#ifndef USE_CUDA

#include "gpu_module.h"

namespace freetrace {
namespace gpu {

bool is_available() { return false; }
int get_gpu_mem_size() { return 0; }
size_t get_gpu_free_mem_bytes() { return 0; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

std::vector<float> likelihood_gpu(
    const std::vector<float>&, const std::vector<float>&, float,
    const std::vector<float>&, const std::vector<float>&,
    int, int, int
) { return {}; }

std::vector<float> image_cropping_gpu(
    const std::vector<float>&,
    int, int, int, int, int, int, int, int&
) { return {}; }

void compute_background_gpu( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    const std::vector<float>&,
    int, int, int,
    float*, float*
) {} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

} // namespace gpu
} // namespace freetrace

#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:40
