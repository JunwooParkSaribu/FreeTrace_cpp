# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
cmake_minimum_required(VERSION 3.18)

# Auto-detect CUDA — enable GPU support if nvcc is found // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    project(FreeTrace_cpp VERSION 0.1.0 LANGUAGES CXX CUDA)
    set(USE_CUDA ON)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} — GPU acceleration enabled")
else()
    project(FreeTrace_cpp VERSION 0.1.0 LANGUAGES CXX)
    set(USE_CUDA OFF)
    message(STATUS "CUDA not found — GPU acceleration disabled (CPU only)")
endif() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Options ---
option(USE_ONNXRUNTIME "Enable ONNX Runtime for NN inference (fBm mode)" OFF)
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation")

# --- Dependencies ---

# OpenMP (optional, parallel localization loops)
find_package(OpenMP QUIET)

# OpenCV (optional, alternative TIFF I/O)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    add_definitions(-DUSE_OPENCV)
endif()

# libtiff (preferred for TIFF I/O when OpenCV is not available)
find_package(TIFF QUIET)
if(TIFF_FOUND AND NOT OpenCV_FOUND)
    add_definitions(-DUSE_LIBTIFF)
endif()

# libpng (for trajectory images and density plots)
find_package(PNG QUIET)
if(PNG_FOUND)
    add_definitions(-DUSE_LIBPNG)
endif()

# --- Source files ---

set(CORE_SOURCES
    src/main.cpp
    src/image_pad.cpp
    src/regression.cpp
    src/cost_function.cpp
    src/localization.cpp
    src/tracking.cpp
)

# NN inference: real or stub
if(USE_ONNXRUNTIME AND EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    list(APPEND CORE_SOURCES src/nn_inference.cpp)
    add_definitions(-DUSE_ONNXRUNTIME)
else()
    list(APPEND CORE_SOURCES src/nn_inference_stub.cpp)
    if(USE_ONNXRUNTIME)
        message(WARNING "ONNX Runtime not found at ${ONNXRUNTIME_DIR}, using NN stub")
    endif()
endif()

# GPU module: CUDA or stub // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
if(USE_CUDA)
    list(APPEND CORE_SOURCES src/gpu_module.cu)
    add_definitions(-DUSE_CUDA)
else()
    list(APPEND CORE_SOURCES src/gpu_module_stub.cpp)
endif() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

# --- Target ---

add_executable(freetrace ${CORE_SOURCES})

target_include_directories(freetrace PRIVATE include)

# CUDA settings // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
if(USE_CUDA)
    set_target_properties(freetrace PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(freetrace PRIVATE cudart)
endif() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

# AVX2 for x86-simd-sort (used in tracking argsort)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAS_AVX2)
if(HAS_AVX2)
    target_compile_options(freetrace PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
endif()

# Disable FP contraction (FMA fusion) for deterministic floating-point results
# Without this, -O3 can produce different results across compilations
set_source_files_properties(src/tracking.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

# --- Link libraries ---

if(OpenCV_FOUND)
    target_include_directories(freetrace PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(freetrace PRIVATE ${OpenCV_LIBS})
endif()

if(TIFF_FOUND)
    target_link_libraries(freetrace PRIVATE ${TIFF_LIBRARIES})
endif()

if(PNG_FOUND)
    target_link_libraries(freetrace PRIVATE ${PNG_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(freetrace PRIVATE OpenMP::OpenMP_CXX)
endif()

if(USE_ONNXRUNTIME AND EXISTS "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    target_include_directories(freetrace PRIVATE ${ONNXRUNTIME_DIR}/include)
    target_link_directories(freetrace PRIVATE ${ONNXRUNTIME_DIR}/lib)
    target_link_libraries(freetrace PRIVATE onnxruntime)
    # Set rpath so the binary finds libonnxruntime at runtime
    if(UNIX AND NOT APPLE)
        set_target_properties(freetrace PROPERTIES
            INSTALL_RPATH "$ORIGIN;${ONNXRUNTIME_DIR}/lib"
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")
    elseif(APPLE)
        set_target_properties(freetrace PROPERTIES
            INSTALL_RPATH "@executable_path;${ONNXRUNTIME_DIR}/lib"
            BUILD_RPATH "${ONNXRUNTIME_DIR}/lib")
    endif()
endif()
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
