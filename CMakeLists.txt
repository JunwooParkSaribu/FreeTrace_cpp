# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
cmake_minimum_required(VERSION 3.18)

# Auto-detect CUDA — enable GPU support if nvcc is found // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    project(FreeTrace_cpp VERSION 1.6.1.2 LANGUAGES CXX CUDA)
    set(USE_CUDA ON)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} — GPU acceleration enabled")
else()
    project(FreeTrace_cpp VERSION 1.6.1.2 LANGUAGES CXX)
    set(USE_CUDA OFF)
    message(STATUS "CUDA not found — GPU acceleration disabled (CPU only)")
endif() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC compatibility // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
if(MSVC)
    add_definitions(-D_USE_MATH_DEFINES -DNOMINMAX)
endif()

# --- Options ---
option(USE_ONNXRUNTIME "Enable ONNX Runtime for NN inference (fBm mode)" OFF)
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation")

# --- Apple Silicon optimizations --- # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
if(APPLE)
    # Detect ARM64 (Apple Silicon)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(APPLE_SILICON ON)
        message(STATUS "Apple Silicon detected — enabling ARM64 optimizations")
    else()
        set(APPLE_SILICON OFF)
    endif()

    # Apple Accelerate framework (vDSP + LAPACK, highly optimized for Apple Silicon)
    find_library(ACCELERATE_LIB Accelerate)
    if(ACCELERATE_LIB)
        add_definitions(-DUSE_ACCELERATE -DUSE_LAPACK)
        message(STATUS "Apple Accelerate framework found — vDSP + LAPACK enabled")
    endif()
endif() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

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

# CoreML native inference (macOS only — GPU/ANE acceleration for alpha models) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
if(APPLE)
    find_library(COREML_LIB CoreML)
    find_library(FOUNDATION_LIB Foundation)
    if(COREML_LIB AND FOUNDATION_LIB)
        enable_language(OBJCXX)
        list(APPEND CORE_SOURCES src/nn_coreml.mm)
        add_definitions(-DUSE_COREML)
        message(STATUS "CoreML framework found — native GPU/ANE inference enabled")
    endif()
endif() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

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

# AVX2 for x86-simd-sort (used in tracking argsort) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
# Only enable if both compiler supports AVX2 AND the x86-simd-sort headers are present
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAS_AVX2)
if(HAS_AVX2 AND EXISTS "${CMAKE_SOURCE_DIR}/x86-simd-sort/src/x86simdsort-static-incl.h")
    target_compile_options(freetrace PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
    message(STATUS "AVX2 enabled with x86-simd-sort")
else()
    message(STATUS "AVX2 disabled (x86-simd-sort not found or AVX2 not supported)")
endif()

# Apple Silicon: ARM tuning flags // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
if(APPLE_SILICON)
    target_compile_options(freetrace PRIVATE -mcpu=apple-m1 -O3)
    message(STATUS "Apple Silicon: -mcpu=apple-m1 -O3 enabled")
endif() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

# Disable FP contraction (FMA fusion) for deterministic floating-point results // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
# Without this, -O3 can produce different results across compilations
if(MSVC)
    set_source_files_properties(src/tracking.cpp PROPERTIES COMPILE_FLAGS "/fp:strict")
else()
    set_source_files_properties(src/tracking.cpp PROPERTIES COMPILE_FLAGS "-ffp-contract=off")
endif()

# --- Link libraries ---

if(OpenCV_FOUND)
    target_include_directories(freetrace PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(freetrace PRIVATE ${OpenCV_LIBS})
endif()

if(TIFF_FOUND) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    if(TARGET TIFF::TIFF)
        target_link_libraries(freetrace PRIVATE TIFF::TIFF)
    else()
        target_include_directories(freetrace PRIVATE ${TIFF_INCLUDE_DIRS})
        target_link_libraries(freetrace PRIVATE ${TIFF_LIBRARIES})
    endif()
endif()

if(PNG_FOUND)
    if(TARGET PNG::PNG)
        target_link_libraries(freetrace PRIVATE PNG::PNG)
    else()
        target_include_directories(freetrace PRIVATE ${PNG_INCLUDE_DIRS})
        target_link_libraries(freetrace PRIVATE ${PNG_LIBRARIES})
    endif()
endif() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

if(OpenMP_CXX_FOUND)
    target_link_libraries(freetrace PRIVATE OpenMP::OpenMP_CXX)
endif()

# Apple Accelerate framework (vDSP + LAPACK) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
if(APPLE AND ACCELERATE_LIB)
    target_link_libraries(freetrace PRIVATE ${ACCELERATE_LIB})
endif()

# Apple CoreML + Foundation frameworks (native GPU/ANE inference) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
if(APPLE AND COREML_LIB)
    target_link_libraries(freetrace PRIVATE ${COREML_LIB} ${FOUNDATION_LIB})
endif() # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16

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
