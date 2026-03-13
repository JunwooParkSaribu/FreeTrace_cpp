# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
cmake_minimum_required(VERSION 3.16)
project(FreeTrace_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Options ---
option(USE_ONNXRUNTIME "Enable ONNX Runtime for NN inference (fBm mode)" OFF)
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation")

# --- Dependencies ---

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

# GPU module: stub (CUDA build requires separate setup)
list(APPEND CORE_SOURCES src/gpu_module_stub.cpp)

# --- Target ---

add_executable(freetrace ${CORE_SOURCES})

target_include_directories(freetrace PRIVATE include)

# AVX2 for x86-simd-sort (used in tracking argsort)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" HAS_AVX2)
if(HAS_AVX2)
    target_compile_options(freetrace PRIVATE -mavx2)
endif()

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
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
