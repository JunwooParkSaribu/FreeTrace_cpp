# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
cmake_minimum_required(VERSION 3.16)
project(FreeTrace_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV QUIET) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
if(OpenCV_FOUND) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    add_definitions(-DUSE_OPENCV) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
endif()

# libtiff (preferred over OpenCV for TIFF I/O) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
find_package(TIFF QUIET) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
if(TIFF_FOUND AND NOT OpenCV_FOUND) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    add_definitions(-DUSE_LIBTIFF) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
endif()

# libpng (for density and trajectory images) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
find_package(PNG QUIET) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
if(PNG_FOUND) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    add_definitions(-DUSE_LIBPNG) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
endif()

add_executable(freetrace # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    src/main.cpp
    src/image_pad.cpp
    src/regression.cpp
    src/cost_function.cpp
    src/localization.cpp # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    src/tracking.cpp # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
)

target_include_directories(freetrace PRIVATE include ${OpenCV_INCLUDE_DIRS}) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
target_link_libraries(freetrace PRIVATE ${OpenCV_LIBS} ${TIFF_LIBRARIES} ${PNG_LIBRARIES}) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
