# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
cmake_minimum_required(VERSION 3.16)
project(FreeTrace_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV QUIET) # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:30
if(OpenCV_FOUND)
    add_definitions(-DUSE_OPENCV)
endif()

add_executable(freetrace
    src/main.cpp
    src/image_pad.cpp
    src/regression.cpp
    src/cost_function.cpp
    src/localization.cpp # Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:30
)

target_include_directories(freetrace PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(freetrace PRIVATE ${OpenCV_LIBS})
# Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 13:20
