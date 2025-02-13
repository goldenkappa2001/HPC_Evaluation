cmake_minimum_required(VERSION 3.10)
project(SWE_Optimized LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA policy and a default architecture (adjust according to your GPU)
cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Gather .cpp, .cu, and benchmark files
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/src/benchmark.cpp")
file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")

# Create the executable with all sources (CPU, GPU, and Benchmark)
add_executable(swe_simulator ${SOURCES} ${CUDA_SOURCES})

# Link CUDA libraries
target_link_libraries(swe_simulator ${CUDA_LIBRARIES})
