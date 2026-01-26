cmake_minimum_required(VERSION 4.1)
project(KernelConvolution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(is_release "$<CONFIG:Release>")
set(is_msvc "$<CXX_COMPILER_ID:MSVC>")
set(is_cuda "$<COMPILE_LANGUAGE:CUDA>")

set(OpenCV_DIR "common/lib/opencv-4.13.0-windows/build")
find_package(OpenCV REQUIRED)

add_subdirectory(cpuConvolution)
add_subdirectory(gpuConvolution)