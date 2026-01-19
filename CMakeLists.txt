cmake_minimum_required(VERSION 4.1)
project(KernelConvolution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(is_release "$<CONFIG:Release>")
set(is_msvc "$<CXX_COMPILER_ID:MSVC>")
set(is_cuda "$<COMPILE_LANGUAGE:CUDA>")

add_subdirectory(cpuConvolution)
add_subdirectory(gpuConvolution)