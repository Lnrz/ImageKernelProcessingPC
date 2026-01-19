cmake_minimum_required(VERSION 4.1)
project(ImageKernelProcessing CXX)
project(ImageKernelProcessingCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

add_executable(ImageKernelProcessing src/main.cpp
        src/image.h
        src/image.cpp
        src/taskLoader.cpp
        src/taskLoader.h
        src/convolution.cpp
        src/convolution.h
        src/utilities.cpp
        src/utilities.h
        src/timer.cpp
        src/timer.h)

add_executable(ImageKernelProcessingCuda src/mainCuda.cu
        src/image.h
        src/image.cpp
        src/convolution.cu
        src/convolution.cuh
        src/buffer.cu
        src/buffer.cuh
        src/utilities.cu
        src/utilities.cuh
        src/taskLoader.cu
        src/taskLoader.cuh
        src/timer.cu
        src/timer.cuh
        src/stream.cu
        src/stream.cuh)

set_target_properties(ImageKernelProcessingCuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native)

add_subdirectory(lib/highway-1.3.0 EXCLUDE_FROM_ALL)

target_link_libraries(ImageKernelProcessing PRIVATE hwy)
target_include_directories(ImageKernelProcessing PRIVATE lib/stb_image)

target_include_directories(ImageKernelProcessingCuda PRIVATE lib/stb_image)

set(is_release "$<CONFIG:Release>")
set(is_msvc "$<CXX_COMPILER_ID:MSVC>")
set(is_cuda "$<COMPILE_LANGUAGE:CUDA>")

target_compile_options(ImageKernelProcessing PRIVATE
    $<$<AND:${is_release},${is_msvc}>:/O2 /Ob3 /fp:fast /arch:AVX2 /Gv /favor:INTEL64>)

target_compile_options(ImageKernelProcessingCuda PRIVATE
    $<${is_cuda}:-rdc=true -dlto -restrict -lineinfo -src-in-ptx>
    $<$<AND:${is_cuda},${is_msvc}>:-Xcompiler=/Zc:preprocessor>
    $<$<AND:${is_cuda},${is_release}>:--use_fast_math --extra-device-vectorization>
    $<$<AND:${is_cuda},${is_release},${is_msvc}>:-Xcompiler=/O2,/Ob3,/fp:fast,/arch:AVX2,/favor:INTEL64>)

target_link_options(ImageKernelProcessingCuda PRIVATE
    $<DEVICE_LINK:-dlto>)