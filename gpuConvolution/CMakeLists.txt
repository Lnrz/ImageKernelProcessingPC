add_executable(gpuKernelConvolution src/main.cu
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
        src/stream.cuh
        ../common/image.cpp
        ../common/image.h)

set_target_properties(gpuKernelConvolution
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpuKernelConvolution
        PRIVATE CUDA::cuda_driver
        PRIVATE "${OpenCV_LIBS}")

target_include_directories(gpuKernelConvolution
        PRIVATE ../common
        PRIVATE "${OpenCV_INCLUDE_DIRS}")

target_compile_options(gpuKernelConvolution PRIVATE
        $<${is_cuda}:-rdc=true -dlto -restrict -lineinfo -src-in-ptx>
        $<$<AND:${is_cuda},${is_msvc}>:-Xcompiler=/Zc:preprocessor>
        $<$<AND:${is_cuda},${is_release}>:--use_fast_math --extra-device-vectorization>
        $<$<AND:${is_cuda},${is_release},${is_msvc}>:-Xcompiler=/O2,/Ob3,/fp:fast,/arch:AVX2,/favor:INTEL64>)

target_link_options(gpuKernelConvolution PRIVATE
        $<DEVICE_LINK:-dlto>)