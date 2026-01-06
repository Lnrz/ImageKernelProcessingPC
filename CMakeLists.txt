cmake_minimum_required(VERSION 4.1)
project(ImageKernelProcessing)

set(CMAKE_CXX_STANDARD 20)

add_executable(ImageKernelProcessing src/main.cpp
        src/image.h
        src/image.cpp
        src/taskLoader.cpp
        src/taskLoader.h
        src/convolution.cpp
        src/convolution.h)

add_subdirectory(lib/highway-1.3.0 EXCLUDE_FROM_ALL)

target_link_libraries(ImageKernelProcessing PRIVATE hwy)
target_include_directories(ImageKernelProcessing PRIVATE lib/stb_image)

set(is_release "$<CONFIG:Release>")
set(is_msvc "$<CXX_COMPILER_ID:MSVC>")

target_compile_options(ImageKernelProcessing PRIVATE
    $<$<AND:${is_release},${is_msvc}>:/O2 /Ob3 /fp:fast /arch:AVX2 /Gv /favor:INTEL64>)