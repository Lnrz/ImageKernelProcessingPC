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