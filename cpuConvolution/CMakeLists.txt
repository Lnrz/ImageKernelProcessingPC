add_executable(cpuKernelConvolution src/main.cpp
        src/taskLoader.cpp
        src/taskLoader.h
        src/convolution.cpp
        src/convolution.h
        src/utilities.cpp
        src/utilities.h
        src/timer.cpp
        src/timer.h
        ../common/image.cpp
        ../common/image.h)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/highway-1.3.0/cmake")
add_subdirectory(lib/highway-1.3.0 EXCLUDE_FROM_ALL)

target_include_directories(cpuKernelConvolution
        PRIVATE ../common)

target_link_libraries(cpuKernelConvolution
        PRIVATE hwy)

target_compile_options(cpuKernelConvolution PRIVATE
        $<$<AND:${is_release},${is_msvc}>:/O2 /Ob3 /fp:fast /arch:AVX2 /Qvec-report:1 /Gv /favor:INTEL64>)