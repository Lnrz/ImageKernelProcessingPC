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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/highway/cmake")
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
add_subdirectory(lib/highway EXCLUDE_FROM_ALL)

target_include_directories(cpuKernelConvolution
        PRIVATE ../common)

target_link_libraries(cpuKernelConvolution
        PRIVATE hwy)

target_compile_options(cpuKernelConvolution PRIVATE
        $<$<AND:${is_release},${is_msvc}>:/O2 /Ob3 /fp:fast /arch:AVX2 /Qvec-report:1 /Gv /favor:INTEL64>)