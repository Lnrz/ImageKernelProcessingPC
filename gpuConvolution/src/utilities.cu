#include "utilities.cuh"
#include <iostream>

void checkCUDAError(const cudaError_t error, const std::string_view msg) {
    if (error == cudaSuccess) [[likely]] return;
    std::cout << msg << std::endl;
    std::cout << ">" << cudaGetErrorString(error) << std::endl;
    exit(-1);
}

void checkCUDAError(const CUresult error, const std::string_view msg) {
    if (error == CUDA_SUCCESS) [[likely]] return;
    const char* errorString;
    cuGetErrorString(error, &errorString);
    std::cout << msg << std::endl;
    std::cout << ">" << errorString << std::endl;
    exit(-1);
}

[[noreturn]]
void explainProgram() {
    std::cout << R"(Program usage:
gpuKernelConvolution.exe inputFolder outputFolder

inputFolder:
    path to the folder containing the images to process and a tasks.txt file
outputFolder:
    path to the folder where to write the output images

The tasks.txt file contains the tasks specified as lines in the following format:
IMAGE image1 filter1_1:padding1_1
IMAGE image2 filter2_1:padding2_1 filter2_2:padding2_2
...

image is the name of the image to process, contained in the inputFolder
filter is the filter to apply to the image
padding is the padding to apply to the image
It is possible to specify more filters and paddings for the same image by separating them with a whitespace

The block size with which to process images can be specified in tasks.txt with:
BLOCK x y
By default it is (32,16)

The number of slots of the GPU input and output buffers can be specified in tasks.txt with:
SLOTS inNum outNum
By default they both have 2 slots

WARNING
Every slot is as big as the largest image
Therefore it's advised to process images of roughly the same size together in order not to waste GPU memory

To activate statistics write in tasks.txt the line:
STATS
The statistics will be written to a log.txt file inside the outputFolder.
Convolution times will be written in binary as float in convolutionTimes.bin inside the outputFolder
Task processing times will be written in binary as float in processingTimes.bin inside the outputFolder
)";
    exit(-1);
}