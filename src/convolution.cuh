#ifndef IMAGEKERNELPROCESSINGCUDA_CONVOLUTION_CUH
#define IMAGEKERNELPROCESSINGCUDA_CONVOLUTION_CUH

#include "image.h"
#include <thrust/device_vector.h>

extern __constant__ float deviceFilters[getFiltersSize()];

void updateFilters(const float* data);

struct CudaConvolutionData {
    float *input{}, *output{};
    int inputImageWidth{};
    int inputImageHeight{};
    int channels{};
    int filterOffset{};
    int halfSize{};
    PaddingMode padding{ PaddingMode::Invalid };
};

__global__ void cudaKernelConvolution(CudaConvolutionData data);

#endif //IMAGEKERNELPROCESSINGCUDA_CONVOLUTION_CUH