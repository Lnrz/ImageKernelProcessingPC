#ifndef IMAGEKERNELPROCESSINGCUDA_CONVOLUTION_CUH
#define IMAGEKERNELPROCESSINGCUDA_CONVOLUTION_CUH

#include "image.h"

extern __constant__ float deviceFilters[getFiltersSize()];

struct CudaConvolutionData {
    float *input{}, *output{};
    int inputImageWidth{};
    int inputImageRowSize{};
    int inputImageHeight{};
    int channels{};
    int outputImageRowSize{};
    int outputImageHeight{};
    PaddingMode padding{ PaddingMode::Invalid };
    int filterOffset{};
    int halfSize{};
    int kernelSize{};
    int cacheRowSize{};
    int cacheHeight{};
    int channelsPerLoad{};
    int loadingSteps{};
};

__global__ void cudaKernelConvolution(CudaConvolutionData data);

#endif //IMAGEKERNELPROCESSINGCUDA_CONVOLUTION_CUH