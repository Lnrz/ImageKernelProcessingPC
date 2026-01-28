#ifndef IMAGEKERNELPROCESSINGCUDA_CONVOLUTION_CUH
#define IMAGEKERNELPROCESSINGCUDA_CONVOLUTION_CUH

#include "image.h"

extern __constant__ float deviceFilters[getFiltersSize()];

// Struct containing the data necessary for convolution.
//
// input is the pointer to the input image data in GPU memory.
// output is the pointer to the output image storage in GPU memory.
// inputImageWidth is the width of the input image.
// inputImageRowSize is the number of channels in a row of the input image.
// inputImageHeight is the height of the input image.
// channels is the number of channels of the images.
// outputImageRowSize is the number of channels in a row of the output image.
// outputImageHeight is the height of the output image.
// padding is the padding to apply to the input image.
// filterOffset is the offset at which the filter coefficients can be found in constant memory.
// halfSize is the filter half size.
// kernelSize is the filter size.
// cacheRowSize is the number of channels in a row of the neighborhood of the block.
// cacheHeight is the height of the neighborhood of the block.
// channelsPerLoad is the number of channels that are loaded in shared memory for each step.
// loadingSteps is the number of steps necessary to load the neighborhood of the block into shared memory
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

// Perform the convolution specified by data.
//
// Expect the input image and the filters to be stored in row major order.
// Store the output image in row major order.
__global__ void cudaKernelConvolution(CudaConvolutionData data);

#endif //IMAGEKERNELPROCESSINGCUDA_CONVOLUTION_CUH