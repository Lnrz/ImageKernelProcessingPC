#ifndef IMAGEKERNELPROCESSING_CONVOLUTION_H
#define IMAGEKERNELPROCESSING_CONVOLUTION_H

struct ConvolutionData {
    const float* inPtr{ nullptr };
    const float* coefPtr{ nullptr };
    float* outPtr{ nullptr };
    int rowSize{ 0 };
    int rowNum{ 0 };
    int channels{ -1 };
    int halfSize{ -1 };
};

size_t getCPULanes();

void kernelConvolution(const ConvolutionData& data);

#endif //IMAGEKERNELPROCESSING_CONVOLUTION_H