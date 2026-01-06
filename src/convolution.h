#ifndef IMAGEKERNELPROCESSING_CONVOLUTION_H
#define IMAGEKERNELPROCESSING_CONVOLUTION_H

struct ConvolutionData {
    const float* inPtr{ nullptr };
    const float* coefPtr{ nullptr };
    float* outPtr{ nullptr };
    size_t rowSize{ 0 };
    size_t rowNum{ 0 };
    int channels{ -1 };
    int halfSize{ -1 };
};

void kernelConvolution(const ConvolutionData& data);

#endif //IMAGEKERNELPROCESSING_CONVOLUTION_H