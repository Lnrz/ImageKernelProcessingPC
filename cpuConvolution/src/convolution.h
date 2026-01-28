#ifndef IMAGEKERNELPROCESSING_CONVOLUTION_H
#define IMAGEKERNELPROCESSING_CONVOLUTION_H

// Struct storing the necessary data to compute the convolution.
//
// inPtr is the pointer to the start of the padded input image data.
// coefPtr is the pointer to the start of the filter data.
// outPtr is the pointer to the start of the output image storage.
// rowSize is the number of channels in a row of the output image.
// rowNum is the height of the output image.
// channels is the number of channels in the images.
// halfSize is the half size of the filter.
struct ConvolutionData {
    const float* inPtr{ nullptr };
    const float* coefPtr{ nullptr };
    float* outPtr{ nullptr };
    int rowSize{ 0 };
    int rowNum{ 0 };
    int channels{ -1 };
    int halfSize{ -1 };
};

// Return the number of float lanes available.
size_t getCPUFloatLanes();

// Perform the convolution specified by data using scalar instructions.
//
// Expect the padded input image and the filter to be stored in row major order.
// The output image will be stored in row major order.
void scalarKernelConvolution(const ConvolutionData& data);

// Perform the convolution specified by data using vector instructions.
//
// Expect the padded input image and the filter to be stored in row major order.
// The output image will be stored in row major order.
void kernelConvolution(const ConvolutionData& data);

#endif //IMAGEKERNELPROCESSING_CONVOLUTION_H