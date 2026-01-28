#ifndef IMAGEKERNELPROCESSINGCUDA_STREAM_CUH
#define IMAGEKERNELPROCESSINGCUDA_STREAM_CUH

#include <cuda.h>
#include "taskLoader.cuh"
#include "buffer.cuh"
#include "image.h"
#include "timer.cuh"

// Enum of the valid values for the load flag
enum LoadFlag : uint32_t {
    LoadFlag_Empty = 0,
    LoadFlag_ImageLoaded = 1
};

// Enum of the valid values for the write flag
enum WriteFlag : uint32_t {
    WriteFlag_Empty = 0,
    WriteFlag_ImageWritten = 1
};

// Struct containing data for the processing of a task.
//
// filterOffset refers to the offset of the filter to use in GPU constant memory.
struct DetailedTaskInfo {
    int inputImageWidth;
    int inputImageHeight;
    int channels;
    int outputImageWidth;
    int outputImageHeight;
    int filterHalfSize;
    int filterOffset;
    PaddingMode padding;
};

// Main function of the thread managing image loading and unloading in CPU memory.
//
// stagingBuffer is the pointer to the page locked memory in which images will be loaded.
// loadFlag is a flag in page locked memory for communicating with the stream that loads images to GPU.
// tasks is the vector of tasks to process.
// timer is a reference to the timer used for timing.
void loadImagesToStagingBuffer(float* stagingBuffer, volatile uint32_t* loadFlag, const std::vector<Task>& tasks, CudaTimer& timer);

// Schedule on stream the transfer of image from stagingPtr to one of buffer's slots.
//
// stagingPtr is the pointer to the page locked memory where the image to load will be stored.
// loadFlag is a flag in page locked memory used to synchronize the stream with the loading thread.
void loadImageToGPU(cudaStream_t stream, InputBuffer& buffer, const std::shared_ptr<Image>& image, float* stagingPtr, CUdeviceptr loadFlag);

// Return DetailedTaskInfo from the provided data.
DetailedTaskInfo getDetailedTaskInfo(const Task& task, const std::vector<Filter>& filters, const std::vector<int>& offsets);

// Schedule on stream the convolution of the image stored in inSlot and store the result in outSlot.
//
// blockSize is the block size to use for kernel execution.
// info contains the data needed for the convolution.
// timer is a reference to the timer used for timing.
void convoluteImageOnGPU(cudaStream_t stream, InputBufferSlot& inSlot, OutputBufferSlot& outSlot, dim3 blockSize, const DetailedTaskInfo& info, CudaTimer& timer);

// Main function of the thread managing image saving.
//
// floatWriteBuffer is the pointer to the page locked memory in which images to be saved are stored.
// uintWriteBuffer is the pointer to CPU memory used to store the processed image before saving it.
// writeFlag is a flag in page locked memory for communicating with the stream that transfer images from GPU.
// tasks is the vector of tasks to process.
// filters is a vector of all available filters.
// outputFolder is the path to the output folder where images will be saved.
// timer is a reference to the timer used for timing.
void writeImagesToDisk(float* floatWriteBuffer, uint8_t* uintWriteBuffer, volatile uint32_t* writeFlag, const std::vector<Task>& tasks, const std::vector<Filter>& filters, const std::filesystem::path& outputFolder, CudaTimer& timer);

// Schedule on stream the transfer of the image stored in outSlot to writingSlot.
//
// writingSlot is the pointer to the page locked memory where images to be saved are transferred.
// info contains the data needed for the transfer.
// writeFlag is a flag in page locked memory used to synchronize the stream with the writing thread.
void writeImageFromGPU(cudaStream_t stream, OutputBufferSlot& outSlot, const DetailedTaskInfo& info, float* writingSlot, CUdeviceptr writeFlag);

#endif //IMAGEKERNELPROCESSINGCUDA_STREAM_CUH