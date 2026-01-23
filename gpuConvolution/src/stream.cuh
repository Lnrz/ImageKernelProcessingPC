#ifndef IMAGEKERNELPROCESSINGCUDA_STREAM_CUH
#define IMAGEKERNELPROCESSINGCUDA_STREAM_CUH

#include <cuda.h>
#include "taskLoader.cuh"
#include "buffer.cuh"
#include "image.h"
#include "timer.cuh"

enum LoadFlag : uint32_t {
    LoadFlag_Empty = 0,
    LoadFlag_ImageLoaded = 1
};

enum WriteFlag : uint32_t {
    WriteFlag_Empty = 0,
    WriteFlag_ImageWritten = 1
};

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

void loadImagesToStagingBuffer(float* stagingBuffer, volatile uint32_t* loadFlag, const std::vector<Task>& tasks, CudaTimer& timer);

void loadImageToGPU(cudaStream_t stream, InputBuffer& buffer, const std::shared_ptr<Image>& image, float* stagingPtr, CUdeviceptr loadFlag);

DetailedTaskInfo getDetailedTaskInfo(const Task& task, const std::vector<Filter>& filters, const std::vector<int>& offsets);

void convoluteImageOnGPU(cudaStream_t stream, InputBufferSlot& inSlot, OutputBufferSlot& outSlot, dim3 blockSize, const DetailedTaskInfo& info, CudaTimer& timer);

void writeImagesToDisk(float* floatWriteBuffer, uint8_t* uintWriteBuffer, volatile uint32_t* writeFlag, const std::vector<Task>& tasks, const std::vector<Filter>& filters, const std::filesystem::path& outputFolder, CudaTimer& timer);

void writeImageFromGPU(cudaStream_t stream, OutputBufferSlot& outSlot, const DetailedTaskInfo& info, float* writingSlot, CUdeviceptr writeFlag);

#endif //IMAGEKERNELPROCESSINGCUDA_STREAM_CUH