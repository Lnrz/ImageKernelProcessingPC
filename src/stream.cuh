#ifndef IMAGEKERNELPROCESSINGCUDA_STREAM_CUH
#define IMAGEKERNELPROCESSINGCUDA_STREAM_CUH

#include "taskLoader.cuh"
#include "buffer.cuh"
#include "image.h"
#include "timer.cuh"

struct DetailedTaskInfo {
    int inputImageWidth;
    int inputImageHeight;
    int channels;
    int outputImageWidth;
    int outputImageHeight;
    int filterHalfSize;
    int filterOffset;
    PaddingMode padding;
    std::string outputPath;
    int tasksCount;
};

void loadImageToGPU(cudaStream_t stream, InputBuffer& buffer, const std::shared_ptr<Image>& image, float* stagingPtr, CudaTimer& timer);

DetailedTaskInfo getDetailedTaskInfo(const Task& task, const std::vector<Filter>& filters, const std::vector<int>& offsets, const std::filesystem::path& outputFolder, int tasksCount);

void convoluteImage(cudaStream_t stream, InputBufferSlot& inSlot, OutputBufferSlot& outSlot, dim3 blockSize, const DetailedTaskInfo& info, CudaTimer& timer);

void writeImage(cudaStream_t stream, OutputBufferSlot& outSlot, const DetailedTaskInfo& info, float* writingSlot, uint8_t* uintImage, CudaTimer& timer);

#endif //IMAGEKERNELPROCESSINGCUDA_STREAM_CUH