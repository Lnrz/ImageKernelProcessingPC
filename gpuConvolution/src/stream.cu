#include "stream.cuh"
#include <nvtx3/nvtx3.hpp>
#include <syncstream>
#include "convolution.cuh"
#include "utilities.cuh"

void loadImagesToStagingBuffer(float *stagingBuffer, volatile uint32_t *loadFlag, const std::vector<std::shared_ptr<Image>> &images) {
    for (const auto& image : images) {
        while (*loadFlag != LoadFlag_Empty) {}

        nvtx3::scoped_range loadingMarker{ "Loading image data to staging buffer" };
        image->load();
        const auto imageSize{ static_cast<size_t>(image->getWidth() * image->getHeight() * image->getChannels()) };
        std::copy_n(image->data(), imageSize, stagingBuffer);
        *loadFlag = LoadFlag_ImageLoaded;
        image->unload();
    }
}

void loadImageToGPU(cudaStream_t stream, InputBuffer& buffer, const std::shared_ptr<Image>& image, float* stagingPtr, CUdeviceptr loadFlag, CudaTimer& timer) {
    auto& inputSlot{ buffer.getAvailableSlot() };
    inputSlot.image = image;
    inputSlot.stagingPtr = stagingPtr;

    checkCUDAError(cuStreamWaitValue32(stream, loadFlag, LoadFlag_ImageLoaded, CU_STREAM_WAIT_VALUE_EQ),
        "An error occurred while issuing a wait on the loadFlag");
    timer.startLoadingImageEvent(stream);

    const size_t inputImageSize{ sizeof(float) * image->getWidth() * image->getHeight() * image->getChannels() };
    checkCUDAError(cudaMemcpyAsync(inputSlot.ptr, stagingPtr, inputImageSize, cudaMemcpyHostToDevice, stream),
        "An error occurred while issuing an async copy from the staging slot to an input slot");

    checkCUDAError(cuStreamWriteValue32(stream, loadFlag, LoadFlag_Empty, CU_STREAM_WRITE_VALUE_DEFAULT),
        "An error occurred while issuing a write to the loadFlag");

    checkCUDAError(cudaEventRecord(inputSlot.transferComplete, stream),
        "An error occurred while recording the transferComplete event of an input slot");
}



DetailedTaskInfo getDetailedTaskInfo(const Task& task, const std::vector<Filter>& filters, const std::vector<int>& offsets, const std::filesystem::path& outputFolder, const int tasksCount) {
    const auto inputWidth{ task.image->getWidth() };
    const auto inputHeight{ task.image->getHeight() };
    const auto filterIndex{ static_cast<FilterTypeInt>(task.filter) };
    const auto halfSize{ filters[filterIndex].halfSize };
    const auto isPaddingNone{ task.padding == PaddingMode::None };
    return {
        .inputImageWidth = inputWidth,
        .inputImageHeight = inputHeight,
        .channels = task.image->getChannels(),
        .outputImageWidth = isPaddingNone? inputWidth - 2 * halfSize : inputWidth,
        .outputImageHeight = isPaddingNone? inputHeight - 2 * halfSize : inputHeight,
        .filterHalfSize = halfSize,
        .filterOffset = offsets[filterIndex],
        .padding = task.padding
    };
}



void convoluteImage(cudaStream_t stream, InputBufferSlot& inSlot, OutputBufferSlot& outSlot, dim3 blockSize, const DetailedTaskInfo& info, CudaTimer& timer) {
    checkCUDAError(cudaStreamWaitEvent(stream, inSlot.transferComplete),
        "An error occurred while issuing a wait on the transferComplete event of an input slot");

    const dim3 gridSize{
        (info.outputImageWidth * info.channels + blockSize.x - 1) / blockSize.x,
        (info.outputImageHeight + blockSize.y - 1) / blockSize.y
    };
    const int channelsPerLoad{ static_cast<int>(blockSize.x * blockSize.y) };
    const int cacheRowSize{ static_cast<int>(blockSize.x + 2 * info.filterHalfSize * info.channels) };
    const int cacheHeight{ static_cast<int>(blockSize.y + 2 * info.filterHalfSize) };
    const size_t sharedMemorySize{ sizeof(float) * cacheRowSize * cacheHeight };
    timer.startConvolutingImageEvent(stream);
    cudaKernelConvolution<<<gridSize, blockSize, sharedMemorySize, stream>>>({
        .input = inSlot.ptr,
        .output = outSlot.ptr,
        .inputImageWidth = info.inputImageWidth,
        .inputImageRowSize = info.inputImageWidth * info.channels,
        .inputImageHeight = info.inputImageHeight,
        .channels = info.channels,
        .outputImageRowSize = info.outputImageWidth * info.channels,
        .outputImageHeight = info.outputImageHeight,
        .padding = info.padding,
        .filterOffset = info.filterOffset,
        .halfSize = info.filterHalfSize,
        .kernelSize = 2 * info.filterHalfSize + 1,
        .cacheRowSize = cacheRowSize,
        .cacheHeight = cacheHeight,
        .channelsPerLoad = channelsPerLoad,
        .loadingSteps = static_cast<int>((sharedMemorySize + channelsPerLoad - 1) / channelsPerLoad)
    });

    checkCUDAError(cudaEventRecord(inSlot.executionFinished, stream),
        "An error occurred while recording the executionFinished event of an input slot");
    checkCUDAError(cudaEventRecord(outSlot.executionFinished, stream),
        "An error occurred while recording the executionFinished event of an output slot");
    timer.endConvolutingImageEvent(stream);
}


void writeImagesToDisk(float* floatWriteBuffer, uint8_t* uintWriteBuffer, volatile uint32_t* writeFlag, const std::vector<Task>& tasks, const std::vector<Filter>& filters, const std::filesystem::path& outputFolder) {
    const auto tasksCount{ tasks.size() };
    for (uint32_t i{ 1 }; const auto& task : tasks) {
        const auto halfSize{ filters[static_cast<FilterTypeInt>(task.filter)].halfSize };
        const auto outputImageWidth{ task.padding == PaddingMode::None ?
            task.image->getWidth() - 2 * halfSize : task.image->getWidth() };
        const auto outputImageHeight{ task.padding == PaddingMode::None ?
            task.image->getHeight() - 2 * halfSize : task.image->getHeight() };
        const auto imageSize{ outputImageWidth * outputImageHeight * task.image->getChannels() };

        while (*writeFlag != WriteFlag_ImageWritten) {}

        nvtx3::scoped_range writingMarker{ "Writing image to disk" };
        floatImageToUIntImage(
            std::ranges::subrange{ floatWriteBuffer, floatWriteBuffer + imageSize },
            std::ranges::subrange{ uintWriteBuffer, uintWriteBuffer + imageSize }
        );
        writeImage((outputFolder / task.image->getPath().stem()).string()
                + "-" + getStringFromFilterType(task.filter)
                + "-" + getStringFromPaddingMode(task.padding) + ".jpg",
                outputImageWidth, outputImageHeight, task.image->getChannels(),
                uintWriteBuffer);

        *writeFlag = WriteFlag_Empty;
        std::osyncstream{ std::cout } << i << "/" << tasksCount << " images processed\r";
        i++;
    }
}

void writeImage(cudaStream_t stream, OutputBufferSlot& outSlot, const DetailedTaskInfo& info, float* writingSlot, CUdeviceptr writeFlag, CudaTimer& timer) {
    checkCUDAError(cudaStreamWaitEvent(stream, outSlot.executionFinished),
        "An error occurred while issuing a wait on the executionFinished event of an output slot");
    checkCUDAError(cuStreamWaitValue32(stream, writeFlag, WriteFlag_Empty, CU_STREAM_WAIT_VALUE_EQ),
        "An error occurred while issuing a wait on the write flag");

    const size_t outputImageSize{ sizeof(float) * info.outputImageWidth * info.outputImageHeight * info.channels };
    checkCUDAError(cudaMemcpyAsync(writingSlot, outSlot.ptr, outputImageSize, cudaMemcpyDeviceToHost, stream),
        "An error occurred while issuing an async copy from an output slot to the writing slot");
    checkCUDAError(cudaEventRecord(outSlot.transferComplete, stream),
        "An error occurred while recording the tranferComplete event of an output slot");

    checkCUDAError(cuStreamWriteValue32(stream, writeFlag, WriteFlag_ImageWritten, CU_STREAM_WRITE_VALUE_DEFAULT),
        "An error occurred while issuing a write to the writeFlag");

    timer.endWritingImageEvent(stream);
}