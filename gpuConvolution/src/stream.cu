#include "stream.cuh"
#include <syncstream>
#include "convolution.cuh"



void loadImageToStagingMemory(void* userData) {
    const auto inputInfo{ static_cast<InputBufferSlot*>(userData) };
    const auto& image{ inputInfo->image };
    image->load();
    const auto imageSize{ static_cast<size_t>(image->getWidth() * image->getHeight() * image->getChannels()) };
    std::copy_n(image->data(), imageSize, inputInfo->stagingPtr);
    image->unload();
}

void loadImageToGPU(cudaStream_t stream, InputBuffer& buffer, const std::shared_ptr<Image>& image, float* stagingPtr, CudaTimer& timer) {
    auto& inputSlot{ buffer.getAvailableSlot() };
    inputSlot.image = image;
    inputSlot.stagingPtr = stagingPtr;
    timer.startLoadingImageEvent(stream);
    cudaLaunchHostFunc(stream, loadImageToStagingMemory, &inputSlot);

    const size_t inputImageSize{ sizeof(float) * image->getWidth() * image->getHeight() * image->getChannels() };
    cudaMemcpyAsync(inputSlot.ptr, stagingPtr, inputImageSize, cudaMemcpyHostToDevice, stream);


    cudaEventRecord(inputSlot.transferComplete, stream);
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
        .padding = task.padding,
        .outputPath = (outputFolder / task.image->getPath().stem()).string()
        + "-" + getStringFromFilterType(task.filter)
        + "-" + getStringFromPaddingMode(task.padding)
        + ".jpg",
        .tasksCount = tasksCount
    };
}



void convoluteImage(cudaStream_t stream, InputBufferSlot& inSlot, OutputBufferSlot& outSlot, dim3 blockSize, const DetailedTaskInfo& info, CudaTimer& timer) {
    cudaStreamWaitEvent(stream, inSlot.transferComplete);

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

    cudaEventRecord(inSlot.executionFinished, stream);
    cudaEventRecord(outSlot.executionFinished, stream);
    timer.endConvolutingImageEvent(stream);
}



void writeImageCallback(void* userData) {
    static std::atomic_uint32_t i { 1 };

    const auto outputInfo{ static_cast<OutputInfo*>(userData) };
    const size_t imageSize{ static_cast<size_t>(outputInfo->width * outputInfo->height * outputInfo->channels) };
    floatImageToUIntImage(
        std::ranges::subrange{ outputInfo->floatData, outputInfo->floatData + imageSize },
        std::ranges::subrange{ outputInfo->uintData, outputInfo->uintData + imageSize }
    );
    writeImage(outputInfo->path,
            outputInfo->width, outputInfo->height, outputInfo->channels,
            outputInfo->uintData);

    const auto prev{ i++ };
    std::osyncstream{ std::cout } << prev << "/" << outputInfo->tasksCount << " images processed\r";
}

void writeImage(cudaStream_t stream, OutputBufferSlot& outSlot, const DetailedTaskInfo& info, float* writingSlot, uint8_t* uintImage, CudaTimer& timer) {
    cudaStreamWaitEvent(stream, outSlot.executionFinished);

    outSlot.outputInfo = {
        .path = std::move(info.outputPath),
        .floatData = writingSlot,
        .uintData = uintImage,
        .width = info.outputImageWidth,
        .height = info.outputImageHeight,
        .channels = info.channels,
        .tasksCount = info.tasksCount
    };
    const size_t outputImageSize{ sizeof(float) * info.outputImageWidth * info.outputImageHeight * info.channels };
    cudaMemcpyAsync(writingSlot, outSlot.ptr, outputImageSize, cudaMemcpyDeviceToHost, stream);

    cudaLaunchHostFunc(stream, writeImageCallback, &outSlot.outputInfo);

    cudaEventRecord(outSlot.transferComplete, stream);
    timer.endWritingImageEvent(stream);
}