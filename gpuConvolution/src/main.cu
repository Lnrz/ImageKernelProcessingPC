#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include "convolution.cuh"
#include "image.h"
#include "taskLoader.cuh"
#include "utilities.cuh"
#include "buffer.cuh"
#include "timer.cuh"
#include "stream.cuh"


void loadFiltersToConstantMemory(const std::vector<Filter>& filters) {
    std::vector<float> filtersData;
    filtersData.reserve(getFiltersSize());
    for (const auto& filter : filters) {
        filtersData.insert(filtersData.cend(), filter.data.begin(), filter.data.end());
    }
    cudaMemcpyToSymbol(deviceFilters, filtersData.data(), getFiltersSize() * sizeof(float));
}

std::vector<int> getFiltersOffsets(const std::vector<Filter>& filters) {
    std::vector<int> offsets{ 0 };
    constexpr auto filtersNum{ static_cast<FilterTypeInt>(FilterType::Num) };
    offsets.reserve(filtersNum);

    for (const auto& filter : filters) {
        offsets.push_back(offsets.back() + static_cast<int>(filter.data.size()));
        if (offsets.size() == filtersNum) break;
    }

    return offsets;
}


int main(int argc ,char* argv[]) {
    if (argc < 3) explainProgram();
    const std::filesystem::path outputFolder{ argv[2] };
    auto [
        images,
        tasks,
        blockSize,
        inputSlots,
        outputSlots,
        enableStats
    ] = loadTasks(argv[1]);

    std::ranges::sort(tasks,
        [](const Task& t1, const Task& t2) {
            return t1.image->getPath() < t2.image->getPath();
        }
    );

    const auto filters{ getFilters() };
    loadFiltersToConstantMemory(filters);
    const auto filtersOffsets{ getFiltersOffsets(filters) };

    cudaStream_t hostDeviceStream, convolutionStream, deviceHostStream;
    for (auto stream : { &hostDeviceStream, &convolutionStream, &deviceHostStream }) {
        checkCUDAError(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking),
            "An error has occurred while initializing the streams");
    }

    const size_t slotSizeInBytes{
        std::ranges::max(images | std::views::values | std::views::transform(
            [](const std::shared_ptr<Image>& image) {
                    return static_cast<size_t>(image->getWidth() * image->getHeight() * image->getChannels()) * sizeof(float);
                }
    ))};
    const size_t slotSizeInChannels{ slotSizeInBytes / sizeof(float) };

    void* pageLockedBasePtr{ nullptr };
    constexpr size_t pageLockedSlots{ 2 };
    constexpr size_t flagsNum{ 2 };
    constexpr size_t flagSize{ 4 };
    checkCUDAError(cudaMallocHost(&pageLockedBasePtr, pageLockedSlots * slotSizeInBytes + flagsNum * flagSize)
        , "An error has occurred while allocating page locked host memory");
    auto stagingSlotPtr{ static_cast<float*>(pageLockedBasePtr) };
    auto writingSlotPtr{ static_cast<float*>(pageLockedBasePtr) + slotSizeInChannels };
    auto loadFlagPtr{ static_cast<uint32_t *>(pageLockedBasePtr) + pageLockedSlots * slotSizeInChannels };
    auto writeFlagPtr{ static_cast<uint32_t *>(pageLockedBasePtr) + pageLockedSlots * slotSizeInChannels + 1 };
    *loadFlagPtr = LoadFlag_Empty;
    *writeFlagPtr = WriteFlag_Empty;
    CUdeviceptr deviceLoadFlagPtr, deviceWriteFlagPtr;
    checkCUDAError(cuMemHostGetDevicePointer(&deviceLoadFlagPtr, loadFlagPtr, 0),
        "An error occurred while getting the device pointer to the load flag");
    checkCUDAError(cuMemHostGetDevicePointer(&deviceWriteFlagPtr, writeFlagPtr, 0),
        "An error occurred while getting the device pointer to the write flag");
    std::vector<uint8_t> uintImage(slotSizeInChannels);

    void* buffersBasePtr{ nullptr };
    checkCUDAError(cudaMalloc(&buffersBasePtr, slotSizeInBytes * (inputSlots + outputSlots)),
        "An error has occurred while allocating device memory");
    InputBuffer inputBuffer{ static_cast<float*>(buffersBasePtr), inputSlots, slotSizeInChannels };
    OutputBuffer outputBuffer{ static_cast<float*>(buffersBasePtr) + inputSlots * slotSizeInChannels, outputSlots, slotSizeInChannels };

    CudaTimer timer{ tasks.size(), enableStats, blockSize, inputSlots, outputSlots };
    std::jthread imageLoaderThread{ loadImagesToStagingBuffer,
        stagingSlotPtr, loadFlagPtr, std::cref(tasks), std::ref(timer) };
    std::jthread imageWriterThread{ writeImagesToDisk,
        writingSlotPtr, uintImage.data(), writeFlagPtr, std::cref(tasks), std::cref(filters), std::cref(outputFolder), std::ref(timer) };

    timer.startingProgram();
    for (const auto& task : tasks) {
        if (!inputBuffer.isImageLoaded(task.image)) {
            loadImageToGPU(hostDeviceStream, inputBuffer, task.image, stagingSlotPtr, deviceLoadFlagPtr);
        }
        auto& inputSlot{ inputBuffer.getImageSlot(task.image) };
        auto& outputSlot{ outputBuffer.getSlot() };
        const auto taskInfo{ getDetailedTaskInfo(task, filters, filtersOffsets) };
        convoluteImageOnGPU(convolutionStream, inputSlot, outputSlot, blockSize, taskInfo, timer);
        writeImageFromGPU(deviceHostStream, outputSlot, taskInfo, writingSlotPtr, deviceWriteFlagPtr);
    }

    checkCUDAError(cudaDeviceSynchronize(), "An error occurred while waiting for the device to finish");
    imageWriterThread.join();
    timer.endingProgram();
    timer.writeLog(outputFolder);
    std::cout << std::endl;
    for (auto stream : { hostDeviceStream, convolutionStream, deviceHostStream }) {
        checkCUDAError(cudaStreamDestroy(stream), "An error occurred while destroying streams");
    }
    checkCUDAError(cudaFreeHost(pageLockedBasePtr), "An error occurred while freeing page locked memory");
    checkCUDAError(cudaFree(buffersBasePtr), "An error occurred while freeing GPU memory");

    return 0;
}