#include <cuda_runtime.h>
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
    checkCUDAError(cudaMallocHost(&pageLockedBasePtr, pageLockedSlots * slotSizeInBytes), "An error has occurred while allocating page locked host memory");
    auto stagingSlotPtr{ static_cast<float*>(pageLockedBasePtr) };
    auto writingSlotPtr{ static_cast<float*>(pageLockedBasePtr) + slotSizeInChannels };
    std::vector<uint8_t> uintImage(slotSizeInChannels);

    void* buffersBasePtr{ nullptr };
    checkCUDAError(cudaMalloc(&buffersBasePtr, slotSizeInBytes * (inputSlots + outputSlots)), "An error has occurred while allocating device memory");
    InputBuffer inputBuffer{ static_cast<float*>(buffersBasePtr), inputSlots, slotSizeInChannels };
    OutputBuffer outputBuffer{ static_cast<float*>(buffersBasePtr) + inputSlots * slotSizeInChannels, outputSlots, slotSizeInChannels };

    const int tasksCount{ static_cast<int>(tasks.size()) };
    CudaTimer timer{ tasks.size(), enableStats, blockSize, inputSlots, outputSlots };
    timer.startingProgram();
    for (const auto& task : tasks) {
        const bool wasImageLoaded{ inputBuffer.isImageLoaded(task.image) };
        if (!wasImageLoaded) {
            loadImageToGPU(hostDeviceStream, inputBuffer, task.image, stagingSlotPtr, timer);
        }
        auto& inputSlot{ inputBuffer.getImageSlot(task.image) };
        auto& outputSlot{ outputBuffer.getAvailableSlot() };
        const auto taskInfo{ getDetailedTaskInfo(task, filters, filtersOffsets, outputFolder, tasksCount) };
        if (wasImageLoaded) timer.startLoadingImageEvent(convolutionStream);
        convoluteImage(convolutionStream, inputSlot, outputSlot, blockSize, taskInfo, timer);
        writeImage(deviceHostStream, outputSlot, taskInfo, writingSlotPtr, uintImage.data(), timer);
    }

    checkCUDAError(cudaDeviceSynchronize(), "An error occurred while waiting for the device to finish");
    timer.endingProgram();
    timer.writeLog(outputFolder / "log.txt");
    std::cout << std::endl;
    for (auto stream : { hostDeviceStream, convolutionStream, deviceHostStream }) {
        cudaStreamDestroy(stream);
    }
    cudaFreeHost(pageLockedBasePtr);
    cudaFree(buffersBasePtr);

    return 0;
}