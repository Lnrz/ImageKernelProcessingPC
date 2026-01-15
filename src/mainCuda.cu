#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <array>
#include "convolution.cuh"
#include "image.h"
#include "taskLoader.h"
#include "utilities.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct OutputImageInfo {
    int width{};
    int height{};
    int channels{};
    std::filesystem::path inputImageName{};
    PaddingMode padding{};
    FilterType filter{};
};

int main(int argc ,char* argv[]) {
    if (argc < 3) explainProgram();
    const std::filesystem::path outputFolder{ argv[2] };

    auto [images, tasks ] = loadTasks(argv[1]);
    std::ranges::sort(tasks,
        [](const Task& t1, const Task& t2) {
            return t1.image->getPath() < t2.image->getPath();
        }
    );

    // put filters data into constant memory
    const auto filters{ getFilters() };
    std::vector<float> filtersData;
    filtersData.reserve(getFiltersSize());
    std::vector<size_t> filtersOffsets;
    filtersOffsets.reserve(static_cast<FilterTypeInt>(FilterType::Num));
    filtersOffsets.push_back(0);
    for (const auto& filter : getFilters()) {
        filtersData.insert(filtersData.cend(), filter.data.begin(), filter.data.end());
        if (filtersOffsets.size() != static_cast<FilterTypeInt>(FilterType::Num)) {
            filtersOffsets.push_back(filtersOffsets.back() + filter.data.size());
        }
    }
    updateFilters(filtersData.data());

    // make buffers
    size_t biggestImageSize{};
    for (const auto& image : images | std::views::values) {
        size_t imageSize{ static_cast<size_t>(image->getWidth() * image->getHeight() * image->getChannels()) };
        if (imageSize > biggestImageSize) {
            biggestImageSize = imageSize;
        }
    }
    thrust::host_vector<float> hostFloatBuffer{ biggestImageSize, thrust::no_init };
    thrust::host_vector<uint8_t> hostUIntBuffer{ biggestImageSize, thrust::no_init };
    std::array deviceInputBuffers{
        thrust::device_vector<float>{ biggestImageSize, thrust::no_init },
        thrust::device_vector<float>{ biggestImageSize, thrust::no_init }
    };
    std::array deviceOutputBuffers{
        thrust::device_vector<float>{ biggestImageSize, thrust::no_init },
        thrust::device_vector<float>{ biggestImageSize, thrust::no_init }
    };
    constexpr uint32_t buffersNum{ deviceInputBuffers.size() };
    OutputImageInfo outputImageInfos[buffersNum];
    uint32_t inputBufferIndex{}, outputBufferIndex{};

    std::shared_ptr<Image> lastImage{};
    bool isFirstTask{ true };
    for (const auto& task : tasks) {
        if (task.image != lastImage) {
            lastImage = task.image;
            task.image->load();
            inputBufferIndex = (inputBufferIndex + 1) % buffersNum;
            deviceInputBuffers[inputBufferIndex].clear();
            deviceInputBuffers[inputBufferIndex].assign(task.image->data()
                , task.image->data() + task.image->getWidth() * task.image->getHeight() * task.image->getChannels());
        }
        outputBufferIndex = (outputBufferIndex + 1) % buffersNum;
        outputImageInfos[outputBufferIndex] = {
            .width = (task.padding == PaddingMode::None) ?
                        task.image->getWidth() - 2 * filters[static_cast<FilterTypeInt>(task.filter)].halfSize :
                        task.image->getWidth(),
            .height = (task.padding == PaddingMode::None) ?
                        task.image->getHeight() - 2 * filters[static_cast<FilterTypeInt>(task.filter)].halfSize :
                        task.image->getHeight(),
            .channels = task.image->getChannels(),
            .inputImageName = task.image->getPath().stem(),
            .padding = task.padding,
            .filter = task.filter
        };
        deviceOutputBuffers[outputBufferIndex].resize(outputImageInfos[outputBufferIndex].width *
            outputImageInfos[outputBufferIndex].height * outputImageInfos[outputBufferIndex].channels);

        cudaDeviceSynchronize();

        // launch kernel
        constexpr dim3 blockDim{ 32, 32 };
        const dim3 gridDim{ (outputImageInfos[outputBufferIndex].width * task.image->getChannels() + blockDim.x - 1) / blockDim.x,
                            (outputImageInfos[outputBufferIndex].height + blockDim.y - 1) / blockDim.y };
        const auto halfSize{ filters[static_cast<FilterTypeInt>(task.filter)].halfSize };
        const size_t sharedMemorySize{ (blockDim.x + 2 * halfSize * task.image->getChannels()) * (blockDim.y + 2 * halfSize) * sizeof(float) };
        cudaKernelConvolution<<<gridDim, blockDim, sharedMemorySize>>>({
            .input = thrust::raw_pointer_cast(deviceInputBuffers[inputBufferIndex].data()),
            .output = thrust::raw_pointer_cast(deviceOutputBuffers[outputBufferIndex].data()),
            .inputImageWidth = task.image->getWidth(),
            .inputImageHeight = task.image->getHeight(),
            .channels = task.image->getChannels(),
            .filterOffset = static_cast<int>(filtersOffsets[static_cast<FilterTypeInt>(task.filter)]),
            .halfSize = halfSize,
            .padding = task.padding
        });

        task.image->unload();
        if (isFirstTask) {
            isFirstTask = false;
            continue;
        }

        // prepare to output
        hostFloatBuffer.clear();
        hostUIntBuffer.clear();


        const auto bufferToSaveIndex { (outputBufferIndex - 1) % buffersNum };
        hostFloatBuffer.assign(deviceOutputBuffers[bufferToSaveIndex].begin()
                             , deviceOutputBuffers[bufferToSaveIndex].end());


        // convert to uint
        floatImageToUIntImage(hostFloatBuffer, hostUIntBuffer);

        // save
        const auto& outputImageInfo{ outputImageInfos[bufferToSaveIndex] };


        auto outImagePath{ outputFolder / outputImageInfo.inputImageName };
        outImagePath += "-" + getStringFromFilterType(outputImageInfo.filter)
                      + "-" + getStringFromPaddingMode(outputImageInfo.padding) + ".jpg";
        stbi_write_jpg(outImagePath.string().c_str(),
            outputImageInfo.width, outputImageInfo.height, outputImageInfo.channels,
            hostUIntBuffer.data(), 100);
    }
    // prepare to output
    hostFloatBuffer.clear();
    hostUIntBuffer.clear();
    const auto bufferToSaveIndex { outputBufferIndex };

    cudaDeviceSynchronize();

    hostFloatBuffer.assign(deviceOutputBuffers[bufferToSaveIndex].begin()
                         , deviceOutputBuffers[bufferToSaveIndex].end());

    // convert to uint
    floatImageToUIntImage(hostFloatBuffer, hostUIntBuffer);

    // save
    const auto& outputImageInfo{ outputImageInfos[bufferToSaveIndex] };

    auto outImagePath{ outputFolder / outputImageInfo.inputImageName };
    outImagePath += "-" + getStringFromFilterType(outputImageInfo.filter)
                  + "-" + getStringFromPaddingMode(outputImageInfo.padding) + ".jpg";
    stbi_write_jpg(outImagePath.string().c_str(),
            outputImageInfo.width, outputImageInfo.height, outputImageInfo.channels,
            hostUIntBuffer.data(), 100);

    return 0;
}