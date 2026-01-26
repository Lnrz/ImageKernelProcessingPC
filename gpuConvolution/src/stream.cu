#include "stream.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <nvtx3/nvtx3.hpp>
#include <syncstream>
#include "convolution.cuh"
#include "utilities.cuh"

void loadImagesToStagingBuffer(float *stagingBuffer, volatile uint32_t *loadFlag, const std::vector<Task>& tasks, CudaTimer& timer) {
    std::shared_ptr<Image> prevImage{ nullptr };
    for (uint32_t i{ 0 }; const auto& task : tasks) {
        if (task.image != prevImage) {
            prevImage = task.image;
            while(*loadFlag != LoadFlag_Empty) {}

            timer.startLoadingImage(i);
            nvtx3::scoped_range loadingMarker{ "Loading image data to staging buffer" };
            task.image->load();
            const auto imageSize{ static_cast<size_t>(task.image->getWidth() * task.image->getHeight() * task.image->getChannels()) };
            std::copy_n(task.image->data(), imageSize, stagingBuffer);
            *loadFlag = LoadFlag_ImageLoaded;
            task.image->unload();
        }
        i++;
    }
}

void loadImageToGPU(cudaStream_t stream, InputBuffer& buffer, const std::shared_ptr<Image>& image, float* stagingPtr, CUdeviceptr loadFlag) {
    auto& inputSlot{ buffer.getSlot() };
    inputSlot.image = image;
    inputSlot.stagingPtr = stagingPtr;

    checkCUDAError(cuStreamWaitValue32(stream, loadFlag, LoadFlag_ImageLoaded, CU_STREAM_WAIT_VALUE_EQ),
        "An error occurred while issuing a wait on the loadFlag");
    checkCUDAError(cudaStreamWaitEvent(stream, inputSlot.executionFinished),
        "An error occurred while issuing a wait on the executionFinished event of an input slot");

    const size_t inputImageSize{ sizeof(float) * image->getWidth() * image->getHeight() * image->getChannels() };
    checkCUDAError(cudaMemcpyAsync(inputSlot.ptr, stagingPtr, inputImageSize, cudaMemcpyHostToDevice, stream),
        "An error occurred while issuing an async copy from the staging slot to an input slot");

    checkCUDAError(cuStreamWriteValue32(stream, loadFlag, LoadFlag_Empty, CU_STREAM_WRITE_VALUE_DEFAULT),
        "An error occurred while issuing a write to the loadFlag");
    checkCUDAError(cudaEventRecord(inputSlot.transferComplete, stream),
        "An error occurred while recording the transferComplete event of an input slot");
}



DetailedTaskInfo getDetailedTaskInfo(const Task& task, const std::vector<Filter>& filters, const std::vector<int>& offsets) {
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



void convoluteImageOnGPU(cudaStream_t stream, InputBufferSlot& inSlot, OutputBufferSlot& outSlot, dim3 blockSize, const DetailedTaskInfo& info, CudaTimer& timer) {
    checkCUDAError(cudaStreamWaitEvent(stream, inSlot.transferComplete),
        "An error occurred while issuing a wait on the transferComplete event of an input slot");
    checkCUDAError(cudaStreamWaitEvent(stream, outSlot.transferComplete),
        "An error occurred while issuing a wait on the transferComplete event of an output slot");

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


void writeImagesToDisk(float* floatWriteBuffer, uint8_t* uintWriteBuffer, volatile uint32_t* writeFlag, const std::vector<Task>& tasks, const std::vector<Filter>& filters, const std::filesystem::path& outputFolder, CudaTimer& timer) {
    const auto tasksCount{ tasks.size() };
    std::vector<float> outImageForCV;
    std::vector<uint8_t> outImageForCVUint;
    std::vector<float> meanSquaredErrors;
    std::ofstream validationFile{ outputFolder / "validation.txt", std::ios::app };
    for (uint32_t i{ 0 }; const auto& task : tasks) {
        const auto& filter{ filters[static_cast<FilterTypeInt>(task.filter)] };
        const auto halfSize{ filter.halfSize };
        const auto outputImageWidth{ task.padding == PaddingMode::None ?
            task.image->getWidth() - 2 * halfSize : task.image->getWidth() };
        const auto outputImageHeight{ task.padding == PaddingMode::None ?
            task.image->getHeight() - 2 * halfSize : task.image->getHeight() };
        const auto imageSize{ outputImageWidth * outputImageHeight * task.image->getChannels() };

        while (*writeFlag != WriteFlag_ImageWritten) {}

        const auto inImageWidth{ task.image->getWidth() };
        const auto inImageHeight{ task.image->getHeight() };
        const auto imageChannels{ task.image->getChannels() };
        const auto imageFormat{ imageChannels == 1 ? CV_32F : CV_32FC3 };
        const auto kernelSize{ 2 * halfSize + 1 };
        Image inImageCopy{ task.image->getPath() };
        inImageCopy.load();
        const auto inImagePtr{ inImageCopy.data() };
        cv::Mat inImageCV{ inImageHeight, inImageWidth, imageFormat, inImagePtr };
        outImageForCV.resize(inImageWidth * inImageHeight * imageChannels);
        outImageForCVUint.resize(inImageWidth * inImageHeight * imageChannels);
        cv::Mat outImageCV{ inImageHeight, inImageWidth, imageFormat, outImageForCV.data() };
        cv::Mat kernel{ kernelSize, kernelSize, CV_32F, const_cast<float*>(filter.data.data()) };
        const auto borderType{ task.padding == PaddingMode::Mirror ? cv::BORDER_REFLECT : cv::BORDER_CONSTANT  };
        cv::filter2D(inImageCV, outImageCV, -1, kernel, cv::Point(-1,-1), 0, borderType );
        float mse{ 0 };
        const auto outImageChannelsNum{ imageSize };
        if (task.padding != PaddingMode::None) {
            for (auto channel{ 0 }; channel < outImageChannelsNum; channel++) {
                mse += std::powf(
                    std::clamp(floatWriteBuffer[channel], 0.f, 1.f) -
                    std::clamp(outImageForCV[channel], 0.f, 1.f)
                    , 2.f);
            }
        } else {
            const auto inImageRowSize{ inImageWidth * imageChannels };
            const auto outImageRowSize{ outputImageWidth * imageChannels };
            for (auto row{ 0 }; row < outputImageHeight; row++) {
                for (auto channel{ 0 }; channel < outImageRowSize; channel++) {
                    mse += std::powf(
                        std::clamp(floatWriteBuffer[row * outImageRowSize + channel], 0.f, 1.f) -
                        std::clamp(outImageForCV[(halfSize + row) * inImageRowSize + halfSize * imageChannels + channel], 0.f, 1.f),
                        2.f
                     );
                }
            }
        }
        *writeFlag = WriteFlag_Empty;
        mse /= static_cast<float>(outImageChannelsNum);
        meanSquaredErrors.push_back(mse);
        validationFile << std::format("{} {}:{} MSE {}\n",
            inImageCopy.getPath().filename().string(),
            getStringFromFilterType(task.filter), getStringFromPaddingMode(task.padding),
            mse);
        i++;
        std::osyncstream{ std::cout } << i << "/" << tasksCount << " images processed\r";
        continue;

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

        timer.endWritingImage(i);
    }
    float meanMSE{ 0.f };
    for (auto mse : meanSquaredErrors) {
        meanMSE += mse;
    }
    meanMSE /= meanSquaredErrors.size();
    validationFile << std::format("Mean MSE {}\n\n",meanMSE);
    std::ofstream meanSquaredErrorsFile{ outputFolder / "meanSquaredErrors.bin", std::ios::binary | std::ios::app };
    meanSquaredErrorsFile.write(reinterpret_cast<char*>(meanSquaredErrors.data()), meanSquaredErrors.size() * sizeof(float));
}

void writeImageFromGPU(cudaStream_t stream, OutputBufferSlot& outSlot, const DetailedTaskInfo& info, float* writingSlot, CUdeviceptr writeFlag) {
    checkCUDAError(cudaStreamWaitEvent(stream, outSlot.executionFinished),
        "An error occurred while issuing a wait on the executionFinished event of an output slot");
    checkCUDAError(cuStreamWaitValue32(stream, writeFlag, WriteFlag_Empty, CU_STREAM_WAIT_VALUE_EQ),
        "An error occurred while issuing a wait on the write flag");

    const size_t outputImageSize{ sizeof(float) * info.outputImageWidth * info.outputImageHeight * info.channels };
    checkCUDAError(cudaMemcpyAsync(writingSlot, outSlot.ptr, outputImageSize, cudaMemcpyDeviceToHost, stream),
        "An error occurred while issuing an async copy from an output slot to the writing slot");

    checkCUDAError(cudaEventRecord(outSlot.transferComplete, stream),
        "An error occurred while recording the transferComplete event of an output slot");
    checkCUDAError(cuStreamWriteValue32(stream, writeFlag, WriteFlag_ImageWritten, CU_STREAM_WRITE_VALUE_DEFAULT),
        "An error occurred while issuing a write to the writeFlag");
}