#include <ranges>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include "image.h"
#include "taskLoader.h"
#include "convolution.h"
#include "utilities.h"
#include "timer.h"

struct TaskData {
    std::shared_ptr<Image> inImage{};
    PaddingMode padding{ PaddingMode::Invalid };
    int halfSize{};
    PaddedImage inPaddedImage{};
    int rowSize{};
    int rowNum{};
    std::vector<float> outFloatImage{};
    std::vector<uint8_t> outUIntImage{};
};

void updateTaskData(const Task&  task, const int halfSize, TaskData& data) {
    const bool isSameImage{ data.inImage == task.image };
    const bool isSamePaddedImage{ isSameImage && data.padding == task.padding && data.halfSize == halfSize };

    if (!isSameImage) {
        if (data.inImage) data.inImage->unload();
        data.inImage = task.image;
        data.inImage->load();
    }
    if (!isSamePaddedImage) {
        data.padding = task.padding;
        data.halfSize = halfSize;
        data.inPaddedImage = (data.padding == PaddingMode::None) ?
        PaddedImage{} : PaddedImage{*data.inImage, data.padding, data.halfSize};
    }

    int outImageWidth{ data.inImage->getWidth() };
    int outImageHeight{ data.inImage->getHeight() };
    const auto outImageChannels{ data.inImage->getChannels() };
    if (data.padding == PaddingMode::None) {
        outImageWidth -= 2 * data.halfSize;
        outImageHeight -= 2 * data.halfSize;
    }
    data.rowSize = outImageWidth * outImageChannels;
    data.rowNum = outImageHeight;

    const auto outImageSize{ outImageHeight * outImageWidth * outImageChannels };
    if (data.outFloatImage.size() != outImageSize) data.outFloatImage.resize(outImageSize);
    if (data.outUIntImage.size() != outImageSize) data.outUIntImage.resize(outImageSize);
}

int main(int argc, char* argv[]) {
    if (argc < 3) explainProgram();
    auto [
        images,
        tasks,
        enableStats,
        disableVect
    ] = loadTasks(argv[1]);
    const std::filesystem::path outputFolder{ argv[2] };
    const auto filters{ getFilters() };

    std::ranges::sort(tasks,
        [&filters](const Task& t1, const Task& t2) {
            if (t1.image != t2.image) return t1.image->getPath() < t2.image->getPath();
            if (t1.padding != t2.padding) return t1.padding < t2.padding;
            return filters[static_cast<FilterTypeInt>(t1.filter)].halfSize <
                   filters[static_cast<FilterTypeInt>(t2.filter)].halfSize;
        }
    );

    TaskData taskData{};
    std::vector<float> outImageForCV;
    std::vector<uint8_t> outImageForCVUint;
    std::vector<float> meanSquaredErrors;
    std::ofstream validationFile{ outputFolder / "validation.txt", std::ios::app };
    const auto tasksCount{ tasks.size() };
    Timer timer{ enableStats? tasksCount : 0, getCPULanes(), disableVect };
    if (enableStats) timer.startingProgram();
    for (int i{ 1 }; const auto& task : tasks) {
        const auto& filter{ filters[static_cast<FilterTypeInt>(task.filter)] };

        if (enableStats) timer.startingImageLoading();
        updateTaskData(task, filter.halfSize, taskData);
        const auto imageChannels{ taskData.inImage->getChannels() };
        if (enableStats) timer.startingImageConvolution();
        ConvolutionData data{
            .inPtr = (taskData.padding == PaddingMode::None) ?
                taskData.inImage->data() : taskData.inPaddedImage.data(),
            .coefPtr = filter.data.data(),
            .outPtr = taskData.outFloatImage.data(),
            .rowSize = taskData.rowSize,
            .rowNum = taskData.rowNum,
            .channels = imageChannels,
            .halfSize = taskData.halfSize
        };
        if (!disableVect) { kernelConvolution(data); }
        else { scalarKernelConvolution(data); }
        if (enableStats) timer.imageConvolutionEnded();

        const auto inImageWidth{ taskData.inImage->getWidth() };
        const auto inImageHeight{ taskData.inImage->getHeight() };
        const auto imageFormat{ imageChannels == 1 ? CV_32F : CV_32FC3 };
        const auto kernelSize{ 2 * taskData.halfSize + 1 };
        const auto inImagePtr{ taskData.inImage->data() };
        cv::Mat inImageCV{ inImageHeight, inImageWidth, imageFormat, inImagePtr };
        outImageForCV.resize(inImageWidth * inImageHeight * imageChannels);
        outImageForCVUint.resize(inImageWidth * inImageHeight * imageChannels);
        cv::Mat outImageCV{ inImageHeight, inImageWidth, imageFormat, outImageForCV.data() };
        cv::Mat kernel{ kernelSize, kernelSize, CV_32F, const_cast<float*>(filter.data.data()) };
        const auto borderType{ taskData.padding == PaddingMode::Mirror ? cv::BORDER_REFLECT : cv::BORDER_CONSTANT  };
        cv::filter2D(inImageCV, outImageCV, -1, kernel, cv::Point(-1,-1), 0, borderType );
        float mse{ 0 };
        const auto outImageChannelsNum{ taskData.rowSize * taskData.rowNum };
        if (taskData.padding != PaddingMode::None) {
            for (auto channel{ 0 }; channel < outImageChannelsNum; channel++) {
                mse += std::powf(
                    std::clamp(taskData.outFloatImage[channel], 0.f, 1.f) -
                    std::clamp(outImageForCV[channel], 0.f, 1.f)
                    , 2.f);
            }
        } else {
            const auto inImageRowSize{ inImageWidth * imageChannels };
            for (auto row{ 0 }; row < taskData.rowNum; row++) {
                for (auto channel{ 0 }; channel < taskData.rowSize; channel++) {
                    mse += std::powf(
                        std::clamp(taskData.outFloatImage[row * taskData.rowSize + channel], 0.f, 1.f) -
                        std::clamp(outImageForCV[(taskData.halfSize + row) * inImageRowSize + taskData.halfSize * imageChannels + channel], 0.f, 1.f),
                        2.f
                     );
                }
            }
        }
        mse /= static_cast<float>(outImageChannelsNum);
        meanSquaredErrors.push_back(mse);
        validationFile << std::format("{} {}:{} MSE {}\n",
            taskData.inImage->getPath().filename().string(),
            getStringFromFilterType(task.filter), getStringFromPaddingMode(task.padding),
            mse);
        std::cout << i << "/" << tasksCount << " images processed\r" << std::flush;
        i++;
        continue;

        floatImageToUIntImage(taskData.outFloatImage, taskData.outUIntImage);
        floatImageToUIntImage(outImageForCV, outImageForCVUint);
        writeImage((outputFolder / taskData.inImage->getPath().stem()).string()
            + "-" + getStringFromFilterType(task.filter)
            + "-" + getStringFromPaddingMode(task.padding) + ".jpg"
            ,taskData.rowSize / imageChannels, taskData.rowNum, imageChannels
            , taskData.outUIntImage.data());
        writeImage((outputFolder / taskData.inImage->getPath().stem()).string()
            + "-" + getStringFromFilterType(task.filter)
            + "-" + getStringFromPaddingMode(task.padding) + "-CV.jpg"
            ,inImageWidth, inImageHeight, imageChannels
            , outImageForCVUint.data());
        if (enableStats) timer.imageWritten();
    }
    if (enableStats) {
        timer.endingProgram();
        timer.writeLog(outputFolder);
    }
    float meanMSE{ 0.f };
    for (auto mse : meanSquaredErrors) {
        meanMSE += mse;
    }
    meanMSE /= meanSquaredErrors.size();
    validationFile << std::format("Mean MSE {}\n\n",meanMSE);
    std::ofstream meanSquaredErrorsFile{ outputFolder / "meanSquaredErrors.bin", std::ios::binary | std::ios::app };
    meanSquaredErrorsFile.write(reinterpret_cast<char*>(meanSquaredErrors.data()), meanSquaredErrors.size() * sizeof(float));
    std::cout << std::endl;

    return 0;
}