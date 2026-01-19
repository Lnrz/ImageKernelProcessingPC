#include <ranges>
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
    if (data.inImage != task.image) {
        if (data.inImage) data.inImage->unload();
        data.inImage = task.image;
        data.inImage->load();
    }
    data.padding = task.padding;
    data.halfSize = halfSize;
    data.inPaddedImage = (data.padding == PaddingMode::None) ?
    PaddedImage{} : PaddedImage{*data.inImage, data.padding, data.halfSize};

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
        enableStats
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
    const auto tasksCount{ tasks.size() };
    Timer timer{ enableStats? tasksCount : 0, getCPULanes() };
    if (enableStats) timer.startingProgram();
    for (int i{ 1 }; const auto& task : tasks) {
        const auto& filter{ filters[static_cast<FilterTypeInt>(task.filter)] };

        if (enableStats) timer.startingImageLoading();
        updateTaskData(task, filter.halfSize, taskData);
        const auto imageChannels{ taskData.inImage->getChannels() };
        if (enableStats) timer.startingImageConvolution();
        kernelConvolution({
            .inPtr = (taskData.padding == PaddingMode::None) ?
                taskData.inImage->data() : taskData.inPaddedImage.data(),
            .coefPtr = filter.data.data(),
            .outPtr = taskData.outFloatImage.data(),
            .rowSize = taskData.rowSize,
            .rowNum = taskData.rowNum,
            .channels = imageChannels,
            .halfSize = taskData.halfSize
        });
        if (enableStats) timer.imageConvolutionEnded();
        floatImageToUIntImage(taskData.outFloatImage, taskData.outUIntImage);
        writeImage((outputFolder / taskData.inImage->getPath().stem()).string()
            + "-" + getStringFromFilterType(task.filter)
            + "-" + getStringFromPaddingMode(task.padding) + ".jpg"
            ,taskData.rowSize / imageChannels, taskData.rowNum, imageChannels
            , taskData.outUIntImage.data());
        if (enableStats) timer.imageWritten();
        std::cout << i << "/" << tasksCount << " images processed\r" << std::flush;
        i++;
    }
    if (enableStats) {
        timer.endingProgram();
        timer.writeLog(outputFolder / "log.txt");
    }
    std::cout << std::endl;

    return 0;
}