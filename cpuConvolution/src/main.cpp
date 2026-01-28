#include <ranges>
#include "image.h"
#include "taskLoader.h"
#include "convolution.h"
#include "utilities.h"
#include "timer.h"

// Struct containing data needed to process the task.
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

// Update data for the current task. halfSize is the half size of the current filter.
//
// The function assumes that data already stores the data used for the previous task
// and performs only what is strictly necessary to update it for the current task.
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

    // Sort the images to process them faster.
    // By processing all the tasks for the same image one after another
    // we do not have to load the image in memory more than once.
    // Similarly, processing all the tasks of the same image by their padding and size
    // permits to reuse the calculated padded image when possible.
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
    Timer timer{ enableStats? tasksCount : 0, getCPUFloatLanes(), disableVect };
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
        timer.writeLog(outputFolder);
    }
    std::cout << std::endl;

    return 0;
}