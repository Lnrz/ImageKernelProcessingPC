#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>
#include <ranges>
#include "image.h"
#include "taskLoader.h"
#include "convolution.h"

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

[[noreturn]]
void explainProgram() {
    std::cout << R"(Program usage:
ImageKernelProcessing.exe inputFolder outputFolder

inputFolder:
    path to the folder containing the images to process and a tasks.txt file
outputFolder:
    path to the folder where to write the output images

The tasks.txt file contains the tasks specified as lines in the following format:
IMAGE image1 filter1_1:padding1_1
IMAGE image2 filter2_1:padding2_1 filter2_2:padding2_2
...

image is the name of the image to process, contained in the inputFolder
filter is the filter to apply to the image
padding is the padding to apply to the image
It is possible to specify more filters and paddings for the same image by separating them with a whitespace


)";
    exit(-1);
}

int main(int argc, char* argv[]) {
    if (argc < 3) explainProgram();
    auto [images, tasks ] = loadTasks(argv[1]);
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
    for (const auto& task : tasks) {
        const auto& filter{ filters[static_cast<FilterTypeInt>(task.filter)] };

        updateTaskData(task, filter.halfSize, taskData);
        const auto imageChannels{ taskData.inImage->getChannels() };
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
        floatImageToUIntImage(taskData.outFloatImage, taskData.outUIntImage);
        auto outImagePath{ outputFolder / taskData.inImage->getPath().stem() };
        outImagePath += "-" + getStringFromFilterType(task.filter)
                      + "-" + getStringFromPaddingMode(task.padding) + ".jpg";
        stbi_write_jpg(outImagePath.string().c_str()
            ,taskData.rowSize / imageChannels, taskData.rowNum, imageChannels
            , taskData.outUIntImage.data(), 100);
    }

    return 0;
}