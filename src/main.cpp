#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <iostream>
#include <ranges>
#include <hwy/highway.h>
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

void updateTaskData(const Task&  task, const std::vector<Filter>& filters, TaskData& data) {
    const auto isSameImage{ data.inImage == task.image };
    const auto filterHalfSize{ filters[static_cast<FilterTypeInt>(task.filter)].halfSize };

    if (!isSameImage) {
        if (data.inImage) data.inImage->unload();
        data.inImage = task.image;
        data.inImage->load();
    }
    if (!isSameImage || data.padding != task.padding || data.halfSize != filterHalfSize) {
        data.padding = task.padding;
        data.halfSize = filterHalfSize;
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
    if (argc < 3) {
        // call function usage
        std::cerr << "Not enough parameters.";
        exit(-1);
    }
    auto [images, tasks ] = loadTasks(argv[1]);
    const std::filesystem::path outputFolder{ argv[2] };
    const auto filters{ getFilters() };

    std::ranges::sort(tasks,
        [&filters](const Task& t1, const Task& t2) {
            using FilterInt = std::underlying_type_t<FilterType>;

            if (t1.image != t2.image) return t1.image->getPath() < t2.image->getPath();
            if (t1.padding != t2.padding) return t1.padding < t2.padding;
            return filters[static_cast<FilterInt>(t1.filter)].halfSize <
                filters[static_cast<FilterInt>(t2.filter)].halfSize;
        }
    );

    std::shared_ptr<Image> previousImage{};
    PaddedImage previousPaddedImage{};
    int previousHalfSize{ -1 };
    auto previousPaddingMode{ PaddingMode::Invalid };
    for (const auto& [image, filter, padding] : tasks) {
        // get convolution data + checks
        PaddedImage paddedImage{};
        auto halfSize{ filters[static_cast<FilterTypeInt>(filter)].halfSize };
        auto filterPtr{ filters[static_cast<FilterTypeInt>(filter)].data.data() };
        auto imageWidth{ image->getWidth() };
        auto imageChannels{ image->getChannels() };
        float * paddedImagePtr{ nullptr };
        size_t paddedRowLength{};
        size_t rowLength{};
        size_t rowNum{};

        auto sameImage{ previousImage == image };
        if (!sameImage) {
            if (previousImage) previousImage->unload();
            image->load();
        }
        if (padding == PaddingMode::None) {
            paddedImagePtr = image->data();
            paddedRowLength = imageWidth * imageChannels;
            rowLength = (imageWidth - 2 * halfSize) * imageChannels;
            rowNum = image->getHeight() - 2 * halfSize;
        } else {
            if (sameImage && previousPaddingMode == padding && previousHalfSize == halfSize) {
                paddedImage = std::move(previousPaddedImage);
            } else {
                paddedImage.pad(*image, padding, halfSize);
            }
            paddedImagePtr = paddedImage.data();
            paddedRowLength = paddedImage.getPaddedWidth() * imageChannels;
            rowLength = imageWidth * imageChannels;
            rowNum = paddedImage.getHeight();
        }
        std::vector<float> destImage(rowLength * rowNum);

        // convolution
        using namespace hwy::HWY_NAMESPACE;
        using D = ScalableTag<float>;
        using V = VFromD<D>;

        D d{};
        size_t vectorLength{ Lanes(d) };
        size_t vectorsInRow{ rowLength / vectorLength };
        size_t remainingChannels{ rowLength - vectorsInRow * vectorLength };
        for (int row{ halfSize }; row < rowNum + halfSize; row++) {
            for (size_t vector{ 0 }; vector < vectorsInRow; vector++) {
                V accum{ Set(d, 0) };
                for (int i{ -halfSize }; i < halfSize; i++) {
                    for (int j{ 0 }; j < 2 * halfSize + 1; j++) {
                        V pixelData{ Load(d, paddedImagePtr
                            + vector * vectorLength
                            + (row + i) * paddedRowLength + j * imageChannels)};
                        V kernelCoef{ Set(d, filterPtr[(i + halfSize) * (2 * halfSize + 1) + j]) };
                        accum = MulAdd(kernelCoef, pixelData, accum);
                    }
                }
                Store(accum, d, destImage.data() + (row - halfSize) * rowLength + vector * vectorLength);
            }
            V accum{ Set(d, 0) };
            for (int i{ -halfSize }; i < halfSize; i++) {
                for (int j{ 0 }; j < 2 * halfSize + 1; j++) {
                    V pixelData{ LoadN(d, paddedImagePtr
                        + vectorsInRow * vectorLength
                        + (row + i) * paddedRowLength + j * imageChannels, remainingChannels)};
                    V kernelCoef{ Set(d, filterPtr[(i + halfSize) * (2 * halfSize + 1) + j]) };
                    accum = MulAdd(kernelCoef, pixelData, accum);
                }
            }
            StoreN(accum, d, destImage.data() + (row - halfSize) * rowLength + vectorsInRow * vectorLength, remainingChannels);
        }

        // float to uint + adjustments
        auto floatToInt { destImage | std::views::transform(
            [](const float x) {
                return static_cast<uint8_t>(
                    std::clamp(
                        std::roundf(
                            std::powf(x, 1/2.2f) * 255
                            ), 0.f, 255.f
                    )
                );
            })
        };
        std::vector<uint8_t> dstImageInInt{ floatToInt.begin(), floatToInt.end() };

        // write image
        auto outImagePath{ outputFolder / image->getPath().stem() };
        outImagePath += "-" + getStringFromFilterType(filter) + "-" + getStringFromPaddingMode(padding) + ".jpg";
        stbi_write_jpg(outImagePath.string().c_str()
            , static_cast<int>(rowLength / imageChannels), static_cast<int>(rowNum), imageChannels
            , dstImageInInt.data(), 100);

        previousImage = image;
        previousPaddedImage = std::move(paddedImage);
        previousHalfSize = halfSize;
        previousPaddingMode = padding;
    }
    previousImage->unload();

    return 0;
}