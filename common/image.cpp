#include "image.h"
#include <ranges>
#include <type_traits>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


FilterType getFilterTypeFromString(const std::string& string) {
    auto filter{ FilterType::Invalid };

    if (string == "BoxBlur3") {
        filter = FilterType::BoxBlur3;
    } else if (string == "GaussBlur3") {
        filter = FilterType::GaussBlur3;
    } else if (string == "LaPlace3_4") {
        filter = FilterType::LaPlace3_4;
    } else if (string == "LaPlace3_8") {
        filter = FilterType::LaPlace3_8;
    } else if (string == "Sharp3") {
        filter = FilterType::Sharp3;
    } else if (string == "LoG5") {
        filter = FilterType::LoG5;
    } else if (string == "LoG9") {
        filter = FilterType::LoG9;
    } else if (string == "EmbossUp3") {
        filter = FilterType::EmbossUp3;
    } else if (string == "SobelUp3") {
        filter = FilterType::SobelUp3;
    } else if (string == "SobelRight3") {
        filter = FilterType::SobelRight3;
    } else {
        std::cerr << "Unknown filter type: " << string << std::endl;
        exit(-1);
    }

    return filter;
}

std::string getStringFromFilterType(FilterType filterType) {
    std::string res{};

    switch (filterType) {
        case FilterType::BoxBlur3: {
            res = "BoxBlur3";
            break;
        }
        case FilterType::GaussBlur3: {
            res = "GaussBlur3";
            break;
        }
        case FilterType::LaPlace3_4: {
            res = "LaPlace3_4";
            break;
        }
        case FilterType::LaPlace3_8: {
            res = "LaPlace3_8";
            break;
        }
        case FilterType::Sharp3: {
            res = "Sharp3";
            break;
        }
        case FilterType::LoG5: {
            res = "LoG5";
            break;
        }
        case FilterType::LoG9: {
            res = "LoG9";
            break;
        }
        case FilterType::EmbossUp3: {
            res = "EmbossUp3";
            break;
        }
        case FilterType::SobelUp3: {
            res = "SobelUp3";
            break;
        }
        case FilterType::SobelRight3: {
            res = "SobelRight3";
            break;
        }
        default: {
            std::cerr << "Unknown filter type: " << static_cast<FilterTypeInt>(filterType) << std::endl;
            exit(-1);
        }
    }

    return res;
}

std::vector<Filter> getFilters() {
    namespace views = std::ranges::views;

    constexpr auto filterNum{ static_cast<FilterTypeInt>(FilterType::Num) };
    constexpr auto filtersView{
        views::iota(0, filterNum) |
        views::transform([](FilterTypeInt x) {
            return Filter{ static_cast<FilterType>(x) };
        })
    };
    return {filtersView.begin(), filtersView.end()};
}

PaddingMode getPaddingModeFromString(const std::string& string) {
    auto paddingMode{ PaddingMode::None };

    if (string == "None") {
        paddingMode = PaddingMode::None;
    } else if (string == "Zero") {
        paddingMode = PaddingMode::Zero;
    } else if (string == "Mirror") {
        paddingMode = PaddingMode::Mirror;
    } else {
        std::cerr << "Unknown padding mode: " << string << std::endl;
        exit(-1);
    }

    return paddingMode;
}

std::string getStringFromPaddingMode(PaddingMode paddingMode) {
    std::string res{};

    switch (paddingMode) {
        case PaddingMode::None: {
            res = "None";
            break;
        }
        case PaddingMode::Zero: {
            res = "Zero";
            break;
        }
        case PaddingMode::Mirror: {
            res = "Mirror";
            break;
        }
        default: {
            std::cerr << "Unknown padding mode: " << static_cast<PaddingModeInt>(paddingMode) << std::endl;
            exit(-1);
        }
    }

    return res;
}

Image::Image(std::filesystem::path path)
    : imagePath{ std::move(path) } {
    if (stbi_info(imagePath.string().c_str(), &width, &height, &channels) == 0) {
        std::cerr << "An error occurred while reading " << imagePath << std::endl;
        std::cerr << stbi_failure_reason() << std::endl;
        exit(-1);
    }
    // This assures the loaded image will have no alpha component
    // 2 is for gray and alpha, 4 is for red, green, blue and alpha
    if (channels % 2 == 0) {
        channels -= 1;
    }
}

Image::~Image() { unload(); }

float * Image::data() { return const_cast<float*>(std::as_const(*this).data()); }

const float * Image::data() const { return imageData; }

void Image::load() {
    if (imageData) return;

    int dummyX, dummyY, dummyC; // Use dummies because we already retrieved the info when constructing
    imageData = stbi_loadf(imagePath.string().c_str(), &dummyX, &dummyY, &dummyC, channels);
    if (!imageData) {
        std::cerr << "An error occurred when loading " << imagePath << std::endl;
        std::cerr << stbi_failure_reason() << std::endl;
        exit(-1);
    }
}

void Image::unload() {
    stbi_image_free(imageData);
    imageData = nullptr;
}

std::filesystem::path Image::getPath() const { return imagePath; }

int Image::getWidth() const { return width; }

int Image::getHeight() const { return height; }

int Image::getChannels() const { return channels; }

PaddedImage::PaddedImage(const Image &image, const PaddingMode mode, const int halfSize) {
    pad(image, mode, halfSize);
}

void PaddedImage::pad(const Image &image, const PaddingMode mode, const int halfSize) {
    if (!image.data()) {
        std::cerr << "Tried to pad an unloaded image: " << image.getPath() << std::endl;
        exit(-1);
    }

    paddingHalfSize = halfSize;
    width = image.getWidth();
    paddedWidth = width + 2 * paddingHalfSize;
    height = image.getHeight();
    paddedHeight = height + 2 * paddingHalfSize;
    channels = image.getChannels();
    paddingMode = mode;
    switch (mode) {
        case PaddingMode::None: {
            paddedWidth = width;
            paddedHeight = height;
            paddingHalfSize = 0;
            imageData = { image.data(), image.data() + width * height * channels };
            break;
        }
        case PaddingMode::Zero: {
            imageData = std::vector<float>( paddedWidth * paddedHeight * channels);
            for (int i{ 0 }; i < height; i++) {
                std::copy_n(image.data() + i * width * channels,
                    width * channels,
                    imageData.data() + ((paddingHalfSize + i) * paddedWidth + paddingHalfSize) * channels);
            }
            break;
        }
        case PaddingMode::Mirror: {
            imageData = std::vector<float>( paddedWidth * paddedHeight * channels);
            for (int i{ 0 }; i < height; i++) {
                std::copy_n(image.data() + i * width * channels,
                    width * channels,
                    imageData.data() + ((paddingHalfSize + i) * paddedWidth + paddingHalfSize) * channels);
                for (int j{ 0 }; j < halfSize; j++) {
                    for (int c{ 0 }; c < channels; c++) {
                        imageData[((paddingHalfSize + i) * paddedWidth + paddingHalfSize - 1 - j) * channels + c] =
                            image.data()[(i * width + 1 + j) * channels + c];
                        imageData[((paddingHalfSize + i) * paddedWidth + paddingHalfSize + width + j) * channels + c] =
                            image.data()[((i + 1) * width - 2 - j) * channels + c];
                    }
                }
            }
            for (int i{ 0 }; i < halfSize; i++) {
                std::copy_n(imageData.data() + (paddingHalfSize + 1 + i) * paddedWidth * channels,
                    paddedWidth * channels,
                    imageData.data() + (paddingHalfSize - 1 - i) * paddedWidth * channels);
                std::copy_n(imageData.data() + (paddingHalfSize + height - 2 - i) * paddedWidth * channels,
                    paddedWidth * channels,
                    imageData.data() + (paddingHalfSize + height + i) * paddedWidth * channels);
            }
            break;
        }
        default: {
            std::cerr << "Unknown padding mode: "  << static_cast<PaddingModeInt>(mode) << std::endl;
            exit(-1);
        }
    }
}

float * PaddedImage::data() { return const_cast<float*>(std::as_const(*this).data()); }

const float * PaddedImage::data() const { return imageData.data(); }

int PaddedImage::getWidth() const { return width; }

int PaddedImage::getPaddedWidth() const { return paddedWidth; }

int PaddedImage::getHeight() const { return height; }

int PaddedImage::getPaddedHeight() const { return paddedHeight; }

int PaddedImage::getChannels() const { return channels; }

void writeImage(const std::string& path, const int width, const int height, const int channels, const uint8_t* data) {
    stbi_write_jpg(path.c_str(), width, height, channels, data, 100);
}