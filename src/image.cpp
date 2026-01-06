#include "image.h"

#include <iostream>
#include <ranges>
#include <type_traits>
#include <stb_image.h>


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
        default: {
            std::cerr << "Unknown filter type: " << static_cast<FilterTypeInt>(filterType) << std::endl;
            exit(-1);
        }
    }

    return res;
}

Filter::Filter(FilterType type)
    : type{ type } {
    switch (type) {
        case FilterType::BoxBlur3: {
            halfSize = 1;
            data = std::vector( 9, 1.f/9.f);
            break;
        }
        case FilterType::GaussBlur3: {
            halfSize = 1;
            data = {
                1.f/16.f, 2.f/16.f, 1.f/16.f,
                2.f/16.f, 4.f/16.f, 2.f/16.f,
                1.f/16.f, 2.f/16.f, 1.f/16.f
            };
            break;
        }
        case FilterType::LaPlace3_4: {
            halfSize = 1;
            data = {
                0.f, 1.f, 0.f,
                1.f, -4.f, 1.f,
                0.f, 1.f, 0.f
            };
            break;
        }
        case FilterType::LaPlace3_8: {
            halfSize = 1;
            data = {
                1.f, 1.f, 1.f,
                1.f, -8.f, 1.f,
                1.f, 1.f, 1.f
            };
            break;
        }
        case FilterType::Sharp3: {
            halfSize = 1;
            data = {
                0.f, -1.f, 0.f,
                -1.f, 5.f, -1.f,
                0.f, -1.f, 0.f
            };
            break;
        }
        default: {
            std::cerr << "Filter type: " << static_cast<std::underlying_type_t<FilterType>>(type)
                    << " still not implemented" << std::endl;
            exit(-1);
        }
    }
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
    } else if (string == "Clamp") {
        paddingMode = PaddingMode::Clamp;
    } else if (string == "Mirror") {
        paddingMode = PaddingMode::Mirror;
    } else if (string == "Reverse") {
        paddingMode = PaddingMode::Reverse;
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
        case PaddingMode::Clamp: {
            res = "Clamp";
            break;
        }
        case PaddingMode::Mirror: {
            res = "Mirror";
            break;
        }
        case PaddingMode::Reverse: {
            res = "Reverse";
            break;
        }
        default: {
            std::cerr << "Unknown padding: " << static_cast<PaddingModeInt>(paddingMode) << std::endl;
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
        default: {
            std::cerr << "Padding mode "  << static_cast<std::underlying_type_t<PaddingMode>>(mode)
                      << " currently not implemented" << std::endl;
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

void floatImageToUIntImage(const std::vector<float>& from, std::vector<uint8_t>& to) {
    std::ranges::copy(
        from | std::views::transform(
            [](const float x) {
                const auto gammaCorrectedValue{ std::powf(x, 1/2.2f) };
                const auto scaledValue{ 255 * gammaCorrectedValue };
                const auto clampedValue{ std::clamp(scaledValue, 0.f, 255.f) };
                const auto roundedValue{ std::roundf(clampedValue) };
                return static_cast<uint8_t>(roundedValue);
            }
        ), to.begin()
    );
}