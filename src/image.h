#ifndef IMAGEKERNELPROCESSING_IMAGE_H
#define IMAGEKERNELPROCESSING_IMAGE_H

#include <filesystem>
#include <ranges>
#include <numeric>
#include <iostream>

enum class FilterType {
    Invalid = -1,
    BoxBlur3,
    GaussBlur3,
    LaPlace3_4,
    LaPlace3_8,
    Sharp3,
    LoG5,
    LoG9,
    EmbossUp3,
    Num
};
using FilterTypeInt = std::underlying_type_t<FilterType>;

FilterType getFilterTypeFromString(const std::string& string);
std::string getStringFromFilterType(FilterType filterType);

struct Filter {
    explicit constexpr Filter(FilterType type)
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
            case FilterType::LoG5: {
                halfSize = 2;
                data = {
                    0.f, 0.f, 1.f, 0.f, 0.f,
                    0.f, 1.f, 2.f, 1.f, 0.f,
                    1.f, 2.f, -16.f, 2.f, 1.f,
                    0.f, 1.f, 2.f, 1.f, 0.f,
                    0.f, 0.f, 1.f, 0.f, 0.f
                };
                break;
            }
            case FilterType::LoG9: {
                halfSize = 4;
                data = {
                    0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 1.f, 1.f, 0.f,
                    1.f, 2.f, 4.f, 5.f, 5.f, 5.f, 4.f, 2.f, 1.f,
                    1.f, 4.f, 5.f, 3.f, 0.f, 3.f, 5.f, 4.f, 1.f,
                    2.f, 5.f, 3.f, -12.f, -24.f, -12.f, 3.f, 5.f, 2.f,
                    2.f, 5.f, 0.f, -24.f, -40.f, -24.f, 0.f, 5.f, 2.f,
                    2.f, 5.f, 3.f, -12.f, -24.f, -12.f, 3.f, 5.f, 2.f,
                    1.f, 4.f, 5.f, 3.f, 0.f, 3.f, 5.f, 4.f, 1.f,
                    1.f, 2.f, 4.f, 5.f, 5.f, 5.f, 4.f, 2.f, 1.f,
                    0.f, 1.f, 1.f, 2.f, 2.f, 2.f, 1.f, 1.f, 0.f
                };
                break;
            }
            case FilterType::EmbossUp3: {
                halfSize = 1;
                data = {
                    0.f, 1.f, 0.f,
                    0.f, 0.f, 0.f,
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
    };

    std::vector<float> data{};
    FilterType type{ FilterType::Invalid };
    int halfSize{};
};

std::vector<Filter> getFilters();

constexpr size_t getFiltersSize() {
    namespace views = std::ranges::views;

    constexpr auto filterNum{ static_cast<FilterTypeInt>(FilterType::Num) };
    constexpr auto filtersSizes{
        views::iota(0, filterNum) |
        views::transform([](FilterTypeInt x) {
            return Filter{ static_cast<FilterType>(x) }.data.size();
        })
    };
    return std::accumulate(filtersSizes.begin(), filtersSizes.end(), static_cast<size_t>(0));
};


enum class PaddingMode {
    Invalid = -1,
    None,
    Zero,
    Clamp,
    Mirror,
    Reverse
};
using PaddingModeInt = std::underlying_type_t<PaddingMode>;

PaddingMode getPaddingModeFromString(const std::string& string);
std::string getStringFromPaddingMode(PaddingMode paddingMode);



class Image {
public:
    explicit Image(std::filesystem::path path);
    Image(const Image&) = delete;
    Image(Image&&) = delete;
    ~Image();

    Image& operator=(const Image&) = delete;
    Image& operator=(Image&&) = delete;

    [[nodiscard]]
    float* data();
    [[nodiscard]]
    const float* data() const;

    void load();
    void unload();

    [[nodiscard]]
    std::filesystem::path getPath() const;
    [[nodiscard]]
    int getWidth() const;
    [[nodiscard]]
    int getHeight() const;
    [[nodiscard]]
    int getChannels() const;

private:
    std::filesystem::path imagePath{};
    int width{}, height{};
    int channels{};
    float* imageData{ nullptr };
};

class PaddedImage {
public:
    PaddedImage() = default;
    PaddedImage(const Image& image, PaddingMode mode, int halfSize);

    void pad(const Image& image, PaddingMode mode, int halfSize);

    [[nodiscard]]
    float* data();
    [[nodiscard]]
    const float* data() const;

    [[nodiscard]]
    int getWidth() const;
    [[nodiscard]]
    int getPaddedWidth() const;
    [[nodiscard]]
    int getHeight() const;
    [[nodiscard]]
    int getPaddedHeight() const;
    [[nodiscard]]
    int getChannels() const;

private:
    std::vector<float> imageData{};
    int width{}, height{};
    int paddingHalfSize{};
    int paddedWidth{}, paddedHeight{};
    int channels{};
    PaddingMode paddingMode{ PaddingMode::Invalid };
};

template<std::ranges::input_range Ri, std::ranges::output_range<uint8_t> Ro>
requires std::floating_point<std::ranges::range_value_t<Ri>>
void floatImageToUIntImage(Ri&& from, Ro&& to) {
    std::ranges::copy(
        from | std::views::transform(
            [](const std::floating_point auto x) {
                const auto gammaCorrectedValue{ std::pow(x, 1/2.2) };
                const auto scaledValue{ 255 * gammaCorrectedValue };
                const auto clampedValue{ std::clamp(scaledValue, 0., 255.) };
                const auto roundedValue{ std::round(clampedValue) };
                return static_cast<uint8_t>(roundedValue);
            }
        ), to.begin()
    );
}

#endif //IMAGEKERNELPROCESSING_IMAGE_H