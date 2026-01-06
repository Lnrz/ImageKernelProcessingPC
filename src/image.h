#ifndef IMAGEKERNELPROCESSING_IMAGE_H
#define IMAGEKERNELPROCESSING_IMAGE_H

#include <filesystem>



enum class FilterType {
    Invalid = -1,
    BoxBlur3,
    GaussBlur3,
    LaPlace3_4,
    LaPlace3_8,
    Sharp3,
    Num
};
using FilterTypeInt = std::underlying_type_t<FilterType>;

FilterType getFilterTypeFromString(const std::string& string);
std::string getStringFromFilterType(FilterType filterType);

struct Filter {
    explicit Filter(FilterType type);

    std::vector<float> data{};
    FilterType type{ FilterType::Invalid };
    int halfSize{};
};

std::vector<Filter> getFilters();



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

void floatImageToUIntImage(const std::vector<float>& from, std::vector<uint8_t>& to);

#endif //IMAGEKERNELPROCESSING_IMAGE_H