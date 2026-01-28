#ifndef IMAGEKERNELPROCESSING_IMAGE_H
#define IMAGEKERNELPROCESSING_IMAGE_H

#include <filesystem>
#include <ranges>
#include <numeric>
#include <iostream>

// Enum of available filters.
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
    SobelUp3,
    SobelRight3,
    Num
};
using FilterTypeInt = std::underlying_type_t<FilterType>;

// Return the FilterType represented by string.
//
// If string does not represent any filter print an error and exit the program.
FilterType getFilterTypeFromString(const std::string& string);
// Return the string representation of filterType.
//
// If filterType is outside the valid values print an error and exit the program.
std::string getStringFromFilterType(FilterType filterType);

// Struct representing a filter.
//
// Holds the filter coefficients, half size and type.
//
// The coefficients are stored in row major order.
struct Filter {
    // Construct the filter corresponding to type.
    //
    // If type is outside the valid values prints an error and exit.
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
            case FilterType::SobelUp3: {
                halfSize = 1;
                data = {
                    1.f, 2.f, 1.f,
                    0.f, 0.f, 0.f,
                    -1.f, -2.f, -1.f
                };
                break;
            }
            case FilterType::SobelRight3: {
                halfSize = 1;
                data = {
                    -1.f, 0.f, 1.f,
                    -2.f, 0.f, 2.f,
                    -1.f, 0.f, 1.f
                };
                break;
            }
            default: {
                std::cerr << "Invalid filter type: " << static_cast<FilterTypeInt>(type) << std::endl;
                exit(-1);
            }
        }
    };

    std::vector<float> data{};
    FilterType type{ FilterType::Invalid };
    int halfSize{};
};

// Return a vector with all the available filters.
//
// The filters are indexable by their corresponding filter types.
std::vector<Filter> getFilters();

// Return the number of coefficients in all filters.
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


// Enum of available padding modes.
enum class PaddingMode {
    Invalid = -1,
    None,
    Zero,
    Mirror
};
using PaddingModeInt = std::underlying_type_t<PaddingMode>;

// Return the PaddingMode represented by string.
//
// If string does not represent any padding mode print an error and exit the program.
PaddingMode getPaddingModeFromString(const std::string& string);
// Return the string representation of paddingMode.
//
// If paddingMode is outside the valid values print an error and exit the program.
std::string getStringFromPaddingMode(PaddingMode paddingMode);


// Class representing an image.
//
// It does not load automatically the image data, only its metadata (width, height and channels).
// Before reading the image you have to load it using the homonymous method.
class Image {
public:
    explicit Image(std::filesystem::path path);
    Image(const Image&) = delete;
    Image(Image&&) = delete;
    ~Image();

    Image& operator=(const Image&) = delete;
    Image& operator=(Image&&) = delete;

    // Return a pointer to the image data.
    //
    // If the image was not loaded it will be a nullptr.
    //
    // The data are stored in row major order.
    [[nodiscard]]
    float* data();
    // Return a const pointer to the image data.
    //
    // If the image was not loaded it will be a nullptr.
    //
    // The data are stored in row major order.
    [[nodiscard]]
    const float* data() const;

    // Load the image data.
    //
    // If an error occurs print a message and exit.
    void load();
    // Unload the image data.
    //
    // Safe to call even if the data are not loaded.
    void unload();

    // Return the path of the image.
    [[nodiscard]]
    std::filesystem::path getPath() const;
    // Return the width of the image.
    [[nodiscard]]
    int getWidth() const;
    // Return the height of the image.
    [[nodiscard]]
    int getHeight() const;
    // Return the number of channels of the image.
    [[nodiscard]]
    int getChannels() const;

private:
    std::filesystem::path imagePath{};
    int width{}, height{};
    int channels{};
    float* imageData{ nullptr };
};

// Class representing a padded image.
//
// If default constructed you need to call the pad method before using it.
class PaddedImage {
public:
    PaddedImage() = default;
    // Construct a PaddedImage by padding image with mode by halfSize.
    PaddedImage(const Image& image, PaddingMode mode, int halfSize);

    // Pad image with mode by halfSize.
    void pad(const Image& image, PaddingMode mode, int halfSize);

    // Return a pointer to the padded image data.
    //
    // The data are stored in row major order.
    [[nodiscard]]
    float* data();
    // Return a const pointer to the padded image data.
    //
    // The data are stored in row major order.
    [[nodiscard]]
    const float* data() const;

    // Return the width of the original image.
    [[nodiscard]]
    int getWidth() const;
    // Return the width of the padded image.
    [[nodiscard]]
    int getPaddedWidth() const;
    // Return the height of the original image.
    [[nodiscard]]
    int getHeight() const;
    // Return the height of the padded image.
    [[nodiscard]]
    int getPaddedHeight() const;
    // Return the number of channels.
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

// Transform the floating point values in from to unsigned 8bit integer values in to.
//
// Perform gamma correction, scaling, clamping and rounding of values.
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

// Write the image, whose data are found in data, in path.
//
// width, height and channels specify the image metadata.
//
// The image data must be in row major order.
//
// If an error occurs print a message and exit.
void writeImage(const std::string& path, int width,  int height, int channels, const uint8_t* data);

#endif //IMAGEKERNELPROCESSING_IMAGE_H