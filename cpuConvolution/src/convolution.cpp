#include "convolution.h"
#include <hwy/highway.h>

size_t getCPULanes() {
    using namespace hwy::HWY_NAMESPACE;
    using D = ScalableTag<float>;

    return Lanes(D{});
}

void kernelConvolution(const ConvolutionData &data) {
    using namespace hwy::HWY_NAMESPACE;
    using D = ScalableTag<float>;
    using V = VFromD<D>;

    constexpr D d{};
    const size_t vectorLength{ Lanes(d) };
    const size_t vectorsInRow{ data.rowSize / vectorLength };
    const size_t remainingChannels{ data.rowSize - vectorsInRow * vectorLength };
    const auto kernelSize{ 2 * data.halfSize + 1 };
    const auto paddedRowSize{ data.rowSize + 2 * data.halfSize * data.channels };

    for (int row{ data.halfSize }; row < data.rowNum + data.halfSize; row++) {
        for (size_t vector{ 0 }; vector < vectorsInRow; vector++) {
            V accum{ Set(d, 0) };
            for (int i{ -data.halfSize }; i <= data.halfSize; i++) {
                for (int j{ 0 }; j < kernelSize; j++) {
                    const V pixelData{ Load(d, data.inPtr
                        + vector * vectorLength
                        + (row + i) * paddedRowSize + j * data.channels)};
                    const V kernelCoef{ Set(d, data.coefPtr[(i + data.halfSize) * kernelSize + j]) };
                    accum = MulAdd(kernelCoef, pixelData, accum);
                }
            }
            Store(accum, d, data.outPtr
                + (row - data.halfSize) * data.rowSize
                + vector * vectorLength);
        }
        if (remainingChannels <= 0) continue;

        V accum{ Set(d, 0) };
        for (int i{ -data.halfSize }; i < data.halfSize; i++) {
            for (int j{ 0 }; j < kernelSize; j++) {
                const V pixelData{ LoadN(d, data.inPtr
                    + vectorsInRow * vectorLength
                    + (row + i) * paddedRowSize + j * data.channels, remainingChannels)};
                const V kernelCoef{ Set(d, data.coefPtr[(i + data.halfSize) * kernelSize + j]) };
                accum = MulAdd(kernelCoef, pixelData, accum);
            }
        }
        StoreN(accum, d, data.outPtr
            + (row - data.halfSize) * data.rowSize
            + vectorsInRow * vectorLength, remainingChannels);
    }
}
