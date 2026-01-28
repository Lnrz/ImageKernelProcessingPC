#include "convolution.cuh"

__constant__ float deviceFilters[getFiltersSize()];


__device__ float padData(const CudaConvolutionData& data, int loadX, int loadY, bool isOutOfBoundsX, bool isOutOfBoundsY) {
    switch (data.padding) {
        case PaddingMode::Zero: {
            return 0.f;
        }
        case PaddingMode::Mirror: {
            // mirrorLoadX is the x coordinate of the channel to load
            int mirrorLoadX{ loadX };
            if (isOutOfBoundsX) {
                // pixelX is the x coordinate of the pixel to which the channel of x coordinate loadX belongs
                // in case the pixel x coordinate is negative we map it to a corresponding positive one
                // -1, -2, -3 correspond to 2width-1, 2width, 2width+1
                // this is done in order not to have to differentiate between negative and positive out of border coordinates
                const int pixelX{ loadX < 0 ?
                    2 * data.inputImageWidth - 2 + abs((loadX - data.channels + 1) / data.channels) :
                    loadX / data.channels };
                // channelOffset is the channel index to which the channel of x coordinate loadX corresponds
                // red would be 0, green 1, blue 2
                // if the image is grayscale it is always 0
                const int channelOffset{ loadX < 0 ?
                    data.channels - 1 + ((loadX + 1) % data.channels) : loadX % data.channels };
                // loadPixelX is the pixel x coordinate inside the input image
                // which corresponds to the positive but out of bounds pixel x coordinate pixelX
                // width, width+1, width+2 would result in width-2, width-3, width-4
                // 2width-1, 2width, 2width+1 would result in 1, 2, 3
                const int loadPixelX{ ((pixelX - data.inputImageWidth) / (data.inputImageWidth - 1)) % 2 == 0 ?
                    data.inputImageWidth - 2 - ((pixelX - data.inputImageWidth) % (data.inputImageWidth - 1)) :
                    1 + ((pixelX - data.inputImageWidth) % (data.inputImageWidth - 1)) };

                mirrorLoadX = loadPixelX * data.channels + channelOffset;
            }

            // the same comments about the x coordinate are valid for the y coordinate
            // here there is no channel offset since the channels are interleaved only along the width
            int mirrorLoadY{ loadY };
            if (isOutOfBoundsY) {
                const int pixelY{ loadY < 0 ?
                    2 * data.inputImageHeight - 2 + abs(loadY ) : loadY };
                const int loadPixelY{ ((pixelY - data.inputImageHeight) / (data.inputImageHeight - 1)) % 2 == 0 ?
                    data.inputImageHeight - 2 - ((pixelY - data.inputImageHeight) % (data.inputImageHeight - 1)) :
                    1 + ((pixelY - data.inputImageHeight) % (data.inputImageHeight - 1)) };

                mirrorLoadY = loadPixelY;
            }

            return data.input[mirrorLoadX + mirrorLoadY * data.inputImageRowSize];
        }
        default: {
#ifdef NDEBUG
            __builtin_unreachable();
#else
            printf("Reached unreachable code in padData\nBlock(%d,%d)  Thread(%d,%d)  Padding %d\n",
                blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, static_cast<PaddingModeInt>(data.padding));
            return 0.f;
#endif
        }
    }
}

__device__ void loadDataToSharedMemory(const CudaConvolutionData& data, float* cache, const int cacheRowSize) {
    // coordinates are expressed w.r.t. the input image with (0,0) being the coordinates of the top-left channel
    // (sourceX,sourceY) are the coordinates of the top-left channel of the neighborhood to load
    const int sourceX{ (data.padding == PaddingMode::None) ? static_cast<int>(blockIdx.x * blockDim.x) :
                      static_cast<int>(blockIdx.x * blockDim.x) - data.halfSize * data.channels };
    const int sourceY{ (data.padding == PaddingMode::None) ? static_cast<int>(blockIdx.y * blockDim.y) :
                      static_cast<int>(blockIdx.y * blockDim.y) - data.halfSize };
    // maximum y coordinate of the neighborhood to load
    const int maxY{ min(
        sourceY + data.cacheHeight,
        (data.padding == PaddingMode::None) ? data.inputImageHeight : data.inputImageHeight + data.halfSize
    )};

    for (int i{ 0 }; i < data.loadingSteps; i++) {
        const int increment{ i * data.channelsPerLoad + static_cast<int>(threadIdx.x + blockDim.x * threadIdx.y) };
        const int incrementX{ increment % cacheRowSize };
        const int incrementY{ increment / cacheRowSize };
        const int loadX{ sourceX + incrementX };
        const int loadY{ sourceY + incrementY };
        if (loadY >= maxY) continue;
        const int storeIndex{ increment };
        const bool isOutOfBoundsX{ loadX < 0 || loadX >= data.inputImageRowSize };
        const bool isOutOfBoundsY{ loadY < 0 || loadY >= data.inputImageHeight };
        if (!(isOutOfBoundsX || isOutOfBoundsY)) {
            cache[storeIndex] = data.input[loadX + loadY * data.inputImageRowSize];
        } else {
            cache[storeIndex] = padData(data, loadX, loadY, isOutOfBoundsX, isOutOfBoundsY);
        }
    }
}

__device__ float convolute(const CudaConvolutionData& data, const float* cache, const int cacheRowSize) {
    float outputChannel{ 0.f };

    // inputIndex is the index in shared memory of the channel corresponding to the top-left filter coefficient
    const int inputIndex{ static_cast<int>(threadIdx.x + threadIdx.y * cacheRowSize) };
    for (int j{ 0 }; j < data.kernelSize; j++) {
        for (int i{ 0 }; i < data.kernelSize; i++) {
            outputChannel += cache[inputIndex + i * data.channels + j * cacheRowSize]
                           * deviceFilters[data.filterOffset + i + j * data.kernelSize];
        }
    }

    return outputChannel;
}

__global__ void cudaKernelConvolution(CudaConvolutionData data) {
    extern __shared__ float cache[];

    // for the border blocks the neighborhood to load is actually smaller
    const int actualCacheRowSize{ min(
        data.cacheRowSize,
        static_cast<int>(data.outputImageRowSize - blockDim.x * blockIdx.x + 2 * data.halfSize * data.channels)
    )};
    loadDataToSharedMemory(data, cache, actualCacheRowSize);
    __syncthreads();
    // (outputX,outputY) are the coordinates of the channel to which the thread corresponds
    // expressed w.r.t. the output image
    const int outputX{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
    const int outputY{ static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };
    const int outputIndex{ outputX + outputY * data.outputImageRowSize };
    if (outputX < data.outputImageRowSize && outputY < data.outputImageHeight)
        data.output[outputIndex] = convolute(data, cache, actualCacheRowSize);
}