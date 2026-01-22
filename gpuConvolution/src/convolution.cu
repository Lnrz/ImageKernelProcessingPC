#include "convolution.cuh"

__constant__ float deviceFilters[getFiltersSize()];

/*
 * Gli indici dei thread si rifanno ai canali che calcolano
 * TUTTI I THREAD CALCOLANO CANALI
 *
 * Tutti i thread caricano i dati
 *
 * Le dimensioni dei block sono in canali!
 */

__device__ float padData(const CudaConvolutionData& data, int loadX, int loadY, bool isOutOfBoundsX, bool isOutOfBoundsY) {
    switch (data.padding) {
        case PaddingMode::Zero: {
            return 0.f;
        }
        case PaddingMode::Mirror: {
            int mirrorLoadX{ loadX };
            if (isOutOfBoundsX) {
                const int pixelX{ loadX < 0 ?
                    2 * data.inputImageWidth - 2 + abs((loadX - data.channels + 1) / data.channels) :
                    loadX / data.channels };
                const int channelOffset{ loadX < 0 ?
                    data.channels - 1 + ((loadX + 1) % data.channels) : loadX % data.channels };
                const int loadPixelX{ ((pixelX - data.inputImageWidth) / (data.inputImageWidth - 1)) % 2 == 0 ?
                    data.inputImageWidth - 2 - ((pixelX - data.inputImageWidth) % (data.inputImageWidth - 1)) :
                    1 + ((pixelX - data.inputImageWidth) % (data.inputImageWidth - 1)) };

                mirrorLoadX = loadPixelX * data.channels + channelOffset;
            }

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

__device__ void loadDataToSharedMemory(const CudaConvolutionData& data, float* cache) {
    // load data into shared memory
    // SOURCEXY coordinate del primo channel da caricare, se negative va usato il padding
    // Le coordinate vanno pensate come in riferimento all'immagine non paddata
    // di conseguenza possono essere negative o maggiori delle dimensioni dell'immagine di input
    const int sourceX{ (data.padding == PaddingMode::None) ? static_cast<int>(blockIdx.x * blockDim.x) :
                      static_cast<int>(blockIdx.x * blockDim.x) - data.halfSize * data.channels };
    const int sourceY{ (data.padding == PaddingMode::None) ? static_cast<int>(blockIdx.y * blockDim.y) :
                      static_cast<int>(blockIdx.y * blockDim.y) - data.halfSize };
    const int maxY{ min(
        sourceY + data.cacheHeight,
        (data.padding == PaddingMode::None) ? data.inputImageHeight : data.inputImageHeight + data.halfSize
    )};
    const int actualCacheRowSize{ min(
        data.cacheRowSize,
        static_cast<int>(data.outputImageRowSize - blockDim.x * blockIdx.x + 2 * data.halfSize * data.channels)
    )};

    for (int i{ 0 }; i < data.loadingSteps; i++) {
        const int increment{ i * data.channelsPerLoad + static_cast<int>(threadIdx.x + blockDim.x * threadIdx.y) };
        const int incrementX{ increment % actualCacheRowSize };
        const int incrementY{ increment / actualCacheRowSize };
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

__device__ float convolute(const CudaConvolutionData& data, const float* cache) {
    float outputChannel{ 0.f };

    const int inputIndex{ static_cast<int>(threadIdx.x + threadIdx.y * data.cacheRowSize) };
    for (int j{ 0 }; j < data.kernelSize; j++) {
        for (int i{ 0 }; i < data.kernelSize; i++) {
            outputChannel += cache[inputIndex + i * data.channels + j * data.cacheRowSize]
                           * deviceFilters[data.filterOffset + i + j * data.kernelSize];
        }
    }

    return outputChannel;
}

__global__ void cudaKernelConvolution(CudaConvolutionData data) {
    extern __shared__ float cache[];

    loadDataToSharedMemory(data, cache);
    __syncthreads();
    const int outputX{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
    const int outputY{ static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };
    const int outputIndex{ outputX + outputY * data.outputImageRowSize };
    if (outputX < data.outputImageRowSize && outputY < data.outputImageHeight)
        data.output[outputIndex] = convolute(data, cache);
}