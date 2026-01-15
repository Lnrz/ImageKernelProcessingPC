#include "convolution.cuh"
#include <nvfunctional>

__constant__ float deviceFilters[getFiltersSize()];

void updateFilters(const float* data) {
    cudaMemcpyToSymbol(deviceFilters, data, getFiltersSize() * sizeof(float));
}

/*
 * Gli indici dei thread si rifanno ai canali che calcolano
 * TUTTI I THREAD CALCOLANO CANALI
 *
 * Tutti i thread caricano i dati
 *
 * Le dimensioni dei block sono in canali!
 *
 * Le width e height di input sono espressi in pixel
 * Quelli calcolati qui in canali, si opera a livello di canali
 */

__global__ void cudaKernelConvolution(CudaConvolutionData data) {
    extern __shared__ float cache[];

    // load data into shared memory
    // SOURCEXY coordinate del primo channel da caricare, se negative va usato il padding
    // Le coordinate vanno pensate come in riferimento all'immagine non paddata

    const int sourceX{ (data.padding == PaddingMode::None) ? static_cast<int>(blockIdx.x * blockDim.x) :
                      static_cast<int>(blockIdx.x * blockDim.x) - data.halfSize * data.channels };
    const int sourceY{ (data.padding == PaddingMode::None) ? static_cast<int>(blockIdx.y * blockDim.y) :
                      static_cast<int>(blockIdx.y * blockDim.y) - data.halfSize };
    const int imageWidth{ data.inputImageWidth * data.channels };
    const int channelsPerLoad{ static_cast<int>(blockDim.x * blockDim.y) };
    const int cacheWidth{ static_cast<int>(blockDim.x) + 2 * data.halfSize * data.channels };
    const int cacheHeight{ static_cast<int>(blockDim.y) + 2 * data.halfSize };
    const int maxY{ sourceY + cacheHeight };
    const int channelsToLoad{ cacheWidth * cacheHeight };
    const int loadingSteps{ (channelsToLoad + channelsPerLoad - 1) / channelsPerLoad };

    nvstd::function<float(int,int)> paddingFun {
        [] __device__ (int x, int y) {
            return 0.f;
        }
    };
    switch (data.padding) {
        default: break; // if padding is none, zero or unknown default to zero padding
    }

    for (int i{ 0 }; i < loadingSteps; i++) {
        const int increment{ i * channelsPerLoad + static_cast<int>(threadIdx.x + blockDim.x * threadIdx.y) };
        const int incrementX{ increment % cacheWidth };
        const int incrementY{ increment / cacheWidth };
        const int loadX{ sourceX + incrementX };
        const int loadY{ sourceY + incrementY };
        if (loadY >= maxY) continue;
        const int storeIndex{ increment };
        if (loadX < 0 || loadX >= imageWidth ||
            loadY < 0 || loadY >= data.inputImageHeight) {
            cache[storeIndex] = paddingFun(loadX, loadY);
        } else {
            cache[storeIndex] = data.input[loadX + loadY * imageWidth];
        }
    }
    const int outputImageWidth{ (data.padding != PaddingMode::None) ? imageWidth :
                     (data.inputImageWidth - 2 * data.halfSize) * data.channels };
    const int outputImageHeight{ (data.padding != PaddingMode::None) ? data.inputImageHeight :
                                                   data.inputImageHeight - 2 * data.halfSize };
    const int outputX{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
    const int outputY{ static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) };

    __syncthreads();

    if (outputX < outputImageWidth && outputY < outputImageHeight) {
        const int inputIndex{ static_cast<int>(threadIdx.x + threadIdx.y * cacheWidth) };
        const int outputIndex{ outputX + outputY * outputImageWidth };
        const int kernelSize{ 2 * data.halfSize + 1 };
        float outputChannel{ 0.f };
        for (int j{ 0 }; j < kernelSize; j++) {
            for (int i{ 0 }; i < kernelSize; i++) {
                outputChannel += (cache[inputIndex + i * data.channels + j * cacheWidth] * deviceFilters[data.filterOffset + i + j * kernelSize]);
            }
        }
        data.output[outputIndex] = outputChannel;
    }
}