#include "timer.cuh"
#include "utilities.cuh"
#include <numeric>
#include <fstream>

CudaTimer::CudaTimer(const size_t tasks, const bool enable, const dim3 blockSize, const size_t inputSlots, const size_t outputSlots)
    : tasks{ tasks }
    , blockX{ blockSize.x }, blockY{ blockSize.y }
    , inputSlots{ inputSlots }, outputSlots{ outputSlots }
    , enable{ enable } {
    if (!enable) return;
    using namespace std::string_view_literals;

    convolutionTimes.reserve(tasks);
    processingTimes.reserve(tasks);
    events.resize(tasks);
    for (constexpr auto errorMsg{ "An error occurred while creating the events for timing"sv };
        auto& event : events) {
        checkCUDAError(cudaEventCreate(&event.startLoading), errorMsg);
        checkCUDAError(cudaEventCreate(&event.startConvolution), errorMsg);
        checkCUDAError(cudaEventCreate(&event.endConvolution), errorMsg);
        checkCUDAError(cudaEventCreate(&event.endWriting), errorMsg);
    }
}

CudaTimer::~CudaTimer() {
    if (!enable) return;

    for (auto& event : events) {
        cudaEventDestroy(event.startLoading);
        cudaEventDestroy(event.startConvolution);
        cudaEventDestroy(event.endConvolution);
        cudaEventDestroy(event.endWriting);
    }
}

void CudaTimer::startingProgram() {
    programStart = std::chrono::steady_clock::now();
}

void CudaTimer::startLoadingImageEvent(cudaStream_t stream) {
    if (!enable) return;

    cudaEventRecord(events[currentTask].startLoading, stream);
}

void CudaTimer::startConvolutingImageEvent(cudaStream_t stream) {
    if (!enable) return;

    cudaEventRecord(events[currentTask].startConvolution, stream);
}

void CudaTimer::endConvolutingImageEvent(cudaStream_t stream) {
    if (!enable) return;

    cudaEventRecord(events[currentTask].endConvolution, stream);
}

void CudaTimer::endWritingImageEvent(cudaStream_t stream) {
    if (!enable) return;

    cudaEventRecord(events[currentTask].endWriting, stream);
    currentTask++;
}

void CudaTimer::endingProgram() {
    programEnd = std::chrono::steady_clock::now();
}

void CudaTimer::writeLog(const std::filesystem::path& path) {
    if (!enable) return;
    using namespace  std::string_view_literals;

    for (constexpr auto errorMsg{ "An error occurred while measuring elapsed time between events"sv };
        const auto& event : events) {
        float convolutionTime, processingTime;
        checkCUDAError(cudaEventElapsedTime(&convolutionTime, event.startConvolution, event.endConvolution), errorMsg);
        checkCUDAError(cudaEventElapsedTime(&processingTime, event.startLoading, event.endWriting), errorMsg);
        convolutionTimes.push_back(convolutionTime);
        processingTimes.push_back(processingTime);
    }

    const auto programTime{
        std::chrono::duration_cast<Duration>(programEnd - programStart).count() / 1000.f
    };
    const auto [minConvolutionTime, maxConvolutionTime] = std::ranges::minmax(convolutionTimes);
    const auto [minProcessingTime, maxProcessingTime] = std::ranges::minmax(processingTimes);
    const auto meanConvolutionTime {
        std::accumulate(convolutionTimes.begin(), convolutionTimes.end(), 0.f) / tasks
    };
    const auto meanProcessingTime {
        std::accumulate(processingTimes.begin(), processingTimes.end(), 0.f) / tasks
    };
    const auto stdConvolutionTime{
        std::sqrtf(
            std::accumulate(convolutionTimes.begin(), convolutionTimes.end(), 0.f,
                [meanConvolutionTime](float x, float y) {
                    return x + std::powf(y - meanConvolutionTime, 2.f);
                }) / tasks
        )
    };
    const auto stdProcessingTime{
        std::sqrtf(
            std::accumulate(processingTimes.begin(), processingTimes.end(), 0.f,
                [meanProcessingTime](float x, float y) {
                    return x + std::powf(y - meanProcessingTime, 2.f);
                }) / tasks
        )
    };

    std::ofstream logFile{ path, std::ios::app };
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    logFile
    << std::put_time(&tm, "Date:%Y-%m-%d  Time:%H:%M")
    << std::format("  Tasks:{}  ProgramTime:{}ms\n",
        tasks, programTime)
    << std::format("GPU  Block:({},{})  InputSlots:{}  OutputSlots:{}\n",
        blockX, blockY, inputSlots, outputSlots)
    << std::format("ConvolutionTimes:[ Mean:{}ms  Std:{}ms  Max:{}ms  Min:{}ms ]\n",
        meanConvolutionTime, stdConvolutionTime, maxConvolutionTime, minConvolutionTime)
    << std::format("ProcessingTimes:[ Mean:{}ms  Std:{}ms  Max:{}ms  Min:{}ms ]\n",
        meanProcessingTime, stdProcessingTime, maxProcessingTime, minProcessingTime)
    << std::endl;
}
