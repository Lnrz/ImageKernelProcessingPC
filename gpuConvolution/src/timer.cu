#include "timer.cuh"
#include <numeric>
#include <fstream>
#include <span>
#include "utilities.cuh"

CudaTimer::CudaTimer(const size_t tasks, const bool enable, const dim3 blockSize, const size_t inputSlots, const size_t outputSlots)
    : tasks{ tasks }
    , blockX{ blockSize.x }, blockY{ blockSize.y }
    , inputSlots{ inputSlots }, outputSlots{ outputSlots }
    , enable{ enable } {
    if (!enable) return;
    using namespace std::string_view_literals;

    convolutionTimes.reserve(tasks);
    processingTimes.reserve(tasks);
    tasksTimePoints.resize(tasks);
    events.resize(tasks);
    for (constexpr auto errorMsg{ "An error occurred while creating the timing events"sv };
        auto& event : events) {
        checkCUDAError(cudaEventCreate(&event.startConvolution), errorMsg);
        checkCUDAError(cudaEventCreate(&event.endConvolution), errorMsg);
    }
}

CudaTimer::~CudaTimer() {
    if (!enable) return;
    using namespace std::string_view_literals;

    for (constexpr auto errorMsg{ "An error occurred while destroying the timing events"sv };
         auto& event : events) {
        checkCUDAError(cudaEventDestroy(event.startConvolution), errorMsg);
        checkCUDAError(cudaEventDestroy(event.endConvolution), errorMsg);
    }
}

void CudaTimer::startingProgram() {
    if (!enable) return;

    programStart = Clock::now();
}

void CudaTimer::startLoadingImage(const size_t taskIndex) {
    if (!enable) return;

    tasksTimePoints[taskIndex].startLoading = Clock::now();
}

void CudaTimer::startConvolutingImageEvent(cudaStream_t stream) {
    if (!enable) return;

    checkCUDAError(cudaLaunchHostFunc(stream,
        [](void* userData) {
            auto* timepoint{ static_cast<TimePoint*>(userData) };
            *timepoint = Clock::now();
        }, &tasksTimePoints[currentStreamTask].startConvolution),
        "An error occurred while scheduling a host function for timing");
    checkCUDAError(cudaEventRecord(events[currentStreamTask].startConvolution, stream),
        "An error occurred while registering a convolution start event for timing");
}

void CudaTimer::endConvolutingImageEvent(cudaStream_t stream) {
    if (!enable) return;

    checkCUDAError(cudaEventRecord(events[currentStreamTask].endConvolution, stream),
        "An error occurred while registering a convolution end event for timing");

    currentStreamTask++;
}

void CudaTimer::endWritingImage(const size_t taskIndex) {
    if (!enable) return;

    tasksTimePoints[taskIndex].endWriting = Clock::now();
}

void CudaTimer::endingProgram() {
    if (!enable) return;

    programEnd = Clock::now();
}

void CudaTimer::writeLog(const std::filesystem::path& folder) {
    if (!enable) return;
    using namespace std::string_view_literals;
    using std::chrono::duration_cast;

    for (constexpr auto errorMsg{ "An error occurred while measuring elapsed time between events"sv };
        const auto& event : events) {
        float convolutionTime;
        checkCUDAError(cudaEventElapsedTime(&convolutionTime, event.startConvolution, event.endConvolution), errorMsg);
        convolutionTimes.push_back(convolutionTime);
        }
    for (const auto& timePoints : tasksTimePoints) {
        const auto processingTime{ timePoints.startLoading != TimePoint{} ?
            duration_cast<Duration>(timePoints.endWriting - timePoints.startLoading).count() / 1000.f :
            duration_cast<Duration>(timePoints.endWriting - timePoints.startConvolution).count() / 1000.f
        };
        processingTimes.push_back(processingTime);
    }

    const auto programTime{
        duration_cast<Duration>(programEnd - programStart).count() / 1000.f
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

    std::ofstream logFile{ folder / "log.txt", std::ios::app };
    std::time_t t{ std::time(nullptr) };
    std::tm tm{ *std::localtime(&t) };
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

    std::ofstream convolutionFile{ folder / "convolutionTimes.bin", std::ios::binary | std::ios::app };
    convolutionFile.write( reinterpret_cast<char*>(convolutionTimes.data()),convolutionTimes.size() * sizeof(float));

    std::ofstream processingFile{ folder / "processingTimes.bin", std::ios::binary | std::ios::app };
    processingFile.write( reinterpret_cast<char*>(processingTimes.data()),processingTimes.size() * sizeof(float));
}
