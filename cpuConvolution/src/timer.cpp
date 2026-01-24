#include "timer.h"
#include <fstream>
#include <numeric>

Timer::Timer(const size_t tasks, const size_t lanes, const bool disableVect)
    : tasks{ tasks }, lanes{ lanes }, disableVect{ disableVect } {
    processingTimes.reserve(tasks);
    convolutionTimes.reserve(tasks);
}

void Timer::startingProgram() {
    programStart = std::chrono::steady_clock::now();
}

void Timer::startingImageLoading() {
    imageLoadStart = std::chrono::steady_clock::now();
}

void Timer::startingImageConvolution() {
    imageConvolutionStart = std::chrono::steady_clock::now();
}

void Timer::imageConvolutionEnded() {
    imageConvolutionEnd = std::chrono::steady_clock::now();

    convolutionTimes.push_back(
        std::chrono::duration_cast<Duration>(imageConvolutionEnd - imageConvolutionStart).count()
        / conversionFactor
    );
}

void Timer::imageWritten() {
    imageWriteEnd = std::chrono::steady_clock::now();

    processingTimes.push_back(
        std::chrono::duration_cast<Duration>(imageWriteEnd - imageLoadStart).count() / conversionFactor
    );
}

void Timer::endingProgram() {
    programEnd = std::chrono::steady_clock::now();

    programTime = std::chrono::duration_cast<Duration>(programEnd - programStart).count() / conversionFactor;
}

void Timer::writeLog(const std::filesystem::path& folder) const {
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
    logFile
    << std::format("Date:{0:%F}  Time:{0:%R}  Tasks:{1}  ProgramTime:{2}ms\n",
        std::chrono::system_clock::now(), tasks, programTime)
    << std::format("CPU  Lanes:{}  NoVect:{}\n", lanes, disableVect)
    << std::format("ConvolutionTimes:[ Mean:{}ms  Std:{}ms  Max:{}ms  Min:{}ms ]\n",
        meanConvolutionTime, stdConvolutionTime, maxConvolutionTime, minConvolutionTime)
    << std::format("ProcessingTimes:[ Mean:{}ms  Std:{}ms  Max:{}ms  Min:{}ms ]\n",
        meanProcessingTime, stdProcessingTime, maxProcessingTime, minProcessingTime)
    << std::endl;

    std::ofstream convolutionFile{ folder / "convolutionTimes.bin", std::ios::binary | std::ios::app };
    convolutionFile.write( reinterpret_cast<const char*>(convolutionTimes.data()),convolutionTimes.size() * sizeof(float));

    std::ofstream processingFile{ folder / "processingTimes.bin", std::ios::binary | std::ios::app };
    processingFile.write( reinterpret_cast<const char*>(processingTimes.data()),processingTimes.size() * sizeof(float));
}
