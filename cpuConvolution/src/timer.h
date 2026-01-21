#ifndef IMAGEKERNELPROCESSINGCUDA_TIMER_H
#define IMAGEKERNELPROCESSINGCUDA_TIMER_H

#include <filesystem>

class Timer {
public:
    using Duration = std::chrono::microseconds;
    using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

    explicit
    Timer(size_t tasks, size_t lanes, bool noVect);

    void startingProgram();
    void startingImageLoading();
    void startingImageConvolution();
    void imageConvolutionEnded();
    void imageWritten();
    void endingProgram();

    void writeLog(const std::filesystem::path& path) const;

private:
    const size_t tasks{};
    const size_t lanes{};
    const bool disableVect{};
    std::vector<float> processingTimes;
    std::vector<float> convolutionTimes;
    float programTime{};
    const float conversionFactor{ 1000.f };
    TimePoint imageLoadStart;
    TimePoint imageConvolutionStart, imageConvolutionEnd;
    TimePoint imageWriteEnd;
    TimePoint programStart, programEnd;
};

#endif //IMAGEKERNELPROCESSINGCUDA_TIMER_H
