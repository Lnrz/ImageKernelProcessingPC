#ifndef IMAGEKERNELPROCESSINGCUDA_TIMER_H
#define IMAGEKERNELPROCESSINGCUDA_TIMER_H

#include <filesystem>

// Class that handles the time measurements of the program.
class Timer {
public:
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::microseconds;
    using TimePoint = std::chrono::time_point<Clock>;

    // Construct the timer.
    //
    // tasks is the number of tasks to process.
    // lanes is the number of available SIMD float lanes.
    // noVect is a bool specifying if the vector instructions are used in the program.
    explicit
    Timer(size_t tasks, size_t lanes, bool noVect);

    // Record the start timepoint of the program.
    void startingProgram();
    // Record the timepoint when the image loading starts.
    void startingImageLoading();
    // Record the start timepoint of the convolution.
    void startingImageConvolution();
    // Record the end timepoint of the convolution.
    //
    // Also store the time elapsed between the start and end convolution timepoints.
    void imageConvolutionEnded();
    // Record the timepoint when the image writing finishes.
    //
    // Also store the time elapsed between the image loading and the image writing timepoints.
    void imageWritten();
    // Record the end timepoint of the program.
    //
    // Also store the time elapsed between the start and end timepoints of the program.
    // Subsequent calls will overwrite the previous value.
    void endingProgram();

    // Append in log.txt the stats of the program.
    //
    // Also save in convolutionTimes.bin the convolution times in binary format as float, and in processingTimes.bin
    // the task processing times in binary format as float.
    //
    // All files are saved in folder.
    void writeLog(const std::filesystem::path& folder) const;

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
