#ifndef IMAGEKERNELPROCESSINGCUDA_TIMER_CUH
#define IMAGEKERNELPROCESSINGCUDA_TIMER_CUH

#include <filesystem>

// Class handling the time measurements of the CUDA program.
class CudaTimer {
public:
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::microseconds;
    using TimePoint = std::chrono::time_point<Clock>;

    // Construct CudaTimer.
    //
    // tasks is the number of tasks to process.
    // enable is a bool specifying if stats are to be recorded.
    // blockSize is the block size used in kernels.
    // inputSlots is the number of slots available in the GPU input buffer.
    // outputSlots is the number of slots available in the GPU output buffer.
    CudaTimer(size_t tasks, bool enable, dim3 blockSize, size_t inputSlots, size_t outputSlots);
    CudaTimer(const CudaTimer& oth) = delete;
    CudaTimer(CudaTimer&& oth) = delete;
    ~CudaTimer();

    CudaTimer& operator=(const CudaTimer& oth) = delete;
    CudaTimer& operator=(CudaTimer&& oth) = delete;

    // Record the start timepoint of the program.
    void startingProgram();
    // Record the timepoint when the image loading starts for the task indexed by taskIndex.
    void startLoadingImage(size_t taskIndex);
    // Record the start convolution event on stream.
    void startConvolutingImageEvent(cudaStream_t stream);
    // Record the end convolution event on stream.
    //
    // The next calls to startConvolutingImageEvent and endConvolutingImageEvent
    // will record the events for the next task.
    void endConvolutingImageEvent(cudaStream_t stream);
    // Record the timepoint when the image writing finishes for the task indexed by taskIndex.
    void endWritingImage(size_t taskIndex);
    // Record the end timepoint of the program.
    void endingProgram();

    // Append in log.txt the stats of the program.
    //
    // Also save in convolutionTimes.bin the convolution times in binary format as float, and in processingTimes.bin
    // the task processing times in binary format as float.
    //
    // All files are saved in folder.
    void writeLog(const std::filesystem::path& folder);

private:
    struct TimingEvents {
        cudaEvent_t startConvolution;
        cudaEvent_t endConvolution;
    };

    struct TimingTimePoints {
        TimePoint startLoading{};
        TimePoint startConvolution{};
        TimePoint endWriting{};
    };

    const size_t tasks;
    const size_t blockX, blockY;
    const size_t inputSlots, outputSlots;
    size_t currentStreamTask{ 0 };
    const bool enable;
    std::vector<TimingTimePoints> tasksTimePoints;
    std::vector<TimingEvents> events;
    std::vector<float> convolutionTimes;
    std::vector<float> processingTimes;
    TimePoint programStart, programEnd;
};

#endif //IMAGEKERNELPROCESSINGCUDA_TIMER_CUH