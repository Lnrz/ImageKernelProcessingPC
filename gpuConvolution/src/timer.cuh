#ifndef IMAGEKERNELPROCESSINGCUDA_TIMER_CUH
#define IMAGEKERNELPROCESSINGCUDA_TIMER_CUH

#include <filesystem>

class CudaTimer {
public:
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::microseconds;
    using TimePoint = std::chrono::time_point<Clock>;

    CudaTimer(size_t tasks, bool enable, dim3 blockSize, size_t inputSlots, size_t outputSlots);
    CudaTimer(const CudaTimer& oth) = delete;
    CudaTimer(CudaTimer&& oth) = delete;
    ~CudaTimer();

    CudaTimer& operator=(const CudaTimer& oth) = delete;
    CudaTimer& operator=(CudaTimer&& oth) = delete;

    void startingProgram();
    void startLoadingImage(size_t taskIndex);
    void startConvolutingImageEvent(cudaStream_t stream);
    void endConvolutingImageEvent(cudaStream_t stream);
    void endWritingImage(size_t taskIndex);
    void endingProgram();

    void writeLog(const std::filesystem::path& path);

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

    static
    void startConvolutingImageCallback(void* userData);

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