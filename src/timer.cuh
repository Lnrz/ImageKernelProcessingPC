#ifndef IMAGEKERNELPROCESSINGCUDA_TIMER_CUH
#define IMAGEKERNELPROCESSINGCUDA_TIMER_CUH

#include <filesystem>

struct TimingEvents {
    cudaEvent_t startLoading, endWriting;
    cudaEvent_t startConvolution, endConvolution;
};

class CudaTimer {
public:
    using Duration = std::chrono::microseconds;
    using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

    CudaTimer(size_t tasks, bool enable, dim3 blockSize, size_t inputSlots, size_t outputSlots);
    ~CudaTimer();

    void startingProgram();
    void startLoadingImageEvent(cudaStream_t stream);
    void startConvolutingImageEvent(cudaStream_t stream);
    void endConvolutingImageEvent(cudaStream_t stream);
    void endWritingImageEvent(cudaStream_t stream);
    void endingProgram();

    void writeLog(const std::filesystem::path& path);

private:
    const size_t tasks;
    const size_t blockX, blockY;
    const size_t inputSlots, outputSlots;
    size_t currentTask{ 0 };
    const bool enable;
    std::vector<TimingEvents> events;
    std::vector<float> convolutionTimes;
    std::vector<float> processingTimes;
    TimePoint programStart, programEnd;
};

#endif //IMAGEKERNELPROCESSINGCUDA_TIMER_CUH