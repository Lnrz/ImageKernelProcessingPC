#ifndef IMAGEKERNELPROCESSINGCUDA_MANAGER_CUH
#define IMAGEKERNELPROCESSINGCUDA_MANAGER_CUH

#include <filesystem>
#include <vector>
#include "image.h"


struct InputBufferSlot {
    float* ptr;
    float* stagingPtr;
    cudaEvent_t transferComplete;
    cudaEvent_t executionFinished;
    std::shared_ptr<Image> image;
};

class InputBuffer {
public:
    InputBuffer(float* basePtr, size_t slotsNum, size_t slotSize);
    InputBuffer(const InputBuffer& oth) = delete;
    InputBuffer(InputBuffer&& oth) = delete;
    ~InputBuffer();

    InputBuffer& operator=(const InputBuffer& oth) = delete;
    InputBuffer& operator=(InputBuffer&& oth) = delete;

    [[nodiscard]]
    bool isImageLoaded(const std::shared_ptr<Image>& image) const;

    [[nodiscard]]
    InputBufferSlot& getImageSlot(const std::shared_ptr<Image>& image);

    [[nodiscard]]
    InputBufferSlot& getAvailableSlot();

private:
    using Slots = std::vector<InputBufferSlot>;

    Slots slots;
};



struct OutputBufferSlot {
    float* ptr;
    cudaEvent_t transferComplete;
    cudaEvent_t executionFinished;
};

class OutputBuffer {
public:
    OutputBuffer(float* basePtr, size_t slotsNum, size_t slotSize);
    OutputBuffer(const OutputBuffer& oth) = delete;
    OutputBuffer(OutputBuffer&& oth) = delete;
    ~OutputBuffer();

    OutputBuffer& operator=(const OutputBuffer& oth) = delete;
    OutputBuffer& operator=(OutputBuffer&& oth) = delete;

    [[nodiscard]]
    OutputBufferSlot& getAvailableSlot();

private:
    using Slots = std::vector<OutputBufferSlot>;

    Slots slots;
};

#endif //IMAGEKERNELPROCESSINGCUDA_MANAGER_CUH