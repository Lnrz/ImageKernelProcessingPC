#ifndef IMAGEKERNELPROCESSINGCUDA_MANAGER_CUH
#define IMAGEKERNELPROCESSINGCUDA_MANAGER_CUH

#include <filesystem>
#include <vector>
#include "image.h"


// Struct containing data specific for a InputBuffer slot.
//
// ptr is the pointer to the start of the slot in GPU memory.
// stagingPtr is the pointer to the staging buffer in CPU memory.
// transferComplete is the event signalling the transfer of the image to GPU memory.
// executionFinished is the event signalling the end of the processing of the image.
// image is a shared pointer to the image stored in the slot.
struct InputBufferSlot {
    float* ptr;
    float* stagingPtr;
    cudaEvent_t transferComplete;
    cudaEvent_t executionFinished;
    std::shared_ptr<Image> image;
};

// Class representing the input buffer on GPU.
class InputBuffer {
public:
    // Construct InputBuffer.
    //
    // basePtr is the start of the InputBuffer in GPU memory.
    // slotsNum is the number of slots in the InputBuffer.
    // slotSize is the size of a slot in float.
    InputBuffer(float* basePtr, size_t slotsNum, size_t slotSize);
    InputBuffer(const InputBuffer& oth) = delete;
    InputBuffer(InputBuffer&& oth) = delete;
    ~InputBuffer();

    InputBuffer& operator=(const InputBuffer& oth) = delete;
    InputBuffer& operator=(InputBuffer&& oth) = delete;

    // Return true if image is present in a slot.
    [[nodiscard]]
    bool isImageLoaded(const std::shared_ptr<Image>& image) const;

    // Return the slot containing image.
    //
    // If no slot contains the image print an error and exit.
    [[nodiscard]]
    InputBufferSlot& getImageSlot(const std::shared_ptr<Image>& image);

    // Return a slot in round-robin order.
    [[nodiscard]]
    InputBufferSlot& getSlot();

private:
    using Slots = std::vector<InputBufferSlot>;

    Slots slots;
    size_t currentSlot{ 0 };
};


// Struct containing data specific for an OutputBuffer slot.
//
// ptr is the pointer to the start of the slot in GPU memory.
// transferComplete is the event signalling the transfer of the image to CPU memory.
// executionFinished is the event signalling the end of the processing of the image.
struct OutputBufferSlot {
    float* ptr;
    cudaEvent_t transferComplete;
    cudaEvent_t executionFinished;
};

// Class representing the output buffer on GPU.
class OutputBuffer {
public:
    // Construct OutputBuffer.
    //
    // basePtr is the start of the OutputBuffer in GPU memory.
    // slotsNum is the number of slots in the OutputBuffer.
    // slotSize is the size of a slot in float.
    OutputBuffer(float* basePtr, size_t slotsNum, size_t slotSize);
    OutputBuffer(const OutputBuffer& oth) = delete;
    OutputBuffer(OutputBuffer&& oth) = delete;
    ~OutputBuffer();

    OutputBuffer& operator=(const OutputBuffer& oth) = delete;
    OutputBuffer& operator=(OutputBuffer&& oth) = delete;

    // Return a slot in round-robin order.
    [[nodiscard]]
    OutputBufferSlot& getSlot();

private:
    using Slots = std::vector<OutputBufferSlot>;

    Slots slots;
    size_t currentSlot{ 0 };
};

#endif //IMAGEKERNELPROCESSINGCUDA_MANAGER_CUH