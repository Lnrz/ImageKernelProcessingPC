#include "buffer.cuh"

InputBuffer::InputBuffer(float* basePtr, const size_t slotsNum, const size_t slotSize) {
    slots.resize(slotsNum);
    for (size_t i{ 0 }; auto& slot : slots) {
        slot.ptr = basePtr + i * slotSize;
        cudaEventCreateWithFlags(&slot.executionFinished, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&slot.transferComplete, cudaEventDisableTiming);
        i++;
    }
}

InputBuffer::~InputBuffer() {
    for (auto& slot : slots) {
        cudaEventDestroy(slot.executionFinished);
        cudaEventDestroy(slot.transferComplete);
    }
}

bool InputBuffer::isImageLoaded(const std::shared_ptr<Image> &image) const {
    return std::ranges::any_of(slots,
        [&image](const InputBufferSlot& slot) {
            return image == slot.image;
        }
    );
}

InputBufferSlot & InputBuffer::getImageSlot(const std::shared_ptr<Image> &image) {
    const auto res { std::ranges::find_if(slots,
        [&image](const InputBufferSlot& slot) {
            return image == slot.image;
        }
    )};
    if (res == slots.end()) {
        std::cout << "InputBuffer does not contain image " << image->getPath() << std::endl;
        exit(-1);
    }
    return *res;
}

InputBufferSlot& InputBuffer::getAvailableSlot() {
    while(true) {
        for (auto& slot : slots) {
            if (cudaEventQuery(slot.executionFinished) == cudaSuccess) {
                return slot;
            }
        }
    }
}

OutputBuffer::OutputBuffer(float *basePtr, const size_t slotsNum, const size_t slotSize) {
    slots.resize(slotsNum);
    for (size_t i{ 0 }; auto& slot : slots) {
        slot.ptr = basePtr + i * slotSize;
        cudaEventCreateWithFlags(&slot.executionFinished, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&slot.transferComplete, cudaEventDisableTiming);
        i++;
    }
}

OutputBuffer::~OutputBuffer() {
    for (auto& slot : slots) {
        cudaEventDestroy(slot.executionFinished);
        cudaEventDestroy(slot.transferComplete);
    }
}

OutputBufferSlot & OutputBuffer::getAvailableSlot() {
    while(true) {
        for (auto& slot : slots) {
            if (cudaEventQuery(slot.transferComplete) == cudaSuccess) {
                return slot;
            }
        }
    }
}
