#include "buffer.cuh"
#include "utilities.cuh"

InputBuffer::InputBuffer(float* basePtr, const size_t slotsNum, const size_t slotSize) {
    using namespace std::string_view_literals;

    slots.resize(slotsNum);
    for (size_t i{ 0 }; auto& slot : slots) {
        slot.ptr = basePtr + i * slotSize;
        constexpr auto errorMsg{ "An error occurred while creating the input slots events"sv };
        checkCUDAError(cudaEventCreateWithFlags(&slot.executionFinished, cudaEventDisableTiming), errorMsg);
        checkCUDAError(cudaEventCreateWithFlags(&slot.transferComplete, cudaEventDisableTiming), errorMsg);
        i++;
    }
}

InputBuffer::~InputBuffer() {
    using namespace  std::string_view_literals;

    for (constexpr auto errorMsg{ "An error occurred while destroying the input slots events"sv };
         auto& slot : slots) {
        checkCUDAError(cudaEventDestroy(slot.executionFinished), errorMsg);
        checkCUDAError(cudaEventDestroy(slot.transferComplete), errorMsg);
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
            switch (const auto res { cudaEventQuery(slot.executionFinished) }; res) {
                case cudaSuccess: {
                    return slot;
                }
                case cudaErrorNotReady: {
                    continue;
                }
                default: {
                    checkCUDAError(res, "An error occurred while checking for an available input slot");
                }
            }
        }
    }
}

OutputBuffer::OutputBuffer(float *basePtr, const size_t slotsNum, const size_t slotSize) {
    using namespace std::string_view_literals;

    slots.resize(slotsNum);
    for (size_t i{ 0 }; auto& slot : slots) {
        slot.ptr = basePtr + i * slotSize;
        constexpr auto errorMsg{ "An error occurred while creating the output slots events"sv };
        checkCUDAError(cudaEventCreateWithFlags(&slot.executionFinished, cudaEventDisableTiming), errorMsg);
        checkCUDAError(cudaEventCreateWithFlags(&slot.transferComplete, cudaEventDisableTiming), errorMsg);
        i++;
    }
}

OutputBuffer::~OutputBuffer() {
    using namespace  std::string_view_literals;

    for (constexpr auto errorMsg{ "An error occurred while destroying the output slots events"sv };
         auto& slot : slots) {
        checkCUDAError(cudaEventDestroy(slot.executionFinished), errorMsg);
        checkCUDAError(cudaEventDestroy(slot.transferComplete), errorMsg);
    }
}

OutputBufferSlot& OutputBuffer::getAvailableSlot() {
    while(true) {
        for (auto& slot : slots) {
            switch (const auto res { cudaEventQuery(slot.transferComplete) }; res) {
                case cudaSuccess: {
                    return slot;
                }
                case cudaErrorNotReady: {
                    continue;
                }
                default: {
                    checkCUDAError(res, "An error occurred while checking for an available output slot");
                }
            }
        }
    }
}
