#ifndef IMAGEKERNELPROCESSINGCUDA_TASKLOADER_CUH
#define IMAGEKERNELPROCESSINGCUDA_TASKLOADER_CUH

#include <unordered_map>
#include "image.h"

struct Task {
    std::shared_ptr<Image> image;
    FilterType filter;
    PaddingMode padding;
};

struct ProgramData {
    std::unordered_map<std::filesystem::path, std::shared_ptr<Image>> images;
    std::vector<Task> tasks;
    dim3 blockSize{ 32, 16 };
    size_t inputSlots{ 2 };
    size_t outputSlots{ 2 };
    bool enableStats{ false };
};

ProgramData loadTasks(const std::filesystem::path& tasksFolder);

#endif //IMAGEKERNELPROCESSINGCUDA_TASKLOADER_CUH