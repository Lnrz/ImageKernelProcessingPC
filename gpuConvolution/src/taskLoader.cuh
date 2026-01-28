#ifndef IMAGEKERNELPROCESSINGCUDA_TASKLOADER_CUH
#define IMAGEKERNELPROCESSINGCUDA_TASKLOADER_CUH

#include <unordered_map>
#include "image.h"

// Struct representing a task.
struct Task {
    std::shared_ptr<Image> image;
    FilterType filter;
    PaddingMode padding;
};

// Struct containing the data and settings of the program.
struct ProgramData {
    std::unordered_map<std::filesystem::path, std::shared_ptr<Image>> images;
    std::vector<Task> tasks;
    dim3 blockSize{ 32, 16 };
    size_t inputSlots{ 2 };
    size_t outputSlots{ 2 };
    bool enableStats{ false };
};

// Load ProgramData from taskFolder.
//
// If a "tasks.txt" file does not exist in tasksFolder or an error arises while reading it,
// print the error and exit the program.
ProgramData loadTasks(const std::filesystem::path& tasksFolder);

#endif //IMAGEKERNELPROCESSINGCUDA_TASKLOADER_CUH