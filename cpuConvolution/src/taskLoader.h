#ifndef IMAGEKERNELPROCESSING_TASKLOADER_H
#define IMAGEKERNELPROCESSING_TASKLOADER_H

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
    bool enableStats{ false };
    bool disableVect{ false };
};

// Load ProgramData from taskFolder.
//
// If a "tasks.txt" file does not exist in tasksFolder or an error arises while reading it,
// print the error and exit the program.
ProgramData loadTasks(const std::filesystem::path& tasksFolder);

#endif //IMAGEKERNELPROCESSING_TASKLOADER_H