#ifndef IMAGEKERNELPROCESSING_TASKLOADER_H
#define IMAGEKERNELPROCESSING_TASKLOADER_H

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
};

ProgramData loadTasks(const std::filesystem::path& tasksFolder);

#endif //IMAGEKERNELPROCESSING_TASKLOADER_H