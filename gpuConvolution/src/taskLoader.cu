#include "taskLoader.cuh"
#include <fstream>
#include <iostream>


ProgramData loadTasks(const std::filesystem::path& tasksFolder) {
    ProgramData data;

    const auto tasksFilePath = tasksFolder / "tasks.txt";
    std::ifstream tasksFileStream(tasksFilePath);
    if (!tasksFileStream.is_open()) {
        std::cerr << "Could not open tasks.txt file at: " << tasksFilePath << std::endl;
        exit(-1);
    }

    std::string keywords{ "IMAGE,BLOCK,SLOTS,STATS" };
    std::string word;
    tasksFileStream >> word;
    do {
        if (word == "BLOCK") {
            tasksFileStream >> data.blockSize.x;
            tasksFileStream >> data.blockSize.y;
        } else if (word == "SLOTS") {
            tasksFileStream >> data.inputSlots;
            tasksFileStream >> data.outputSlots;
        } else if (word == "STATS") {
            data.enableStats = true;
        } else if (word == "IMAGE") {
            tasksFileStream >> word;
            const auto imagePath{ tasksFolder / word };
            std::shared_ptr<Image> imagePtr;
            if (data.images.contains(imagePath)) {
                imagePtr = data.images[imagePath];
            } else {
                imagePtr = std::make_shared<Image>(imagePath);
                data.images.emplace(imagePath, imagePtr);
            }
            tasksFileStream >> word;
            while (keywords.find(word) == std::string::npos && tasksFileStream) {
                auto filter{ FilterType::Invalid };
                auto padding{ PaddingMode::None };
                if (const auto sep{ word.find(':')}; sep != std::string::npos) {
                    padding = getPaddingModeFromString(word.substr(sep + 1, word.length() - sep - 1));
                    word = word.substr(0, sep);
                }
                filter = getFilterTypeFromString(word);
                data.tasks.emplace_back(imagePtr, filter, padding);
                tasksFileStream >> word;
            }
            continue;
        }
        tasksFileStream >> word;
    }
    while (tasksFileStream);

    if (data.inputSlots == 0 || data.outputSlots == 0) {
        std::cerr << "GPU buffers must have at least 1 slot" << std::endl;
        std::cerr << "Input buffers slots was " << data.inputSlots << std::endl;
        std::cerr << "Output buffers slots was " << data.outputSlots << std::endl;
        exit(-1);
    }
    if (data.blockSize.x == 0 || data.blockSize.y == 0) {
        std::cerr << "Block size must not have any dimension of size 0" << std::endl;
        std::cerr << "Block size was (" << data.blockSize.x << "," << data.blockSize.y << ")" << std::endl;
        exit(-1);
    }

    return data;
}
