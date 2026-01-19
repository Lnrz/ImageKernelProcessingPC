#include "utilities.h"
#include <iostream>

[[noreturn]]
void explainProgram() {
    std::cout << R"(Program usage:
ImageKernelProcessing.exe inputFolder outputFolder

inputFolder:
    path to the folder containing the images to process and a tasks.txt file
outputFolder:
    path to the folder where to write the output images

The tasks.txt file contains the tasks specified as lines in the following format:
IMAGE image1 filter1_1:padding1_1
IMAGE image2 filter2_1:padding2_1 filter2_2:padding2_2
...

image is the name of the image to process, contained in the inputFolder
filter is the filter to apply to the image
padding is the padding to apply to the image
It is possible to specify more filters and paddings for the same image by separating them with a whitespace

To activate statistics write in tasks.txt the line:
STATS
The statistics will be written to a log.txt file inside the outputFolder
)";
    exit(-1);
}
