#ifndef IMAGEKERNELPROCESSINGCUDA_UTILITIES_CUH
#define IMAGEKERNELPROCESSINGCUDA_UTILITIES_CUH

#include <string_view>

void checkCUDAError(cudaError_t error, std::string_view msg);

[[noreturn]]
void explainProgram();

#endif //IMAGEKERNELPROCESSINGCUDA_UTILITIES_CUH