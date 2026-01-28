#ifndef IMAGEKERNELPROCESSINGCUDA_UTILITIES_CUH
#define IMAGEKERNELPROCESSINGCUDA_UTILITIES_CUH

#include <cuda.h>
#include <string_view>

// Check if error is cudaSuccess.
//
// If it is not, print msg, the error message and then exit.
void checkCUDAError(cudaError_t error, std::string_view msg);

// Check if error is CUDA_SUCCESS.
//
// If it is not, print msg, the error message and then exit.
void checkCUDAError(CUresult error, std::string_view msg);

// Print the program usage and exit.
[[noreturn]]
void explainProgram();

#endif //IMAGEKERNELPROCESSINGCUDA_UTILITIES_CUH