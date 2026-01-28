# Kernel Image Processing

Project for the "Parallel Computing" course focusing on how much we can speed up a program by accelerating it with CUDA.

## About the Program

Image linear filtering is a method with which images can be processed.

It consists of applying a filter, or kernel, to every pixel of an image and calculating the convolution of
its coefficients with the neighboring pixels values to calculate an output image.

There are many uses:
+ features extraction,
+ derivatives calculation,
+ data augmentation,
+ visual effects and so on.

Here an example of applying the Laplacian of gaussian 5x5 filter to the image of a lizard.

<img src=img/lizard.jpg width=48% hspace=1%><img src=img/lizard-LoG5-Mirror.jpg width=48% hspace=1%>

## How to Build

**IMPORTANT**\
If you are not using MSVC to compile the code you have to add your compiler flags.\
Also, even if you are using MSVC, there are flags that you may want to change:
+ /arch:AVX2 if your CPU doesn't support it or support something better,
+ /favor:INTEL64 if your CPU isn't an Intel processor.

In the root folder [CMakeLists](CMakeLists.txt) you should add:
```cmake
set(is_my_compiler "$<CXX_COMPILER_ID:MY_COMPILER_ID>")
```
Then in the cpuConvolution [CMakeLists](cpuConvolution/CMakeLists.txt) you should change "target_compile_options" to:
```cmake
target_compile_options(cpuKernelConvolution PRIVATE
        $<$<AND:${is_release},${is_msvc}>:/O2 /Ob3 /fp:fast /arch:AVX2 /Qvec-report:1 /Gv /favor:INTEL64>
        $<$<AND:${is_release},${is_my_compiler}>:my_release_flags>)
```
Then in gpuConvolution [CMakeLists](gpuConvolution/CMakeLists.txt) you should change it to:
```cmake
target_compile_options(gpuKernelConvolution PRIVATE
        $<${is_cuda}:-rdc=true -dlto -restrict -lineinfo -src-in-ptx>
        $<$<AND:${is_cuda},${is_msvc}>:-Xcompiler=/Zc:preprocessor>
        $<$<AND:${is_cuda},${is_release}>:--use_fast_math --extra-device-vectorization>
        $<$<AND:${is_cuda},${is_release},${is_msvc}>:-Xcompiler=/O2,/Ob3,/fp:fast,/arch:AVX2,/favor:INTEL64>
        $<$<AND:${is_cuda},${is_release},${is_my_compiler}>:-Xcompiler=my_comma_separated_release_flags>)
```

---

1. Create a folder called "lib" in "cpuConvolution"
2. Download [Highway](https://github.com/google/highway) and put it into the "lib" folder renaming it "highway"\
   (the program was tested with Highway 1.3.0)
3. Build with CMake\
   (make sure to build in "Release" mode, how to do it depends on which generator you use)
```sh
cmake -B build
cmake --build build --target cpuConvolution
cmake --build build --target gpuConvolution
```


## How to Run

To run the program, doesn't matter which one, you have to specify two folders: the input and output folders.

In the input folder the program expects to find the images to process and a "tasks.txt" file
containing the tasks and settings.

As an example:
```sh
cpuConvolution.exe path/to/the/input/folder path/to/the/output/folder
```
```sh
gpuConvolution.exe path/to/the/input/folder path/to/the/output/folder
```

When run without providing the folders the program will print its usage and exit.

## Settings

The "tasks.txt" file should contain the following lines.

Valid for both programs:
+ "IMAGE name.extension filter:padding [filter:padding...]"\
Add one or more tasks to process the specified image with the specified filters and paddings.
+ "STATS"\
Enables statistics gathering.\
The program will append in "log.txt" descriptive statistics, in "convolutionTimes.bin" the times it took to
calculate the images convolutions and in "processingTimes.bin" the times it took to process the tasks, all in single
precision floating point format.

For cpuConvolution:
+ "NOVECT"\
To disable the use of vectorized code when convoluting.

For gpuConvolution:
+ "BLOCK x y"\
Use as the block dimension (x,y,1) when launching the convolution kernel.\
Default to 32 16.
+ "SLOTS in out"\
Use in slots for the GPU input buffer and out slots for the GPU output buffer.\
Default to 2 2.

Available filters and padding modes are listed in [image.h](common/image.h) (look for the FilterType and PaddingMode enums).

If you want to add your filters just add their values in the FilterType enum, between Invalid and Num, their cases in the
Filter constructor and in the getFilterTypeFromString and getStringFromFilterType functions. All of these can be found
inside [image.h](common/image.h) and [image.cpp](common/image.cpp).

Adding a padding mode is a little trickier. As before you have to add its value in the PaddingMode enum, after Invalid,
then its cases in getPaddingModeFromString and getStringFromPaddingMode, always found in [image.h](common/image.h) and
[image.cpp](common/image.cpp). Then you have to implement the padding itself in the PaddedImage pad method, found in
[image.cpp](common/image.cpp) and in the kernel padding function found in [convolution.cu](gpuConvolution/src/convolution.cu).

## Speedup

The following graphs are box plots of the convolution times for the different programs, tested on 100 2K images.

<img src=img/convolutionTimes.png width=50%><img src=img/convolutionTimesCloseup.png width=50%>

The following graph is the achieved speedup w.r.t. the CPU program based on the median of the recorded convolution times.

<div align=center>
<img src=img/speedup.png width=60%>
</div>