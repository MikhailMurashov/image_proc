# Image processing
**Image processing with CUDA on GPU: to grayscale and gaussian blur**

Build the project with CMake on Unix system:

`$ cmake -G "Unix Makefiles" <path_to_folder_with_CMakeLists>`

`$ cmake --build <path_to_folder_with_Makefile> --target image_proc_cuda`

Command to run:

`./image_proc_cuda <path_to_image>`

To check your CUDA environment use `hello_world` function in `hello_world.cu` (uncomment line 21 in `main.cpp`).

##### Tested on:
 - gcc-7.4.0
 - CMake 3.15
 - OpenCV 4.1.2
 - CUDA 9.1
 
 **!Warning!** If you using CLion IDE install 
 [CLion CUDA Run Patcher](https://plugins.jetbrains.com/plugin/10691-clion-cuda-run-patcher) 
 for running CUDA executable.