# Image processing
**Parallel image processing: to grayscale and gaussian blur**

You can find CUDA implementation in *cuda* folder.

Build the project with CMake on Unix system:

`$ cmake -G "Unix Makefiles" <path_to_folder_with_CMakeLists>`

`$ cmake --build <path_to_folder_with_Makefile> --target image_proc`

Command to run:

`./image_proc <path_to_image> <num_of_threads>`

#### Tested on:
 - gcc-7.4.0
 - CMake 3.15
 - OpenCV 4.1.2
 - OpenMP 4.5

#### Results:

Hardware config: AMD FX-6300 (6 core), NVidia GTX 1070

Image: [link](https://commons.m.wikimedia.org/wiki/File:Fronalpstock_big.jpg), size: 10,109 × 4,542 pixels

Build with *release* CMake flag 

| | Grayscale, sec | Gaussian blur, sec |
| :------: | :------: | :------: |
| CPU 1 thread | 0,256 | 2,451 |
| CPU 2 threads | 0,144 | 1,297 |
| CPU 4 threads | 0,087 | 0,908 |
| GPU | 0,288 | 0,086 |