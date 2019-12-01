#ifndef __grayscale_h__
#define __grayscale_h__

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;

Mat to_grayscale(const Mat&);

#endif
