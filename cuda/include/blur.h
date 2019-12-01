#ifndef __blur_h__
#define __blur_h__

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace cv;

Mat gaussian_blur(const Mat&, const uint = 5);

#endif