#include "grayscale.h"
#include <cstdio>

__global__
void grayscale__kernel(const uchar3 *orig_image, uchar *gray, const uint rows, const uint cols) {
    // calculate coordinates of current pixel
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < cols && y < rows) {
        uint xy = y * cols + x;
        const uchar3 orig_pixel = orig_image[xy];

        uchar B = (float) orig_pixel.x * 0.114f;
        uchar G = (float) orig_pixel.y * 0.587f;
        uchar R = (float) orig_pixel.z * 0.299f;

        gray[xy] = B + G + R;
    }
}

Mat to_grayscale(const Mat& image) {
    Mat gray(image.rows, image.cols, CV_8UC1);

    // creates pointers for gpu to original image and grey
    uchar3 *dev_image;  // using uchar3 for keeping BGR format
    uchar *dev_gray;

    const size_t num_pixels = image.rows * image.cols;

    auto *p_image = new uchar3[num_pixels];
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++) {
            p_image[i * image.cols + j].x = image.at<Vec3b>(i, j)[0];  // B
            p_image[i * image.cols + j].y = image.at<Vec3b>(i, j)[1];  // G
            p_image[i * image.cols + j].z = image.at<Vec3b>(i, j)[2];  // R
        }

    auto start = std::chrono::high_resolution_clock::now();

    // allocate memory on gpu
    cudaMalloc((void **)&dev_image, sizeof(uchar3) * num_pixels);
    cudaMalloc((void **)&dev_gray, sizeof(uchar) * num_pixels);

    // set gray image memory to 0
    cudaMemset(dev_gray, 0, sizeof(uchar) * num_pixels);

    // copy original image to gpu
    cudaMemcpy(dev_image, p_image, sizeof(uchar3) * num_pixels, cudaMemcpyHostToDevice);

    const dim3 blockSize(16, 16);
    const dim3 gridSize(image.cols / blockSize.x + 1, image.rows / blockSize.y + 1);

    // calculate grayscale image
    grayscale__kernel <<< gridSize, blockSize >>> (dev_image, dev_gray, image.rows, image.cols);

    // copy result from gpu
    cudaMemcpy(gray.data, dev_gray, sizeof(uchar) * num_pixels, cudaMemcpyDeviceToHost);

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "GPU grayscale time: " << (float)time.count() / 1000 << " sec" << std::endl;

    // free memory on gpu
    cudaFree(dev_image); cudaFree(dev_gray);

    return gray;
}