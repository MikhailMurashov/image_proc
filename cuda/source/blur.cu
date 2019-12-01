#include "blur.h"

__global__
void apply_filter(const uchar *input, uchar *output, const uint rows, const uint cols, const double *kernel,
        const uint kernel_size) {
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;

    int half = static_cast<int>(kernel_size / 2);
    double blur = 0.0f;

    for(int i = -half; i <= half; i++)
        for(int j = -half; j <= half; j++) {
            uint orig_x = max(0, min(cols-1, x+j));
            uint orig_y = max(0, min(rows-1, y+i));

            double pixel = input[orig_x + orig_y * cols];
            blur += pixel * kernel[(i+half) * kernel_size + (j+half)];
        }

    output[x + y * cols] = static_cast<uchar>(blur);
}

Mat gaussian_blur(const Mat& image, const uint kernel_size) {
    Mat blur(image.rows, image.cols, CV_8UC1);

    // creates pointers for gpu to original image and blur
    uchar *dev_image;
    uchar *dev_blur;

    Mat kernel = getGaussianKernel(kernel_size, 1) * getGaussianKernel(kernel_size, 1).t();
    double *dev_kernel;

    const size_t num_pixels = image.rows * image.cols;

    auto start = std::chrono::high_resolution_clock::now();

    // allocate memory on gpu
    cudaMalloc((void **)&dev_image, sizeof(uchar) * num_pixels);
    cudaMalloc((void **)&dev_blur, sizeof(uchar) * num_pixels);
    cudaMalloc((void **)&dev_kernel, sizeof(double) * kernel_size * kernel_size);

    // set blur image memory to 0
    cudaMemset(dev_blur, 0, sizeof(uchar) * num_pixels);

    // copy original image and kernel to gpu
    cudaMemcpy(dev_image, image.ptr<uchar>(), sizeof(uchar) * num_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel, kernel.ptr<double>(), sizeof(double) * kernel_size * kernel_size, cudaMemcpyHostToDevice);

    const dim3 blockSize(16, 16);
    const dim3 gridSize(image.cols / blockSize.x, image.rows / blockSize.y);

    // calculate blur image
    apply_filter <<< gridSize, blockSize >>> (dev_image, dev_blur, image.rows, image.cols, dev_kernel, kernel_size);

    // copy result from gpu
    cudaMemcpy(blur.ptr<uchar>(), dev_blur, sizeof(uchar) * num_pixels, cudaMemcpyDeviceToHost);

    // free memory on gpu
    cudaFree(dev_image); cudaFree(dev_blur); cudaFree(dev_kernel);

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "GPU blur time: " << (float)time.count() / 1000 << " sec" << std::endl;

    return blur;
}