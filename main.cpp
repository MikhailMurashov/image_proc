#include <iostream>
#include <omp.h>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat to_grayscale(const Mat& image) {
    Mat grayscale_img = Mat(image.rows, image.cols, CV_8U);

    auto start = std::chrono::high_resolution_clock::now();

    int i, j;

    #pragma omp parallel for private(j)
    for (i = 0; i < image.rows; i++)
        #pragma omp parallel for
        for (j = 0; j < image.cols; j++) {
            uchar B = 0.114 * image.at<Vec3b>(i,j)[0];
            uchar R = 0.299 * image.at<Vec3b>(i,j)[1];
            uchar G = 0.587 * image.at<Vec3b>(i,j)[2];
            grayscale_img.at<uchar>(i,j) = B + R + G;
        }

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cout << "Grayscale time: " << (float)time.count() / 1000 << " sec" << endl;
    return grayscale_img;
}

Mat gaussian_blur(const Mat& image, const int kernel_size = 3) {
    Mat kernel = getGaussianKernel(kernel_size, 1) * getGaussianKernel(kernel_size, 1).t();
    Mat blur; image.copyTo(blur);

    int i, j, k, l;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for private(j, k, l)
    for(i = kernel_size/2; i < image.rows - kernel_size/2; i++)
        #pragma omp parallel for private(k, l)
        for(j = kernel_size/2; j < image.cols - kernel_size/2; j++) {
            double p = 0;

            for(k = 0; k < kernel_size; k++)
                for(l = 0; l < kernel_size; l++) {
                    double pixel = image.at<uchar>(i + k - kernel_size/2, j + l - kernel_size/2);
                    p += pixel * kernel.at<double>(k, l);
                }

            blur.at<uchar>(i,j) = p;
        }

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    cout << "Blur time: " << (float)time.count() / 1000 << " sec" << endl;

    return blur;
}


int main(int argc, char* argv[]) {
    Mat image;
    if (argc >= 2)
        image = imread(argv[1]);
    if (image.empty()) {
        cerr << "Could not open file" << endl
             << "Please provide valid path to image in second argument" << endl;
        return -1;
    }

    if (argc == 3) {
        cout << "Set threads number to " << argv[2] << endl;
        omp_set_num_threads(atoi(argv[2]));
    } else {
        cout << "Using system threads number: " << omp_get_max_threads() << endl;
    }

    Mat gray = to_grayscale(image);
    Mat gray_blur = gaussian_blur(gray);

    return 0;
}
