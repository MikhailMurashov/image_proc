#include <iostream>
#include <omp.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace cv;
using namespace std;

Mat to_grayscale(const Mat& image) {
    Mat grayscale_img(image.rows, image.cols, CV_8UC1);

    auto start = std::chrono::high_resolution_clock::now();

    int i, j;

    #pragma omp parallel for private(j)
    for (i = 0; i < image.rows; i++)
        #pragma omp parallel for
        for (j = 0; j < image.cols; j++) {
            uchar B = 0.114f * image.at<Vec3b>(i,j)[0];
            uchar G = 0.587f * image.at<Vec3b>(i,j)[1];
            uchar R = 0.299f * image.at<Vec3b>(i,j)[2];
            grayscale_img.at<uchar>(i,j) = B + G + R;
        }

    auto stop = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cout << "Grayscale time: " << (float)time.count() / 1000 << " sec" << endl;
    return grayscale_img;
}

Mat gaussian_blur(const Mat& image, const int kernel_size = 5) {
    Mat kernel = getGaussianKernel(kernel_size, 1) * getGaussianKernel(kernel_size, 1).t();
    Mat blur(image.rows, image.cols, CV_8UC1);

    int i, j, k, l;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for private(j, k, l)
    for(i = 0; i < image.rows; i++)
        #pragma omp parallel for private(k, l)
        for(j = 0; j < image.cols; j++) {
            int half = static_cast<int>(kernel_size / 2);
            double b = 0.0f;

            for(k = -half; k < half; k++)
                for(l = -half; l < half; l++) {
                    int x = max(0, min(image.rows-1, i+k));
                    int y = max(0, min(image.cols-1, j+l));
                    double pixel = image.at<uchar>(x, y);
                    b += pixel * kernel.at<double>(k + half, l + half);
                }

            blur.at<uchar>(i,j) = static_cast<uchar>(b);
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
