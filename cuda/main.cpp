#include <iostream>
#include <opencv2/opencv.hpp>

#include "grayscale.h"
#include "blur.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Mat image;
    if (argc == 2)
        image = imread(argv[1]);
    if (image.empty()) {
        cerr << "Could not open file" << endl
             << "Please provide valid path to image in second argument" << endl;
        return -1;
    }

    Mat gray = to_grayscale(image);
    Mat gray_blur = gaussian_blur(gray);

    return 0;
}