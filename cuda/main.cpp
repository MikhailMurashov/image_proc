#include <iostream>
#include <opencv2/opencv.hpp>

#include "hello_world.h"
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

//    hello_world();

    Mat gray = to_grayscale(image);
    Mat gray_blur = gaussian_blur(gray);

    imwrite("grayscale.jpg", gray);
    imwrite("blur.jpg", gray_blur);

    return 0;
}