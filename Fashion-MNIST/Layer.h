#pragma once
#include "opencv2/opencv.hpp"

class Layer
{
public:
    virtual cv::Mat run(cv::Mat img) { return cv::Mat(); };
    virtual cv::Mat train(cv::Mat input,cv::Mat target) { return cv::Mat(); };
};

