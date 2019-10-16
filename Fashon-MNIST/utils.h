#pragma once
#include <cinttypes>
#include "opencv2/opencv.hpp"

cv::Mat to_one_hot(uint8_t val, uint8_t max)
{
    cv::Mat temp = cv::Mat::zeros(max + 1, 1, CV_32FC1);
    temp.at<float>(val, 0) = 1.0;
    return temp;
}