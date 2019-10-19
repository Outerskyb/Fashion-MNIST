#pragma once
#include <cinttypes>
#include "globals.h"
#include "opencv2/opencv.hpp"

cv::Mat to_one_hot(uint8_t val, uint8_t max)
{
    cv::Mat temp = cv::Mat::zeros(max + 1, 1, CV_32FC1);
    temp.at<float>(val, 0) = 1.0;
    return temp;
}

cv::Mat get_last_delta(cv::Mat result, cv::Mat lable) 
{
    cv::Mat delta(result.rows,result.cols,CV_32FC1);
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            
            /*if (lable.at<float>(i, j) == 1) {
                delta.at<float>(i, j) = (result.at<float>(i, j) - 1)*etha;
            }
            else {
                delta.at<float>(i, j) = result.at<float>(i, j)*etha;
            }*/
            
            delta.at<float>(i, j) 
                = result.at<float>(i, j) 
                * (1 - result.at<float>(i, j)) 
                * (lable.at<float>(i, j) - result.at<float>(i, j));
        }
    }
    return delta;
}