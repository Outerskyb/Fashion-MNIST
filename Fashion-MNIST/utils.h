#pragma once
#include <cinttypes>
#include "globals.h"
#include "opencv2/opencv.hpp"
#include <vector>

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
            
            if (lable.at<float>(i, j) == 1) {
                delta.at<float>(i, j) = (result.at<float>(i, j) - 1);//*etha;
            }
            else {
                delta.at<float>(i, j) = result.at<float>(i, j);// *etha;
            }
            
            /*delta.at<float>(i, j) 
                = result.at<float>(i, j) 
                * (1 - result.at<float>(i, j)) 
                * (lable.at<float>(i, j) - result.at<float>(i, j));*/
        }
    }
    return delta;
}

cv::Mat histPlot(cv::Mat hist) 
{
    cv::Mat result = cv::Mat::zeros(512, 512, CV_8UC1);
    float max = 0;
    for (int i = 0; i < hist.rows; i++) if(max<hist.at<float>(i,0)) max = hist.at<float>(i,0);
    float adj = 512 / max;
    for (int i = 1; i < hist.rows; i++) {
        cv::line(result, cv::Point(i - 1, 512-hist.at<float>(i - 1, 0)*adj), cv::Point(i, 512-hist.at<float>(i, 0)*adj), cv::Scalar(255, 255, 255), 1);
    }
    return result;
}

int get_max_idx(std::vector<float> vec) 
{
    float max = 0; int max_idx = 0;
    for (int i = 0; i < vec.size(); i++) if (max < vec[i]) max = vec[i], max_idx = i;
    return max_idx;
}