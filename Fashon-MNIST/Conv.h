#pragma once
#include "Layer.h"
class Conv :
    public Layer
{
public:
    Conv();
    ~Conv();

    Conv(int filter_height, int filter_width, int filter_channel) {
        //제대로 돌아가는지 확인 필요
        filter = cv::Mat(cv::Size(filter_width, filter_height), CV_32FC(filter_channel));
    }
    
private:
    cv::Mat filter;
    
public:
    cv::Mat run(cv::Mat);
    cv::Mat train(cv::Mat input, cv::Mat target);

};

