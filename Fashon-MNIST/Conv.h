#pragma once
#include "Layer.h"
#include <random>
class Conv :
    public Layer
{
public:
    Conv();
    ~Conv();

    Conv(int filter_height, int filter_width, int filter_channel) {
        //제대로 돌아가는지 확인 필요
        int sz[] = { filter_channel,filter_height,filter_width };
        filter = cv::Mat(filter_channel,sz, CV_32FC(filter_channel));
        std::random_device rd;
        std::mt19937 mt(rd());
        std::normal_distribution<> nd(0, sqrt(6.0 /filter_height/filter_width));
        srand(time(0));

        for (int k = 0; k < filter_channel; k++) {
            for (int i = 0; i < filter_height; i++) {
                for (int j = 0; j < filter_width + 1; j++) {
                    filter.at<float>(i, j) = ((rand() % 2) ? 1 : -1) * nd(mt);
                }
            }
        }
    }
    
private:
    cv::Mat filter;
    cv::Mat ip;

public:
    cv::Mat run(cv::Mat);
    cv::Mat train(cv::Mat input, cv::Mat target);

};

