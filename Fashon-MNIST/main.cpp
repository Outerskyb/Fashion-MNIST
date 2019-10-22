#include <iostream>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include "utils.h"
#include "Layer.h"
#include "FC.h"
#include "Flatten.h"

int main()
{
    FILE* fp = fopen("train", "rb");
    FILE* lb = fopen("label", "rb");

    std::vector<cv::Mat> vec;
    std::vector<cv::Mat> lbl;

    char dummy[16];
    fread(dummy, 1, 16, fp);
    fread(dummy, 1, 8, lb);

    while (!feof(fp)) {
        cv::Mat mat(28, 28, CV_8UC1);
        fread(mat.data, 28 * 28 * 1, 1, fp);
        mat.convertTo(mat, CV_32FC1, 1.0 / 255);
        vec.push_back(mat);
    }vec.pop_back();

    while (!feof(lb)) {
        uint8_t temp;
        fread(&temp, 1, 1, lb);
        lbl.push_back(to_one_hot(temp, 9));
    }

    std::vector<Layer*> model;
    Flatten fl;
    FC relu1 = FC(FC::ActivationFunction::Relu, 28 * 28, 128);
    FC softmax1 = FC(FC::ActivationFunction::SoftMax, 128, 10);
    model.push_back(&fl);
    model.push_back(&relu1);
    model.push_back(&softmax1);

    //////////////////////////////////
    cv::Mat result;
    vec[0].copyTo(result);
    for (auto& layer : model)
    {
        result = layer->run(result);
    }
    for (int i = 0; i < result.rows; i++) {
        cout << (result.at<float>(i, 0)) << '\n';
    }

    //////////////////////////////////
    int cnt = 0;
    for (int i = 0; i < vec.size(); i++, cnt++) {
        if (cnt % 100 == 0) {
            cv::Mat we = ((FC*)model[1])->debug_get_weights();
            float max = 0;
            for (int a = 0; a < we.rows; a++) for (int b = 0; b < we.cols; b++) if (max < we.at<float>(a, b)) max = we.at<float>(a, b);
            const int* ch = { 0 };
            float channel_range[] = { 0.0,5 };
            const int bins = 512;
            const float* channel_ranges = channel_range;
            cv::Mat hist;
            cv::calcHist(&we,1,ch,cv::Mat(),hist,1,&bins,&channel_ranges);
            we = histPlot(hist);
            cv::putText(we, std::to_string(max), cv::Point(350,512- 450), 1, 1.0, cv::Scalar(255, 255, 255));
            cv::imwrite(string("weights/weight") + to_string(i) + string(".png"), we);
        }
        vec[i].copyTo(result);
        for (auto& layer : model) {
            result = layer->train(result, cv::Mat());
        }
        result = get_last_delta(result, lbl[i]);
        for (auto it = model.rbegin(); it != model.rend(); it++) {
            result = (*it)->train(cv::Mat(), result);
        }
    }

    //////////////////////////////////
    cout << '\n';
    vec[0].copyTo(result);
    for (auto& layer : model)
    {
        result = layer->run(result);
    }
    for (int i = 0; i < result.rows; i++) {
        cout << (result.at<float>(i, 0)) << '\n';
    }

    //////////////////////////////////
    vec[0].copyTo(result);
    for (auto& layer : model) {
        result = layer->train(result, cv::Mat());
    }
    result = get_last_delta(result, lbl[0]);
    for (auto it = model.rbegin(); it != model.rend(); it++) {
        result = (*it)->train(cv::Mat(), result);
    }

    //////////////////////////////////
    cout << '\n';
    vec[1].copyTo(result);
    for (auto& layer : model)
    {
        result = layer->run(result);
    }
    for (int i = 0; i < result.rows; i++) {
        cout << (result.at<float>(i, 0)) << '\n';
    }
}
