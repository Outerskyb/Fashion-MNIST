#pragma once
#include "Layer.h"
#include <stdlib.h>
#include <time.h>
#include "opencv2/opencv.hpp"


class FC :
    public Layer
{
public:
    enum class ActivationFunction
    {
        Relu = 1,
        SoftMax = 2
    };

public:
    FC(ActivationFunction af, int number_of_input, int number_of_node) : activation_function(af), number_of_input(number_of_input), number_of_node(number_of_node)
    {
        srand(time(0));
        weights = cv::Mat(number_of_node, number_of_input, CV_32FC1);
        bias = cv::Mat(number_of_node, number_of_input, CV_32FC1);

        for (int i = 0; i < number_of_node; i++) {
            for (int j = 0; j < number_of_input; j++) {
                weights.at<float>(i, j) = ((rand() % 2) ? 1 : -1) * rand() / 1000.0 + 1;
                bias.at<float>(i, j) = ((rand() % 2) ? 1 : -1) * rand() / 1000.0 + 1;
            }
        }
    }

private:
    ActivationFunction activation_function;
    int number_of_input;
    int number_of_node;
    cv::Mat weights;
    cv::Mat bias;

private:
    float relu(float);
    float softmax(float x, cv::Mat result, bool re);

public:
    cv::Mat run(cv::Mat);

};

