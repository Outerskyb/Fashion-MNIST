#include "FC.h"
#include <cmath>
#include "opencv2/opencv.hpp"

float FC::relu(float x)
{
    if (x >= 0) return x;
    return 0.0f;
}

float FC::softmax(float x , cv::Mat result,bool re)
{
    static float sum = 0;
    if (re) {
        for (int i = 0; i < number_of_node; i++) {
            sum += exp(result.at<float>(i, 0));
        }
        return 0;
    }
    return x/sum;
}

 cv::Mat FC::run(cv::Mat input)
{
    cv::Mat result = cv::Mat::zeros(number_of_node, 1, CV_32FC1);
    for (int k = 0; k < number_of_node; k++) {
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                result.at<float>(k, 0) += weights.at<float>(k, i * input.cols + j) * input.at<float>(i, j);
            }
        }
        result.at<float>(k, 0) /= number_of_input;
        result.at<float>(k, 0) += bias.at<float>(k, 0);
    }

    if (activation_function == ActivationFunction::Relu) {
        for (int k = 0; k < number_of_node; k++) {
            result.at<float>(k, 0) = relu(result.at<float>(k, 0));
        }
    }
    else {
        softmax(0, result, true);
        for (int k = 0; k < number_of_node; k++) {
            result.at<float>(k, 0) = softmax(result.at<float>(k, 0),result,false);
        }
    }
    
    return result;
}
