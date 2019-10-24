#include "Conv.h"

cv::Mat Conv::run(cv::Mat input)
{
    int sz[] = { input.rows - filter.rows + 1,input.cols - filter.cols + 1 };
    cv::Mat result = cv::Mat::zeros(filter.channels()*input.channels(), sz, CV_32FC(filter.channels()));
    for (int k = 0; k < input.channels(); k++){
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                for (int c = 0; c < filter.channels(); c++) {
                    float sum = 0;
                    for (int a = 0; a < filter.rows; a++) {
                        for (int b = 0; b < filter.cols; b++) {
                            int iv[] = { k,i,j }, fv[] = { c,a,b };
                            sum += input.at<float>(iv) * filter.at<float>(fv);
                        }
                    }
                    int rv[] = { k*filter.channels()+c,i,j };
                    result.at<float>(rv) = sum;
                }
            }
        }
    }
    return result;
}

cv::Mat Conv::train(cv::Mat input, cv::Mat target)
{
    return cv::Mat();
}

