#include "Conv.h"

cv::Mat Conv::run(cv::Mat input)
{
    int sz[] = { filter.channels() * input.channels(), input.rows - filter.rows + 1,input.cols - filter.cols + 1 };
    cv::Mat result = cv::Mat::zeros(filter.channels()*input.channels(), sz, CV_32FC1);
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
    if (target.rows == 0) {
        ip = input;
        return  this->run(input);
    }
   /* else {
        //calculate previous layer's delta
        //activation`(prev layer's output == curr's input) * sigma(curr's node's weight * curr's node's delta)
        cv::Mat delta = cv::Mat::zeros(ip.rows, ip.cols, CV_32FC1);
        for (int k = 0; k < number_of_input; k++) {
            for (int i = 0; i < number_of_node; i++) {
                delta.at<float>(k, 0) += weights.at<float>(i, k) * target.at<float>(i, 0);
            }
            delta.at<float>(k, 0) *= (ip.at<float>(k, 0) == 0) ? 0 : 1; //relu derivative // *(1 - ip.at<float>(k, 0));
        }

        for (int k = 0; k < number_of_node; k++) {
            for (int i = 0; i < ip.rows; i++) {
                for (int j = 0; j < ip.cols; j++) {
                    weights.at<float>(k, i * ip.cols + j)
                        -= etha * ip.at<float>(i, j) * target.at<float>(k, 0);
                }
            }
            weights.at<float>(k, ip.rows * ip.cols) -= etha * target.at<float>(k, 0);
        }
        ip = cv::Mat();
        return delta;
    }*/
}

