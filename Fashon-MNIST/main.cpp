#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "Layer.h"
#include "FC.h"
#include "Flatten.h"

int main()
{
	FILE* fp = fopen("train", "rb");

	std::vector<cv::Mat> vec;

	char dummy[16];
	fread(dummy, 1, 16, fp);

	while (!feof(fp)) {
		cv::Mat mat(28, 28, CV_8UC1);
		fread(mat.data, 28 * 28 * 1, 1, fp);
        mat.convertTo(mat, CV_32FC1, 1.0 / 255);
		vec.push_back(mat);
	}vec.pop_back();

	std::vector<Layer*> model;
    Flatten fl;
    FC relu1 = FC(FC::ActivationFunction::Relu, 28 * 28, 128);
    FC softmax1 = FC(FC::ActivationFunction::SoftMax, 128, 10);
	model.push_back(&fl);
	model.push_back(&relu1);
	model.push_back(&softmax1);

    cv::Mat result;
    vec[0].copyTo(result);
	for (auto& layer : model)
	{
		result = layer->run(result);
	}
    for (int i = 0; i < result.rows; i++) {
        cout << (result.at<float>(i, 0)) << '\n';
    }
}
