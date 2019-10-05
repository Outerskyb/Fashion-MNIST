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
		vec.push_back(mat);
	}vec.pop_back();

	Flatten fl;
	cv::Mat test = fl.run(vec[0]);

	std::vector<Layer> model;
	model.push_back(Flatten());
	model.push_back(FC(FC::ActivationFunction::Relu, 28 * 28, 128));
	model.push_back(FC(FC::ActivationFunction::SoftMax, 128, 10));

	cv::Mat result = vec[0];
	for (auto& layer : model)
	{
		result = layer.run(result);
	}
    for (int i = 0; i < result.rows; i++) {
        cout << (result.at<float>(i, 0)) << ' ';
    }
}