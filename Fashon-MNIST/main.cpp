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
    FILE* ti = fopen("test", "rb");
    FILE* tl = fopen("tlbl", "rb");

    std::vector<cv::Mat> vec;
    std::vector<cv::Mat> lbl;
    std::vector<cv::Mat> test_image;
    std::vector<uint8_t> test_label;

    char dummy[16];
    fread(dummy, 1, 16, fp);
    fread(dummy, 1, 8, lb);
    fread(dummy, 1, 16, ti);
    fread(dummy, 1, 8, tl);

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
    while (!feof(ti)) {
        cv::Mat mat(28, 28, CV_8UC1);
        fread(mat.data, 28 * 28 * 1, 1, ti);
        mat.convertTo(mat, CV_32FC1, 1.0 / 255);
        test_image.push_back(mat);
    }vec.pop_back();

    while (!feof(tl)) {
        uint8_t temp;
        fread(&temp, 1, 1, tl);
        test_label.push_back(temp);
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

    for (int j = 0; j < 5; j++) {
        cout << "epoch : " << j << '\n';
        for (int i = 0; i < vec.size(); i++) {
          //  if (i % 200 == 0) cout << i << '/' << vec.size() << '\n';
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
        int cnt = 0;
        for (int i = 0; i < test_image.size(); i++) {
            test_image[i].copyTo(result);
            for (auto& layer : model)
            {
                result = layer->run(result);
            }
            if (test_label[i] == get_max_idx(result)) cnt++;
       //    cout << (int)test_label[i] << " , " << get_max_idx(result)<<'\n';
        }

        cout << "accuracy : " << cnt / 10000.0 << '\n';
    }

    
    
    //////////////////////////////////

}
