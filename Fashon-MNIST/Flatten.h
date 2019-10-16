#pragma once
#include "opencv2/opencv.hpp"
#include "Layer.h"
#include "defines.h"

class Flatten :
	public Layer
{
	
public:

	 cv::Mat run(cv::Mat mat ) 
     {
		mat = mat.reshape(1,mat.rows*mat.cols);
		return mat;
	 }

     cv::Mat train(cv::Mat input, cv::Mat target) 
     {
         return run(input);
     }
};

