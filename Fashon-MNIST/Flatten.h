#pragma once
#include "opencv2/opencv.hpp"
#include "Layer.h"
#include "defines.h"

class Flatten :
	public Layer
{
	
public:

	virtual cv::Mat run(cv::Mat mat ) {
		mat = mat.reshape(1,1);
		return mat;
	}
};

