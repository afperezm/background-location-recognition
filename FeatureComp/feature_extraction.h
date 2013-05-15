/*
 * feature_extraction.h
 *
 *  Created on: May 15, 2013
 *      Author: andresf
 */

#ifndef FEATURE_EXTRACTION_H_
#define FEATURE_EXTRACTION_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <vector>

using cv::Mat;
using cv::KeyPoint;

using std::string;
using std::vector;

struct Features {
	vector<KeyPoint> keypoints;
	Mat descriptors;
};

Features detectAndDescribeFeatures(const string& imageName);

#endif /* FEATURE_EXTRACTION_H_ */
