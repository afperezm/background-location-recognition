/*
 * feature_extraction.cpp
 *
 *  Created on: May 15, 2013
 *      Author: andresf
 */

#include "extractor.h"

#include <opencv2/features2d/features2d.hpp>

#include <stdio.h>

using cv::DescriptorExtractor;
using cv::FeatureDetector;
using cv::Ptr;
using cv::imread;

Features detectAndDescribeFeatures(const std::string& imageName) {

	Features features;

	Mat img = imread(imageName.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	if (!img.data) {
		printf("Error reading image [%s]\n", imageName.c_str());
	} else {
		// Create smart pointer for SIFT feature detector
		Ptr < FeatureDetector > featureDetector = FeatureDetector::create(
				"SIFT");

		// Detect the keypoints
		printf("Detecting keypoints from image [%s]\n", imageName.c_str());
		featureDetector->detect(img, features.keypoints);

		// Create smart pointer for SIFT descriptor extractor
		Ptr < DescriptorExtractor > featureExtractor =
				DescriptorExtractor::create("SIFT");

		// Describe keypoints
		printf("Describing keypoints from image [%s]\n", imageName.c_str());
		featureExtractor->compute(img, features.keypoints,
				features.descriptors);
	}

	return features;
}
