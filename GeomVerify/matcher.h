/*
 * matcher.h
 *
 *  Created on: May 15, 2013
 *      Author: andresf
 */

#ifndef MATCHER_H_
#define MATCHER_H_

#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>

using std::string;
using std::vector;
using cv::KeyPoint;
using cv::Mat;
using cv::DMatch;
using cv::Point2f;

Mat computeCorrelationMatrix(const Mat& templateImg, const Mat& sourceImg,
		vector<KeyPoint>& templateKeypoints, vector<KeyPoint>& sourceKeypoints,
		int& windowHalfLength, double& thresholdNCC, double& distanceThreshold);

void matchKeypoints(const Mat& templateImg, vector<KeyPoint>& templateKeypoints,
		const Mat& sourceImg, vector<KeyPoint>& sourceKeypoints,
		vector<DMatch>& good_matches, vector<Point2f>& matchedTemplatePoints,
		vector<Point2f>& matchedSourcePoints, double& proximityThreshold,
		double& similarityThreshold);

int geometricVerification(string& templateImgFilepath,
		string& templateKeypointsFilepath, string& sourceImgFilepath,
		string& sourceKeypointsFilepath, double ransacReprojThreshold = 10.0,
		double proximityThreshold = 100.0, double similarityThreshold = 0.8);

#endif /* MATCHER_H_ */
