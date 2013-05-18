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

/**
 * Computes the correlation matrix between two images for a given set of keypoint locations.
 *
 * @param templateImg
 * @param sourceImg
 * @param templateKeypoints
 * @param sourcesKeypoints
 * @param windowHalfLength
 * @param windowSize
 * @param thresholdNCC
 * @param distanceThreshold
 *
 * @return Matrix of normalized correlation coefficients for template and source keypoints
 */
Mat computeCorrelationMatrix(const Mat& templateImg, const Mat& sourceImg,
		vector<KeyPoint>& templateKeypoints, vector<KeyPoint>& sourceKeypoints,
		int& windowHalfLength, double& thresholdNCC, double& distanceThreshold);

/**
 * Finds a set of putative matches based on proximity and similarity applied
 * to a pair of images for which a set of keypoints is available.
 *
 * @param templateImg Reference o the template image
 * @param templateKeypoints Reference to the vector of keypoints from the template image
 * @param sourceImg Reference to the source image
 * @param sourceKeypoints Reference to the vector of keypoints from the source image
 * @param proximityThreshold
 * @param similarityThreshold
 *
 * @param good_matches Reference to the vector where the putative matches will be returned
 * @param matchedTemplatePoints Reference to the vector where the best matches for the template keypoints will be returned
 * @param matchedSourcePoints Reference to the vector where the best matches for the source keypoints will be returned
 */
void matchKeypoints(const Mat& templateImg, vector<KeyPoint>& templateKeypoints,
		const Mat& sourceImg, vector<KeyPoint>& sourceKeypoints,
		vector<DMatch>& good_matches, vector<Point2f>& matchedTemplatePoints,
		vector<Point2f>& matchedSourcePoints, double& proximityThreshold,
		double& similarityThreshold);

int geometricVerification(string& templateImgFilepath,
		string& templateKeypointsFilepath, string& sourceImgFilepath,
		string& sourceKeypointsFilepath, double ransacReprojThreshold = 10.0,
		double similarityThreshold = 0.8);

#endif /* MATCHER_H_ */
