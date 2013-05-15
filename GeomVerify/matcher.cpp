/*
 * matcher.cpp
 *
 *  Created on: May 15, 2013
 *      Author: andresf
 */

#include "matcher.h"
#include <opencv2/features2d/features2d.hpp>
#include <map>
#include "../DataLib/reader.h"
#include "../Common/StringUtils.h"

using cv::Scalar;
using cv::Range;
using cv::norm;
using cv::Point;
using cv::imread;
using std::map;
using cv::DrawMatchesFlags;

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
		int& windowHalfLength, double& thresholdNCC,
		double& distanceThreshold) {

	int windowSize = 2 * windowHalfLength + 1;

	Mat corrMat = -Mat::ones(templateKeypoints.size(), sourceKeypoints.size(),
			CV_32F);

	Mat A, B;
	Scalar meanB, stdDevB;
	Scalar meanA, stdDevA;

	// Loop over keypoints vector of template image
	for (int i = 0; i < (int) templateKeypoints.size(); ++i) {
		KeyPoint pA = templateKeypoints[i];
		// Ignore features close to the border since they don't have enough support
		if (pA.pt.x - windowHalfLength < 0 || pA.pt.y - windowHalfLength < 0
				|| pA.pt.x + windowHalfLength > templateImg.cols
				|| pA.pt.y + windowHalfLength > templateImg.rows) {
			continue;
		}

		// Extract patch from the template image
		templateImg(
				Range(pA.pt.y - windowHalfLength,
						pA.pt.y + windowHalfLength + 1),
				Range(pA.pt.x - windowHalfLength,
						pA.pt.x + windowHalfLength + 1)).convertTo(A, CV_32F);
		meanStdDev(A, meanA, stdDevA);

		// Loop over keypoints vector of source image
		for (int j = 0; j < (int) sourceKeypoints.size(); ++j) {
			KeyPoint pB = sourceKeypoints[j];
			// Ignore features close to the border since they don't have enough support
			if (pB.pt.x - windowHalfLength < 0 || pB.pt.y - windowHalfLength < 0
					|| pB.pt.x + windowHalfLength + 1 > sourceImg.cols
					|| pB.pt.y + windowHalfLength + 1 > sourceImg.rows) {
				continue;
			}

			// Ignore features which are far one another
			if (norm(Point(pA.pt.x, pA.pt.y) - Point(pB.pt.x, pB.pt.y))
					> distanceThreshold) {
				continue;
			}

			// Extract path from the source image
			sourceImg(
					Range(pB.pt.y - windowHalfLength,
							pB.pt.y + windowHalfLength + 1),
					Range(pB.pt.x - windowHalfLength,
							pB.pt.x + windowHalfLength + 1)).convertTo(B,
					CV_32F);
			meanStdDev(B, meanB, stdDevB);

			// Computing normalized cross correlation for the patches A and B
			Mat NCC;
			subtract(A, meanA, A);
			subtract(B, meanB, B);
			multiply(A, B, NCC);
			divide(NCC, stdDevA, NCC);
			divide(NCC, stdDevB, NCC);

			double ncc = sum(NCC)[0] / std::pow((double) windowSize, 2);

			// Applying threshold to the just computed normalized cross correlation
			if (ncc >= thresholdNCC) {
				corrMat.at<float>(i, j) = ncc;
			}
		}
	}

	return corrMat;
}

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
		double& similarityThreshold) {

	printf("Matching source to template keypoints\n");
	clock_t start = clock();

	// Computing correlation matrix for the keypoint locations
	printf("  Computing correlation matrix for the keypoint locations\n");
	int windowHalfLength = 10;

	Mat corrMat = computeCorrelationMatrix(templateImg, sourceImg,
			templateKeypoints, sourceKeypoints, windowHalfLength,
			similarityThreshold, proximityThreshold);

	// Looking for maximum by rows
	printf("  Looking for maximum by rows\n");

	map<int, int> templateKeypointsMatches;

	for (int i = 0; i < corrMat.rows; ++i) {
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(corrMat.row(i), &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		templateKeypointsMatches.insert(map<int, int>::value_type(i, maxLoc.x));
	}

	// Looking for maximum by rows
	printf("  Looking for maximum by columns\n");

	map<int, int> sourceKeypointsMatches;

	for (int i = 0; i < corrMat.cols; ++i) {
		Point maxLoc;
		minMaxLoc(corrMat.col(i), NULL, NULL, NULL, &maxLoc, Mat());
		sourceKeypointsMatches.insert(map<int, int>::value_type(i, maxLoc.y));
	}

	printf("  Looking for coincident matches\n");

	for (int i = 0; i < corrMat.cols; ++i) {
		// sourceKeypointsMatches.at(i) is the best template keypoint match for the ith source keypoint
		if (sourceKeypointsMatches.at(i) != -1
				&& templateKeypointsMatches.at(sourceKeypointsMatches.at(i))
						== i) {
			Point2f sourcePoint = sourceKeypoints[i].pt;
			Point2f templatePoint = templateKeypoints[sourceKeypointsMatches.at(
					i)].pt;
			good_matches.push_back(
					DMatch(sourceKeypointsMatches.at(i), i,
							norm(sourcePoint - templatePoint)));
			matchedSourcePoints.push_back(sourcePoint);
			matchedTemplatePoints.push_back(templatePoint);
		}
	}

	clock_t end = clock();

	printf("  Found [%d] putative matches in [%0.3fs]\n",
			(int) good_matches.size(), (double) (end - start) / CLOCKS_PER_SEC);
}

int geometricVerification(string& templateImgFilepath,
		string& templateKeypointsFilepath, string& sourceImgFilepath,
		string& sourceKeypointsFilepath, double ransacReprojThreshold,
		double proximityThreshold, double similarityThreshold) {

	// 1) Load template image and template keypoints file
	Mat templateImg = imread(templateImgFilepath.c_str(),
			CV_LOAD_IMAGE_GRAYSCALE);

	vector<KeyPoint> templateKeypoints;
	Mat templateDescriptors;
	readKeypoints(templateKeypointsFilepath.c_str(), templateKeypoints,
			templateDescriptors);

	// 2) Load source image and source keypoints file
	Mat sourceImg = imread(sourceImgFilepath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	vector<KeyPoint> sourceKeypoints;
	Mat sourceDescriptors;
	readKeypoints(sourceKeypointsFilepath.c_str(), sourceKeypoints,
			sourceDescriptors);

	// 3) Find putative matches
	vector<DMatch> good_matches;
	vector<Point2f> matchedSourcePoints;
	vector<Point2f> matchedTemplatePoints;

	matchKeypoints(templateImg, templateKeypoints, sourceImg, sourceKeypoints,
			good_matches, matchedTemplatePoints, matchedSourcePoints,
			proximityThreshold, similarityThreshold);

	if (((int) good_matches.size()) < 4) {
		fprintf(stderr,
				"  Error while matching keypoints, at least 4 putative matches are needed for homography computation\n");
		return -1;
	}

	// 4) Draw resulting putative matches
//	Mat img_matches;
//	drawMatches(templateImg, templateKeypoints, sourceImg, sourceKeypoints,
//			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//	namedWindow("Good Matches & Object detection", CV_WINDOW_NORMAL);
//	imshow("Good Matches & Object detection", img_matches);
//
//	while (1) {
//		if (waitKey(1000) == 27) {
//			break;
//		}
//	}

// 5) Compute a projective transformation
	Mat inliers_idx;
	clock_t start = clock();
	Mat H = findHomography(matchedSourcePoints, matchedTemplatePoints,
			CV_RANSAC, ransacReprojThreshold, inliers_idx);
	clock_t end = clock();
	printf("  Computed homography in [%0.3fs] and found [%d] inliers\n",
			(double) (end - start) / CLOCKS_PER_SEC, (int) sum(inliers_idx)[0]);

// 6) Drawing resulting inliers
	vector<DMatch> inliers;
	for (int i = 0; i < inliers_idx.rows; ++i) {
		if ((int) inliers_idx.at<uchar>(i) == 1) {
			inliers.push_back(good_matches.at(i));
		}
	}
	Mat inlier_matches;
	drawMatches(templateImg, templateKeypoints, sourceImg, sourceKeypoints,
			inliers, inlier_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	string outputInliersFilepath = templateImgFilepath.substr(0,
			templateImgFilepath.size() - 4) + "_matches_"
			+ StringUtils::split(sourceImgFilepath, '/').back();
	printf("  Writing inlier matches to [%s]\n", outputInliersFilepath.c_str());

	imwrite(
			(templateImgFilepath.substr(0, templateImgFilepath.size() - 4)
					+ "_matches_"
					+ StringUtils::split(sourceImgFilepath, '/').back()).c_str(),
			inlier_matches);

//	namedWindow("Inlier matches", CV_WINDOW_NORMAL);
//	imshow("Inlier matches", inlier_matches);
//
//	while (1) {
//		if (waitKey(1000) == 27) {
//			break;
//		}
//	}

	return (int) sum(inliers_idx)[0];
}

