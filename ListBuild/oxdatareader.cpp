/*
 * oxdatareader.cpp
 *
 *  Created on: Apr 13, 2013
 *      Author: andresf
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <string>
#include <vector>
#include <map>

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <utility>

#include <sys/stat.h>

#include "../Common/StringUtils.h"
#include "../Common/FileUtils.h"

#include <VocabLib/keys2.h>

using std::endl;
using std::string;
using std::vector;
using std::map;
using std::ofstream;
using std::ios;
using std::find;

using std::cout;

using std::ostringstream;

using namespace cv;

struct Features {
	vector<KeyPoint> keypoints;
	Mat descriptors;
};

/**
 *
 * @param filename Path to the a features file.
 * @param keypoints Keypoints stored in the specified input file.
 * @param feature_descriptors Descriptors stored in the specified input file. Row i is the descriptor for keypoint i.
 */
void readKeypoints(const char *filename, vector<KeyPoint>& keypoints,
		Mat& descriptors) {

	printf("Reading keypoints from file [%s]\n", filename);

	int num_keys = 0;
	short int *keys;
	keypt_t* info = NULL;
	num_keys = ReadKeyFile(filename, &keys, &info);

	int dim = 128;

	descriptors = Mat(num_keys, dim, DataType<float>::type);

	for (int i = 0; i < num_keys; i++) {
		KeyPoint key_point = KeyPoint();
		key_point.pt.x = info[i].x;
		key_point.pt.y = info[i].y;
		key_point.size = info[i].scale;
		key_point.angle = info[i].orient;

		keypoints.push_back(key_point);

		for (int j = i * dim; j < (i + 1) * dim; j++) {
			descriptors.at<float>(i, j - i * dim) = (float) keys[j];
		}

	}

	delete[] keys;
	if (info != NULL) {
		delete[] info;
	}

	printf("  Read [%d] keypoints\n", (int) keypoints.size());

}

void createListDbTxt(const char* folderName,
		const vector<string>& geometryFiles, string& keypointsFilename,
		const vector<string>& queryKeypointFiles,
		bool appendLandmarkId = false) {

	vector<string> dbKeypointFiles;

	ofstream keypointsFile;
	keypointsFile.open(keypointsFilename.c_str(), ios::out | ios::trunc);

	vector<string> landmarks;

	for (vector<string>::const_iterator fileName = geometryFiles.begin();
			fileName != geometryFiles.end(); ++fileName) {
		if ((*fileName).find("1") != string::npos
				&& (*fileName).find("query") == string::npos) {
			// Open file
			fprintf(stdout, "Reading file [%s]\n", (*fileName).c_str());

			string landmarkName = StringUtils::parseLandmarkName(fileName);

			if (std::find(landmarks.begin(), landmarks.end(), landmarkName)
					== landmarks.end()) {
				// If landmarkName wasn't found then add it
				landmarks.push_back(landmarkName);
			}

			std::ifstream infile(
					(string(folderName) + "/" + *fileName).c_str());

			// Extract data from file
			string line;

			while (std::getline(infile, line)) {
				if (std::find(queryKeypointFiles.begin(),
						queryKeypointFiles.end(), line)
						== queryKeypointFiles.end()) {

					string imageName = "db/" + string(line.c_str())
							+ string(KEYPOINT_FILE_EXTENSION);

					if (appendLandmarkId == true) {
						// Position of the landmarkName in the vector of landmarks
						ostringstream temp;
						temp << ((int) landmarks.size()) - 1;
						imageName += " " + temp.str();
					}

					if (std::find(dbKeypointFiles.begin(),
							dbKeypointFiles.end(), line)
							== dbKeypointFiles.end()
							|| appendLandmarkId == true) {
						dbKeypointFiles.push_back(line);
						keypointsFile << imageName << endl;
					}
				}
			}

			//Close file
			infile.close();
		}
	}

	keypointsFile.close();
}

vector<string> createListQueriesTxt(const char* folderName,
		const vector<string>& geometryFiles, string& keypointsFilename,
		bool appendLandmarkId = false) {

	vector<string> queryKeypointFiles;

	ofstream keypointsFile;
	keypointsFile.open(keypointsFilename.c_str(), ios::out | ios::trunc);

	vector<string> landmarks;

	for (vector<string>::const_iterator fileName = geometryFiles.begin();
			fileName != geometryFiles.end(); ++fileName) {
		if ((*fileName).find("query") != string::npos) {
			// Open file
			fprintf(stdout, "Reading file [%s]\n", (*fileName).c_str());

			string landmarkName = StringUtils::parseLandmarkName(fileName);
			if (std::find(landmarks.begin(), landmarks.end(), landmarkName)
					== landmarks.end()) {
				// If landmarkName wasn't found then add it
				landmarks.push_back(landmarkName);
			}

			std::ifstream infile(
					(string(folderName) + "/" + *fileName).c_str());

			// Extract data from file
			string line;

			while (std::getline(infile, line)) {
				vector<string> lineSplitted = StringUtils::split(line, ' ');
				string qName = "queries/" + lineSplitted[0].substr(5)
						+ string(KEYPOINT_FILE_EXTENSION);

				if (appendLandmarkId == true) {
					// Position of the landmarkName in the vector of landmarks
					ostringstream temp;
					temp << ((int) landmarks.size()) - 1;
					qName += " " + temp.str();
				}

				keypointsFile << qName << endl;

				queryKeypointFiles.push_back(lineSplitted[0].substr(5));
			}

			//Close file
			infile.close();
		}
	}

	return queryKeypointFiles;
}

void displayImage(string imageName, vector<KeyPoint> keypoints) {

	printf("Displaying image [%s]", imageName.c_str());

	Mat img = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);

	namedWindow(imageName.c_str(), CV_WINDOW_NORMAL);
	drawKeypoints(img, keypoints, img, cvScalar(255, 0, 0),
			DrawMatchesFlags::DEFAULT);
	imshow(imageName.c_str(), img);
	cvResizeWindow(imageName.c_str(), 480, 640);

	waitKey(0);

}

Features detectAndDescribeFeatures(const std::string& imageName) {

	Features features;

	Mat img = imread(imageName.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

	if (!img.data) {
		printf("Error reading image [%s]\n", imageName.c_str());
	} else {
		// Create smart pointer for SIFT feature detector
		Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");

		// Detect the keypoints
		printf("Detecting keypoints from image [%s]\n", imageName.c_str());
		featureDetector->detect(img, features.keypoints);

		// Create smart pointer for SIFT descriptor extractor
		Ptr<DescriptorExtractor> featureExtractor = DescriptorExtractor::create(
				"SIFT");

		// Describe keypoints
		printf("Describing keypoints from image [%s]\n", imageName.c_str());
		featureExtractor->compute(img, features.keypoints,
				features.descriptors);
	}

	return features;
}

string op_getkey(map<string, vector<KeyPoint> >::value_type pair) {
	return pair.first;
}

template<class K, class V> vector<K> getMapKeys(map<K, V>& images) {
	vector<K> keys;
	keys.resize(images.size());
	std::transform(images.begin(), images.end(), keys.begin(), op_getkey);
	return keys;
}

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
		vector<KeyPoint>& templateKeypoints, vector<KeyPoint>& sourcesKeypoints,
		int& windowHalfLength, double& thresholdNCC,
		double& distanceThreshold) {

	int windowSize = 2 * windowHalfLength + 1;

	Mat corrMat = Mat::zeros(templateKeypoints.size(), sourcesKeypoints.size(),
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
		for (int j = 0; j < (int) sourcesKeypoints.size(); ++j) {
			KeyPoint pB = sourcesKeypoints[j];
			// Ignore features close to the border since they don't have enough support
			if (pB.pt.x - windowHalfLength < 0 || pB.pt.y - windowHalfLength < 0
					|| pB.pt.x + windowHalfLength + 1 > sourceImg.cols
					|| pB.pt.y + windowHalfLength + 1 > sourceImg.rows) {
				continue;
			}

			// Ignore features which are far one another
			if (abs(pA.pt.x - pB.pt.x) > distanceThreshold
					|| abs(pA.pt.y - pB.pt.y) > distanceThreshold) {
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

	map<int, int> sourceKeypointsMatches;

	for (int i = 0; i < corrMat.rows; ++i) {
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(corrMat.row(i), &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		sourceKeypointsMatches.insert(
				map<int, int>::value_type(i, (int) maxLoc.x));
	}

	// Looking for maximum by rows
	printf("  Looking for maximum by columns\n");

	map<int, int> templateKeypointsMatches;

	for (int i = 0; i < corrMat.cols; ++i) {
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(corrMat.col(i), &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		templateKeypointsMatches.insert(map<int, int>::value_type(i, maxLoc.y));
	}

	printf("  Looking for coincident matches\n");

	for (int i = 0; i < corrMat.rows; ++i) {
		// sourceKeypointsMatches.at(i) is the best template keypoint match for the ith source keypoint
		if (templateKeypointsMatches.at(sourceKeypointsMatches.at(i)) == i) {
			Point2f sourcePoint = sourceKeypoints[i].pt;
			Point2f templatePoint = templateKeypoints[sourceKeypointsMatches.at(
					i)].pt;
			good_matches.push_back(
					DMatch(i, sourceKeypointsMatches.at(i),
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
		string& sourceKeypointsFilepath, double ransacReprojThreshold = 5,
		double proximityThreshold = 300, double similarityThreshold = 0.8) {

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

	// 4) Draw resulting putative matches
	Mat img_matches;
	drawMatches(templateImg, templateKeypoints, sourceImg, sourceKeypoints,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	namedWindow("Good Matches & Object detection", CV_WINDOW_NORMAL);
	imshow("Good Matches & Object detection", img_matches);

	while (1) {
		if (waitKey(1000) == 27) {
			break;
		}
	}

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

int main(int argc, char **argv) {

	Mat corrMat = Mat::ones(3, 3, CV_32F);
	corrMat.at<float>(0, 0) = 3;
	corrMat.at<float>(1, 1) = 2;

//	if (argc != 4) {
//		printf(
//				"Usage: %s {output specification} <path_to_geometry_files_folder>"
//						"\nOUTPUT:\n  -db: create list of SIFT feature files of database images.\n  -dbld: create list of SIFT feature files of database images and corresponding landmark ID.\n  -q: create list of SIFT feature files of query images.\n  -gt: create list of SIFT feature files of query images and corresponding landmark ID.\n  -cf: compute image features.\n",
//				argv[0]);
//		return EXIT_FAILURE;
//	}

	if (string(argv[1]).compare("-gvc") == 0) {

		string imagesFolderpath(argv[2]);
		string keysFolderpath(argv[3]);

		string line, templateFilename, sourceFilename;
		std::ifstream candidatesFile(argv[4], std::fstream::in);
		ofstream candidatesGvFile("candidates_gv.txt", std::fstream::out);

		Mat candidates_inliers, candidates_inliers_idx;

		while (std::getline(candidatesFile, line)) {
			candidates_inliers = Mat::zeros(1, 50, DataType<int>::type);
			candidates_inliers_idx = Mat::zeros(1, 50, DataType<int>::type);

			vector<string> splitted_line = StringUtils::split(line.c_str(),
					' ');
			templateFilename = splitted_line[0];
			string templateImgFilepath(
					imagesFolderpath
							+ StringUtils::parseImgFilename(templateFilename));

			for (int i = 1; i < (int) splitted_line.size(); ++i) {
				sourceFilename = splitted_line[i];

				string templateKeypointsFilepath(
						keysFolderpath + templateFilename);

				string sourceImgFilepath(
						imagesFolderpath
								+ StringUtils::parseImgFilename(
										sourceFilename));

				string sourceKeypointsFilepath(keysFolderpath + sourceFilename);

				int num_inliers = geometricVerification(templateImgFilepath,
						templateKeypointsFilepath, sourceImgFilepath,
						sourceKeypointsFilepath);

				candidates_inliers.at<int>(i - 1) = num_inliers;
			}
			sortIdx(candidates_inliers, candidates_inliers_idx,
					CV_SORT_DESCENDING);
			// Print number of inliers for each candidate
			// candidatesGvFile << templateFilename << " " << candidates_inliers << endl;
			// Print indexes of ordered Mat of candidates inliers
			// candidatesGvFile << templateFilename << " " << candidates_inliers_idx << endl;
			candidatesGvFile << templateFilename;
			for (int j = 0; j < candidates_inliers_idx.cols; ++j) {
				// Print index of ordered element at j position
				// candidatesGvFile << " " << candidates_inliers.at<int>( candidates_inliers_idx.at<int>(j));
				// Print ordered element at j+1 position since the first element of candidate's line is the query name
				candidatesGvFile << " "
						<< splitted_line[candidates_inliers_idx.at<int>(j) + 1];
			}
			candidatesGvFile << endl;
		}
		candidatesGvFile.close();
		candidatesFile.close();

		return EXIT_SUCCESS;
	}

	if (string(argv[1]).compare("-gv") == 0) {

		string imagesFolderpath(argv[2]);
		// TODO Check if imagesFolderpath exists
		string keysFolderpath(argv[3]);
		// TODO Check if keysFolderpath exists

		string templateFilename(argv[4]);
		string sourceFilename(argv[5]);

		double ransacReprojThreshold;
		double proximityThreshold, similarityThreshold;
		if (argc >= 7) {
			ransacReprojThreshold = atof(argv[6]);
		} else {
			ransacReprojThreshold = 10.0;
		}
		if (argc >= 8) {
			proximityThreshold = atof(argv[7]);
		} else {
			proximityThreshold = 500.0;
		}
		if (argc >= 9) {
			similarityThreshold = atof(argv[8]);
		} else {
			similarityThreshold = 0.8;
		}

		string templateImgFilepath(
				imagesFolderpath + "/"
						+ StringUtils::parseImgFilename(templateFilename));
		// TODO Check that templateImgFilepath is a valid image filepath starting from the images folder as root

		string templateKeypointsFilepath(
				keysFolderpath + "/" + templateFilename);
		// TODO Check that templateFilename is a valid template keypoints filepath starting from the keypoints folder as root

		string sourceImgFilepath(
				imagesFolderpath + "/"
						+ StringUtils::parseImgFilename(sourceFilename));
		// TODO Check that sourceImgFilepath is a valid image filepath starting from the images folder as root

		string sourceKeypointsFilepath(keysFolderpath + "/" + sourceFilename);
		// TODO Check that templateFilename is a valid template keypoints filepath starting from the keypoints folder as root

		geometricVerification(templateImgFilepath, templateKeypointsFilepath,
				sourceImgFilepath, sourceKeypointsFilepath,
				ransacReprojThreshold, proximityThreshold, similarityThreshold);

		return EXIT_SUCCESS;
	}

	vector<string> folderFiles;
	int result = FileUtils::readFolder(argv[2], folderFiles);
	if (result == EXIT_FAILURE) {
		return result;
	}

	ofstream outputFile;

	// TODO Do not include query files into the list of db files
	// TODO Ensure that each image file db or query has at most one landmark id associated
	if (string(argv[1]).compare("-lists") == 0) {

		string keypointsFilename = string(argv[3]) + "/list_queries.txt";
		vector<string> queryKeypointFiles = createListQueriesTxt(argv[2],
				folderFiles, keypointsFilename);

		keypointsFilename = string(argv[3]) + "/list_db.txt";
		createListDbTxt(argv[2], folderFiles, keypointsFilename,
				queryKeypointFiles);

	} else if (string(argv[1]).compare("-gt") == 0) {

		string keypointsFilename = string(argv[3]) + "/list_gt.txt";
		vector<string> queryKeypointFiles = createListQueriesTxt(argv[2],
				folderFiles, keypointsFilename, true);

		keypointsFilename = string(argv[3]) + "/list_db_ld.txt";
		createListDbTxt(argv[2], folderFiles, keypointsFilename,
				queryKeypointFiles, true);

	} else if (string(argv[1]).compare("-cf") == 0) {
		vector<string>::iterator start_image;
		if (argc == 3) {
			// Set first image as start point of the loop over the folder of images
			start_image = folderFiles.begin();
		} else {
			// Set received argument as start point of the loop over the folder of images
			start_image = std::find(folderFiles.begin(), folderFiles.end(),
					argv[3]);
		}

		for (vector<string>::iterator image = start_image;
				image != folderFiles.end(); ++image) {
			if ((*image).find(".jpg") != string::npos) {
				printf("%s\n", (*image).c_str());
				Features features = detectAndDescribeFeatures(
						argv[2] + string("/") + (*image));

				string descriptorFileName(argv[2]);
				descriptorFileName += "/"
						+ (*image).substr(0, (*image).size() - 4) + ".key";
				printf("Writing feature descriptors to [%s]\n",
						descriptorFileName.c_str());
				outputFile.open(descriptorFileName.c_str(),
						ios::out | ios::trunc);
				outputFile << (int) features.keypoints.size() << " 128" << endl;
				for (int i = 0; i < (int) features.keypoints.size(); ++i) {
					outputFile << (float) features.keypoints[i].pt.y << " "
							<< (float) features.keypoints[i].pt.x << " "
							<< (float) features.keypoints[i].size << " "
							<< (float) features.keypoints[i].angle << endl
							<< " ";
					for (int j = 0; j < features.descriptors.cols; ++j) {
						outputFile
								<< (int) round(
										features.descriptors.at<float>(i, j))
								<< " ";
						if ((j + 1) % 20 == 0) {
							outputFile << endl << " ";
						}
					}
					outputFile << endl;
				}
				outputFile.close();
			}
		}
	}

#if 0
	vector<string> geom_files;
	FileUtils::readFolder(argv[1], geom_files);
	map<string, vector<KeyPoint> > images;
	FileUtils::readDescriptorFiles(argv[1], geom_files, images);
	vector<string> keys = getMapKeys<string, vector<KeyPoint> >(images);
	map<string, vector<KeyPoint> >::value_type imageV(keys[3].c_str(),
			images.at(keys[3]));
	displayImage(
			("/home/andresf/Documents/POLIMI_maestria/Computer Vision/Project/oxbuild_images/"
					+ imageV.first + ".jpg").c_str(), features);
#endif

	return EXIT_SUCCESS;

}
