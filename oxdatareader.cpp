/*
 * oxdatareader.cpp
 *
 *  Created on: Apr 13, 2013
 *      Author: andresf
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <string>
#include <vector>
#include <map>

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <utility>

#include "StringUtils.h"
#include "FileUtils.h"

using std::endl;
using std::string;
using std::vector;
using std::map;
using std::ofstream;
using std::ios;

using namespace cv;

struct Features {
	vector<KeyPoint> keypoints;
	Mat descriptors;
};

void createListDbTxt(const char* folderName,
		vector<string>::const_iterator fileName, vector<string>* objects,
		bool appendLandmarkId) {
	if ((*fileName).find("1") != string::npos
			&& (*fileName).find("query") == string::npos) {

		string landmarkName;
		if (appendLandmarkId == true) {
			landmarkName = StringUtils::parseLandmarkName(fileName);
		}

		std::ifstream infile((string(folderName) + "/" + *fileName).c_str());

		// Extract data from file
		string line;

		while (std::getline(infile, line)) {
			string imageName = "db/" + string(line.c_str())
					+ string(KEYPOINT_FILE_EXTENSION);
			if (appendLandmarkId == true) {
				imageName += " " + string(landmarkName.c_str());
			}
			(*objects).push_back(imageName);
		}

		//Close file
		infile.close();
	}
}

void createListQueriesTxt(const char* folderName,
		vector<string>::const_iterator fileName, vector<string>* objects,
		bool appendLandmarkId) {

	if ((*fileName).find("query") != string::npos) {
		// Open file
		fprintf(stdout, "Reading file [%s]\n", (*fileName).c_str());

		string landmarkName;
		if (appendLandmarkId == true) {
			landmarkName = StringUtils::parseLandmarkName(fileName);
		}

		std::ifstream infile((string(folderName) + "/" + *fileName).c_str());

		// Extract data from file
		string line;

		while (std::getline(infile, line)) {
			vector<string> lineSplitted = StringUtils::split(line, ' ');
			string qName = "queries/" + lineSplitted[0].substr(5)
					+ string(KEYPOINT_FILE_EXTENSION);

			if (appendLandmarkId == true) {
				qName += " " + string(landmarkName.c_str());
			}

			(*objects).push_back(qName);
		}

		//Close file
		infile.close();
	}

}

void displayImage(string imageName, vector<KeyPoint> keypoints) {

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

int main(int argc, char **argv) {

//	vector<string> geom_files;
//	FileUtils::readFolder(argv[1], geom_files);
//	map<string, vector<KeyPoint> > images;
//	FileUtils::readDescriptorFiles(argv[1], geom_files, images);
//	vector<string> keys = getMapKeys<string, vector<KeyPoint> >(images);
//	map<string, vector<KeyPoint> >::value_type imageV(keys[3].c_str(),
//			images.at(keys[3]));

//	displayImage(
//			("/home/andresf/Documents/POLIMI_maestria/Computer Vision/Project/oxbuild_images/"
//					+ imageV.first + ".jpg").c_str(), features);

//	if (argc != 3) {
//		printf(
//				"Usage: %s {output specification} <path_to_geometry_files>"
//						"\nOUTPUT:\n  -db: create list of SIFT feature files of database images.\n  -dbld: create list of SIFT feature files of database images and corresponding landmark ID.\n  -q: create list of SIFT feature files of query images.\n  -qld: create list of SIFT feature files of query images and corresponding landmark ID.",
//				argv[0]);
//		return EXIT_FAILURE;
//	}
//
//	vector<string> geometry_files;
//	vector<string> featureFilesNames;
//	FileUtils::readFolder(argv[2], &geometry_files);

	ofstream outputFile;

	vector<string> images;
	FileUtils::readFolder(argv[1], images);
//	vector<string>::iterator imagee = std::find(images.begin(),images.end(),"magdalen_000985.jpg");

	for (vector<string>::iterator image = images.begin(); image != images.end();
			++image) {
		if ((*image).find(".jpg") != string::npos) {
			printf("%s\n", (*image).c_str());
			Features features = detectAndDescribeFeatures(
					argv[1] + string("/") + (*image));

			string descriptorFileName(argv[1]);
			descriptorFileName += "/" + (*image).substr(0, (*image).size() - 4)
					+ ".key";
			printf("Writing feature descriptors to [%s]\n",
					descriptorFileName.c_str());
			outputFile.open(descriptorFileName.c_str(), ios::out | ios::trunc);
			outputFile << (int) features.keypoints.size() << " 128" << endl;
			for (int i = 0; i < (int) features.keypoints.size(); ++i) {
				outputFile << (float) features.keypoints[i].pt.x << " "
						<< (float) features.keypoints[i].pt.y << " "
						<< (float) features.keypoints[i].size << " "
						<< (float) features.keypoints[i].angle << endl << " ";
				for (int j = 0; j < features.descriptors.cols; ++j) {
					outputFile
							<< (int) round(features.descriptors.at<float>(i, j))
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

//	if (string(argv[1]).compare("-db") == 0) {
//		featureFilesNames.clear();
//		FileUtils::readFiles(argv[2], &geometry_files, &featureFilesNames,
//				&createListDbTxt);
//		outputFile.open("./list_db.txt", ios::out | ios::trunc);
//		for (vector<string>::iterator it = featureFilesNames.begin();
//				it != featureFilesNames.end(); ++it) {
//			outputFile << *it << endl;
//		}
//		outputFile.close();
//	} else if (string(argv[1]).compare("-dbld") == 0) {
//		featureFilesNames.clear();
//		FileUtils::readFiles(argv[1], &geometry_files, &featureFilesNames,
//				&createListDbTxt, true);
//		outputFile.open("./list_db_ld.txt", ios::out | ios::trunc);
//		for (vector<string>::iterator it = featureFilesNames.begin();
//				it != featureFilesNames.end(); ++it) {
//			outputFile << *it << endl;
//		}
//		outputFile.close();
//	} else if (string(argv[1]).compare("-q") == 0) {
//		featureFilesNames.clear();
//		FileUtils::readFiles(argv[1], &geometry_files, &featureFilesNames,
//				&createListQueriesTxt);
//		outputFile.open("./list_queries.txt", ios::out | ios::trunc);
//		for (vector<string>::iterator it = featureFilesNames.begin();
//				it != featureFilesNames.end(); ++it) {
//			outputFile << *it << endl;
//		}
//		outputFile.close();
//	} else if (string(argv[1]).compare("-qld") == 0) {
//		featureFilesNames.clear();
//		FileUtils::readFiles(argv[1], &geometry_files, &featureFilesNames,
//				&createListQueriesTxt, true);
//		outputFile.open("./list_gt.txt", ios::out | ios::trunc);
//
//		for (vector<string>::iterator it = featureFilesNames.begin();
//				it != featureFilesNames.end(); ++it) {
//			outputFile << *it << endl;
//		}
//		outputFile.close();
//	}

	return EXIT_SUCCESS;

}
