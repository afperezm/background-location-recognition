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

#include <iostream>

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

void createListDbTxt(const char* folderName,
		const vector<string>& geometryFiles, string& keypointsFilename,
		bool appendLandmarkId = false) {

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
				string imageName = "db/" + string(line.c_str())
						+ string(KEYPOINT_FILE_EXTENSION);
				if (appendLandmarkId == true) {
					// Position of the landmarkName in the vector of landmarks
					ostringstream temp;
					temp << ((int) landmarks.size()) - 1;
					imageName += " " + temp.str();
				}

				keypointsFile << imageName << endl;
			}

			//Close file
			infile.close();
		}
	}

	keypointsFile.close();
}

void createListQueriesTxt(const char* folderName,
		const vector<string>& geometryFiles, string& keypointsFilename,
		bool appendLandmarkId = false) {

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
			}

			//Close file
			infile.close();
		}
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

	if (argc != 4) {
		printf(
				"Usage: %s {output specification} <path_to_geometry_files_folder>"
						"\nOUTPUT:\n  -db: create list of SIFT feature files of database images.\n  -dbld: create list of SIFT feature files of database images and corresponding landmark ID.\n  -q: create list of SIFT feature files of query images.\n  -gt: create list of SIFT feature files of query images and corresponding landmark ID.\n  -cf: compute image features.\n",
				argv[0]);
		return EXIT_FAILURE;
	}

	vector<string> geometryFiles;
	vector<string> featureFilesNames;
	FileUtils::readFolder(argv[2], geometryFiles);

	ofstream outputFile;

	if (string(argv[1]).compare("-db") == 0) {
		string keypointsFilename = string(argv[3]) + "/list_db.txt";
		createListDbTxt(argv[2], geometryFiles, keypointsFilename);
	} else if (string(argv[1]).compare("-dbld") == 0) {
		string keypointsFilename = string(argv[3]) + "/list_db_ld.txt";
		createListDbTxt(argv[2], geometryFiles, keypointsFilename, true);
	} else if (string(argv[1]).compare("-q") == 0) {
		string keypointsFilename = string(argv[3]) + "/list_queries.txt";
		createListQueriesTxt(argv[2], geometryFiles, keypointsFilename);
	} else if (string(argv[1]).compare("-gt") == 0) {
		string keypointsFilename = string(argv[3]) + "/list_gt.txt";
		createListQueriesTxt(argv[2], geometryFiles, keypointsFilename, true);
	} else if (string(argv[1]).compare("-cf") == 0) {
		vector<string> images;
		FileUtils::readFolder(argv[2], images);
		vector<string>::iterator start_image;
		if (argc == 3) {
			// Set first image as start point of the loop over the folder of images
			start_image = images.begin();
		} else {
			// Set received argument as start point of the loop over the folder of images
			start_image = std::find(images.begin(), images.end(), argv[3]);
		}

		for (vector<string>::iterator image = start_image;
				image != images.end(); ++image) {
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
				cout << (int) features.keypoints.size() << " 128" << endl;
				for (int i = 0; i < (int) features.keypoints.size(); ++i) {
					outputFile << (float) features.keypoints[i].pt.x << " "
							<< (float) features.keypoints[i].pt.y << " "
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
