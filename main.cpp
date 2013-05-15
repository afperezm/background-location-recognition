/*
 * oxdatareader.cpp
 *
 *  Created on: Apr 13, 2013
 *      Author: andresf
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sys/stat.h>
#include <VocabLib/keys2.h>

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>

#include "Common/StringUtils.h"
#include "Common/FileUtils.h"
#include "DataLib/reader.h"
#include "FeatureExtract/extractor.h"
#include "GeomVerify/matcher.h"
#include "ListBuild/lists_builder.h"

using cv::Point;
using cv::DataType;
using cv::DrawMatchesFlags;
using cv::drawKeypoints;
using cv::imread;
using cv::imshow;
using cv::namedWindow;
using cv::waitKey;

using std::cout;
using std::endl;
using std::find;
using std::ios;
using std::map;
using std::ofstream;
using std::ostringstream;
using std::string;
using std::vector;

void displayImage(string& imageName, vector<KeyPoint>& keypoints);

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

	/**
	 * For this purpose it reads and uses the query images ground truth
	 * labels and the matrix of candidate db images for each query image.
	 *
	 * Additionally it computes the voting matrix. The idea behind ranking
	 * several images at once is to get a more reliable landmark prediction
	 * by implementing a voting scheme on this ranked list.
	 */
	if (string(argv[1]).compare("-perf") == 0) {

		map<string, int> query_ld;
		map<string, vector<int> > db_ld;
		std::ifstream infile;
		ofstream votes_file, candidates_occurence;
		string line;
		vector<string> lineSplitted;
//		int num_query_images = 55;
//		int num_candidates = 50;
		int num_landmarks = 11;
		int occurrence = 0;
		Mat hist;

		// Reading file of ground truth landmark id of query images
		// Note: each query has at most one landmark id
		// Line format: <query_image_name> <landmark_id>
		infile.open(argv[2], std::fstream::in);
		while (std::getline(infile, line)) {
			lineSplitted = StringUtils::split(line, ' ');
			string query_name = lineSplitted[0];
			int landmark_id = atoi(lineSplitted[1].c_str());
			// Save results on map of string keys to integer values
			query_ld[query_name] = landmark_id;
		}
		infile.close();

		// Reading file of ground truth landmark id of db images
		// Note: each db might have more than one landmark id
		// Line format: <db_image_name> <landmark_id>
		infile.open(argv[3], std::fstream::in);
		while (std::getline(infile, line)) {
			lineSplitted = StringUtils::split(line, ' ');
			string db_name = lineSplitted[0];
			int landmark_id = atoi(lineSplitted[1].c_str());
			// Save results on map of string keys to vector of integers
			// since it might be associated with more than one landmark id
			db_ld[db_name].push_back(landmark_id);
		}
		infile.close();

		// Reading candidates file
		infile.open(argv[4], std::fstream::in);
		votes_file.open("voted_landmarks.txt", std::fstream::out);
		candidates_occurence.open("candidates_occurrence.txt",
				std::fstream::out);
		int line_number = 0;
		while (std::getline(infile, line)) {
			lineSplitted = StringUtils::split(line, ' ');
			string query_name = lineSplitted[0];
			lineSplitted.erase(lineSplitted.begin());
			// Add to the beginning of the line the query name and its landmark id
//			candidates_occurence << query_name << " " << query_ld[query_name] << ": ";
//			votes_file << query_name << " " << query_ld[query_name] << ": ";
			// Loop over candidates
			int k = 0;
			hist = Mat::zeros(1, num_landmarks, DataType<int>::type);

			for (string candidate : lineSplitted) {
				// By default no occurrence exist
				occurrence = 0;
				// Loop over associated landmark ids for a candidate
				for (int landmark_id : db_ld[candidate]) {
					// Mark coincidence between candidate and query
					if (query_ld[query_name] == landmark_id) {
						occurrence = 1;
					}
					// Load histogram counts of landmark ids
					hist.at<int>(landmark_id)++;}
				Point max_landmark;
				// Find landmark with max votes among the set of k considered candidates
				minMaxLoc(hist, NULL, NULL, NULL, &max_landmark, Mat());

				// Store occurrence and max voted landmark
				candidates_occurence << occurrence << " ";
				votes_file << max_landmark.x << " ";
				k++;
			}
			candidates_occurence << endl;
			votes_file << endl;
			line_number++;
		}
		votes_file.close();
		candidates_occurence.close();
		infile.close();

		return EXIT_SUCCESS;
	}

	if (string(argv[1]).compare("-gvc") == 0) {

		// TODO Check if imagesFolderpath exists
		string imagesFolderpath(argv[2]);
		// TODO Check if keysFolderpath exists
		string keysFolderpath(argv[3]);

		string line, templateFilename, sourceFilename;
		std::ifstream candidatesFile(argv[4], std::fstream::in);
		ofstream candidatesGvFile("candidates_gv.txt", std::fstream::out);
		ofstream candidatesInliersFile("candidates_inliers.txt",
				std::fstream::out);

		double ransacReprojThreshold;
		double proximityThreshold, similarityThreshold;
		if (argc >= 7) {
			ransacReprojThreshold = atof(argv[5]);
		} else {
			ransacReprojThreshold = 10.0;
		}
		if (argc >= 8) {
			proximityThreshold = atof(argv[6]);
		} else {
			proximityThreshold = 100.0;
		}
		if (argc >= 9) {
			similarityThreshold = atof(argv[7]);
		} else {
			similarityThreshold = 0.8;
		}

		Mat candidates_inliers, candidates_inliers_idx;

		while (std::getline(candidatesFile, line)) {
			candidates_inliers = Mat::zeros(1, 50, DataType<int>::type);
			candidates_inliers_idx = Mat::zeros(1, 50, DataType<int>::type);

			vector<string> splitted_line = StringUtils::split(line.c_str(),
					' ');
			templateFilename = splitted_line[0];
			// TODO Check that templateImgFilepath is a valid image filepath starting from the images folder as root
			string templateImgFilepath(
					imagesFolderpath + "/"
							+ StringUtils::parseImgFilename(templateFilename));

			for (int i = 1; i < (int) splitted_line.size(); ++i) {
				sourceFilename = splitted_line[i];

				// TODO Check that templateFilename is a valid template keypoints filepath starting from the keypoints folder as root
				string templateKeypointsFilepath(
						keysFolderpath + "/" + templateFilename);

				// TODO Check that sourceImgFilepath is a valid image filepath starting from the images folder as root
				string sourceImgFilepath(
						imagesFolderpath + "/"
								+ StringUtils::parseImgFilename(
										sourceFilename));

				// TODO Check that templateFilename is a valid template keypoints filepath starting from the keypoints folder as root
				string sourceKeypointsFilepath(
						keysFolderpath + "/" + sourceFilename);

				int num_inliers = geometricVerification(templateImgFilepath,
						templateKeypointsFilepath, sourceImgFilepath,
						sourceKeypointsFilepath, ransacReprojThreshold,
						proximityThreshold, similarityThreshold);

				candidates_inliers.at<int>(i - 1) = num_inliers;
			}
			sortIdx(candidates_inliers, candidates_inliers_idx,
					CV_SORT_DESCENDING);
			// Print number of inliers for each candidate
			// candidatesGvFile << templateFilename << " " << candidates_inliers << endl;
			// Print indexes of ordered Mat of candidates inliers
			// candidatesGvFile << templateFilename << " " << candidates_inliers_idx << endl;
			candidatesGvFile << templateFilename;
			candidatesInliersFile << templateFilename;
			for (int j = 0; j < candidates_inliers_idx.cols; ++j) {
				// Print index of ordered element at j position
				// candidatesGvFile << " " << candidates_inliers.at<int>( candidates_inliers_idx.at<int>(j));
				// Print ordered element at j+1 position since the first element of candidate's line is the query name
				candidatesGvFile << " "
						<< splitted_line[candidates_inliers_idx.at<int>(j) + 1];
				candidatesInliersFile << " "
						<< candidates_inliers.at<int>(
								candidates_inliers_idx.at<int>(j));
			}
			candidatesGvFile << endl;
			candidatesInliersFile << endl;
		}
		candidatesGvFile.close();
		candidatesInliersFile.close();
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
			proximityThreshold = 100.0;
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

		int result = geometricVerification(templateImgFilepath,
				templateKeypointsFilepath, sourceImgFilepath,
				sourceKeypointsFilepath, ransacReprojThreshold,
				proximityThreshold, similarityThreshold);

		return result == -1 ? EXIT_FAILURE : EXIT_SUCCESS;
	}

	vector<string> folderFiles;
	int result = FileUtils::readFolder(argv[2], folderFiles);
	if (result == EXIT_FAILURE) {
		return result;
	}

	ofstream outputFile;

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
	} else if (string(argv[1]).compare("-visualkp") == 0) {

		string keypointsFolderPath(argv[2]);
		string imagesFolderPath(argv[3]);
		string outputImagesFolderPath(argv[4]);

		Features features;
		for (string filename : folderFiles) {
			if (filename.find(".key") != string::npos) {

				string keypointFilepath = keypointsFolderPath + "/" + filename;
				readKeypoints(keypointFilepath.c_str(), features.keypoints,
						features.descriptors);

				string imgPath = imagesFolderPath + "/"
						+ StringUtils::parseImgFilename(filename);

				printf("Reading image [%s]\n", imgPath.c_str());
				Mat img = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);

				drawKeypoints(img, features.keypoints, img, cvScalar(255, 0, 0),
						DrawMatchesFlags::DEFAULT);
				string imgWithKeysPath = outputImagesFolderPath + "/"
						+ StringUtils::parseImgFilename(filename, "_with_keys");
				printf("Writing image [%s]\n", imgWithKeysPath.c_str());
				cv::imwrite(imgWithKeysPath, img);

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
	string op_getkey(map<string, vector<KeyPoint> >::value_type pair) {
		return pair.first;
	}

	template<class K, class V> vector<K> getMapKeys(map<K, V>& images) {
		vector<K> keys;
		keys.resize(images.size());
		std::transform(images.begin(), images.end(), keys.begin(), op_getkey);
		return keys;
	}
#endif

	return EXIT_SUCCESS;
}

void displayImage(string& imageName, vector<KeyPoint>& keypoints) {

	printf("Displaying [%d] keypoints for image [%s]\n", (int) keypoints.size(),
			imageName.c_str());

	Mat img = imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);

	namedWindow(imageName.c_str(), CV_WINDOW_NORMAL);
	drawKeypoints(img, keypoints, img, cvScalar(255, 0, 0),
			DrawMatchesFlags::DEFAULT);
	imshow(imageName.c_str(), img);
	cvResizeWindow(imageName.c_str(), 480, 640);

	waitKey(0);

}
