/*
 * oxdatareader.cpp
 *
 *  Created on: Apr 13, 2013
 *      Author: andresf
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <fstream>
#include <stdlib.h>
#include <vector>

#include "Common/StringUtils.h"

using cv::drawKeypoints;
using cv::imread;
using cv::imshow;
using cv::namedWindow;
using cv::waitKey;
using cv::DataType;
using cv::DrawMatchesFlags;
using cv::KeyPoint;
using cv::Mat;
using cv::Point;

using std::endl;
using std::map;
using std::ofstream;
using std::vector;

void displayImage(string& imageName, vector<KeyPoint>& keypoints);

int main(int argc, char **argv) {

//	if (argc != 4) {
//		printf(
//				"Usage: %s {output specification} <path_to_geometry_files_folder>"
//						"\nOUTPUT:\n  -db: create list of SIFT feature files of database images.\n  -dbld: create list of SIFT feature files of database images and corresponding landmark ID.\n  -q: create list of SIFT feature files of query images.\n  -gt: create list of SIFT feature files of query images and corresponding landmark ID.\n  -cf: compute image features.\n",
//				argv[0]);
//		return EXIT_FAILURE;
//	}

	if (argc < 4) {
		printf(
				"\nUsage: %s <in.queries.ground.truth> <in.db.ground.truth> "
						"<in.geom.ranked.candidates> [occurrence_matrix.txt] [voted_landmarks.txt]\n\n",
				argv[0]);
		return EXIT_FAILURE;
	}

	printf(
			"Computing candidate occurrences and voting matrices for varying number of candidates\n");

	map<string, int> query_ld;
	map<string, vector<int> > db_ld;
	std::ifstream infile;
	ofstream votes_file, candidates_occurence;
	string line;
	vector<string> lineSplitted;
	int num_landmarks = 11;
	int occurrence = 0;
	Mat hist;

	// Reading file of ground truth landmark id of query images
	printf("  Reading file of ground truth landmark id of query images\n");
	// Note: each query has at most one landmark id
	// Line format: <query_image_name> <landmark_id>
	infile.open(argv[1], std::fstream::in);
	while (std::getline(infile, line)) {
		lineSplitted = StringUtils::split(line, ' ');
		string query_name = lineSplitted[0];
		int landmark_id = atoi(lineSplitted[1].c_str());
		// Save results on map of string keys to integer values
		query_ld[query_name] = landmark_id;
	}
	infile.close();

	// Reading file of ground truth landmark id of db images
	printf("  Reading file of ground truth landmark id of db images\n");
	// Note: each db might have more than one landmark id
	// Line format: <db_image_name> <landmark_id>
	infile.open(argv[2], std::fstream::in);
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
	printf("  Reading candidates file\n");
	infile.open(argv[3], std::fstream::in);

	if (argc >= 6) {
		printf("  Writing occurrence matrix to file [%s]\n", argv[4]);
		candidates_occurence.open(argv[4], std::fstream::out);
	} else {
		printf("  Writing occurrence matrix to file [occurrence_matrix.txt]\n");
		candidates_occurence.open("occurrence_matrix.txt", std::fstream::out);
	}

	if (argc >= 7) {
		printf("  Writing occurrence matrix to file [%s]\n", argv[5]);
		votes_file.open(argv[5], std::fstream::out);
	} else {
		printf("  Writing occurrence matrix to file [voted_landmarks.txt]\n");
		votes_file.open("voted_landmarks.txt", std::fstream::out);
	}

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
	}
	votes_file.close();
	candidates_occurence.close();
	infile.close();

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
