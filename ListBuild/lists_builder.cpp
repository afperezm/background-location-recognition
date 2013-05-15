/*
 * lists_builder.cpp
 *
 *  Created on: May 15, 2013
 *      Author: andresf
 */

#include "lists_builder.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include "../Common/StringUtils.h"
#include "../Common/FileUtils.h"

using std::ofstream;
using std::ostringstream;
using std::endl;

void createListDbTxt(const char* folderName,
		const vector<string>& geometryFiles, string& keypointsFilename,
		const vector<string>& queryKeypointFiles, bool appendLandmarkId) {

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
		bool appendLandmarkId) {

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
