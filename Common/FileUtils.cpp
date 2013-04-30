#include "FileUtils.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <string>
#include <vector>
#include <cstdlib>
#include <dirent.h>
#include <stdio.h>
#include <algorithm>
#include <map>
#include <fstream>
#include "StringUtils.h"
#include <VocabLib/keys2.h>

#include <stdlib.h>
#include <utility>

using std::string;
using std::vector;
using std::sort;

using cv::KeyPoint;
using cv::Point2f;

int FileUtils::readFolder(const char* folderName, vector<string>& files) {

	DIR *dir;
	struct dirent *ent;
	// Try opening folder
	if ((dir = opendir(folderName)) != NULL) {
		fprintf(stdout, "Opening directory [%s]\n", folderName);
		// Save all true directory names into a vector of strings
		while ((ent = readdir(dir)) != NULL) {
			// Ignore . and .. as valid folder names
			string name = string(ent->d_name);
			if (name.compare(".") != 0 && name.compare("..") != 0) {
				files.push_back(string(ent->d_name));
			}
		}
		closedir(dir);
		// Sort alphabetically vector of folder names
		sort(files.begin(), files.end());
		fprintf(stdout, "Found [%d] files\n", (int) files.size());
	} else {
		fprintf(stderr, "Could not open directory [%s]", folderName);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int FileUtils::readDescriptorFiles(const char* folderName,
		const vector<string>& files, map<string, vector<KeyPoint> >& images) {

	unsigned int totNDescriptors = 0;

	// Loop over collection of files
	for (vector<string>::const_iterator it = files.begin(); it != files.end();
			++it) {

		if (*it == files[5]) {
			break;
		}

		// Open geometry file
		fprintf(stdout, "Reading geometry file [%s]\n", (*it).c_str());
		std::ifstream infile((string(folderName) + "/" + *it).c_str());

		// Extract data from each descriptor
		string line;
		unsigned int count = 1;
		vector<KeyPoint> keypoints;
		keypoints.clear();
		while (std::getline(infile, line)) {
			if (count == 2) {
				unsigned int nDescriptors = atoi(line.data());
				totNDescriptors += nDescriptors;
			} else if (count > 2) {
				vector<string> lineSplitted = StringUtils::split(line, ' ');
				unsigned int word_id;
				float x, y, a, b, c;

				word_id = atoi(lineSplitted[0].data());
				x = atof(lineSplitted[1].data());
				y = atof(lineSplitted[2].data());
				a = atof(lineSplitted[3].data());
				b = atof(lineSplitted[4].data());
				c = atof(lineSplitted[5].data());

//				printf("word_id=[%d] x=[%fd] y=[%fd] a=[%ed] b=[%ed] c=[%ed]\n", word_id, x, y, a, b, c);

				keypoints.push_back(KeyPoint(Point2f(x, y), 10.0));
			}
			count++;
		}

		printf("Read [%d] keypoints\n", (int) keypoints.size());

		string imageName(*it);

		images.insert(
				map<string, vector<KeyPoint> >::value_type(
						imageName.substr(5, imageName.size() - 9), keypoints));

	}
	printf("Read in total [%d] keypoints\n", totNDescriptors);

	return EXIT_SUCCESS;
}

vector<string> FileUtils::readFiles(const char* folderName,
		const vector<string>& files) {

	vector<string> objects;

	// Loop over collection of files
	for (vector<string>::const_iterator file = files.begin();
			file != files.end(); ++file) {
		objects.push_back((*file));
	}

	return objects;
}

int FileUtils::readOriginalDescriptors(char* fileName) {
	int totalDescriptors = 5; // 16334970
	int descriptorSize = 128;

	FILE *fin;
	fin = fopen(fileName, "rb");
	if (fin == 0) {
		fprintf(stderr, "File [%s] couldn't be opened", fileName);
		return EXIT_FAILURE;
	}
	fprintf(stdout, "Size of unsigned char: %ld\n", sizeof(unsigned char));
	// Reading SIFT descriptors stepping by chunks of 128 bytes, ignore the last 12 bytes
	unsigned char *descriptor = new unsigned char[descriptorSize];
	for (int var = 0; var < totalDescriptors; ++var) {
		fprintf(stdout, "Reading descriptor [%d]\n", var);
		fread(descriptor, sizeof(unsigned char), descriptorSize, fin);
		for (int k = 0; k < descriptorSize; ++k) {
			float value = descriptor[k];
			fprintf(stdout, "%Fd ", value);
		}
		fprintf(stdout, "\n");
	}
	fclose(fin);

	return EXIT_SUCCESS;
}
