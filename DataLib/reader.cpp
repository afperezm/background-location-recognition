/*
 * reader.cpp
 *
 *  Created on: May 14, 2013
 *      Author: andresf
 */

#include <fstream>

#include "Common/StringUtils.h"
#include "DataLib/reader.h"

using cv::DataType;
using std::ifstream;

void readKeypoints(const char *filename, vector<KeyPoint>& keypoints,
		Mat& descriptors) {

	printf("Reading keypoints from file [%s]\n", filename);

	int num_keys = 0;
	short int *keys;
	keypt_t* info = NULL;
	num_keys = ReadKeyFile(filename, &keys, &info);

	int dim = 128;

	keypoints.clear();
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

map<string, vector<KeyPoint> > readDescriptorFiles(const char* folderPath,
		const vector<string>& files) {

	map<string, vector<KeyPoint> > images;

	unsigned int totNDescriptors = 0;

	// Loop over collection of files
	for (string file : files) {

		if (file == files[5]) {
			break;
		}

		// Open geometry file
		fprintf(stdout, "Reading geometry file [%s]\n", file.c_str());

		ifstream infile((string(folderPath) + "/" + file).c_str());

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

				printf("word_id=[%d] x=[%fd] y=[%fd] a=[%ed] b=[%ed] c=[%ed]\n",
						word_id, x, y, a, b, c);

				keypoints.push_back(KeyPoint(Point2f(x, y), 10.0));
			}
			count++;
		}
		infile.close();

		printf("Read [%d] keypoints\n", (int) keypoints.size());

		images.insert(
				map<string, vector<KeyPoint> >::value_type(
						file.substr(5, file.size() - 9), keypoints));

	}
	printf("Read in total [%d] keypoints\n", totNDescriptors);

	return images;
}

int readOriginalDescriptors(char* fileName) {
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

vector<Point2f> readMask(const char*maskFilepath) {

	vector<Point2f> points;
	string line;

	printf("Reading mask file [%s]\n", maskFilepath);

	ifstream infile(maskFilepath, std::fstream::in);

	std::getline(infile, line);
	int num_vertices = atoi(line.c_str());

	while (std::getline(infile, line)) {
		vector<string> lineSplitted = StringUtils::split(line, ' ');
		points.push_back(
				Point2f(atof(lineSplitted[0].c_str()),
						atof(lineSplitted[1].c_str())));
	}

	infile.close();

	printf("  Read [%d] pairs of coordinates\n", num_vertices);

	return points;
}
