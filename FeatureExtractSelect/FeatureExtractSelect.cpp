/*
 * FeatureExtractSelect.cpp
 *
 *  Created on: Jul 10, 2013
 *      Author: andresf
 */

#include <fstream>

#include "Common/Constants.h"
#include "Common/StringUtils.h"
#include "Common/FileUtils.h"
#include "DataLib/reader.h"
#include "FeatureExtractSelect/extractor.h"

using cv::drawKeypoints;
using cv::imread;
using cv::pointPolygonTest;
using cv::DrawMatchesFlags;
using cv::KeyPoint;
using cv::Mat;
using cv::Point2f;

using std::endl;
using std::ios;
using std::ofstream;

void writeFeaturesToFile(string& outputFilepath, Features& keypoints);

int main(int argc, char **argv) {

	if (argc != 4 && argc != 5) {
		printf(
				"\nUsage:\n"
						"\t%s -cf <in.imgs.folder> <out.keys.folder> {in.startimg.id}\n\n"
						"\t%s -featsel <in.query.keys.folder> <in.queries.mask.folder> <out.queries.maskedkeys.folder>\n\n"
						"\t%s -visualkp <in.query.keys.folder> <in.query.imgs.folder> <out.query.withkeys.folder>\n\n"
						"Options:\n"
						"\t-cf:\t\n"
						"\t-featseal:\t\n"
						"\t-visualkp:\t\n\n", argv[0], argv[0], argv[0]);
		return EXIT_FAILURE;
	}

	vector<string> folderFiles;
	int result = FileUtils::readFolder(argv[2], folderFiles);
	if (result == EXIT_FAILURE) {
		return result;
	}

	if (string(argv[1]).compare("-cf") == 0) {
		vector<string>::iterator start_image;
		if (argc == 4) {
			// Set first image as start point of the loop over the folder of images
			start_image = folderFiles.begin();
		} else {
			// Set received argument as start point of the loop over the folder of images
			start_image = std::find(folderFiles.begin(), folderFiles.end(),
					argv[4]);
		}

		for (vector<string>::iterator image = start_image;
				image != folderFiles.end(); ++image) {
			if ((*image).find(".jpg") != string::npos) {
				printf("%s\n", (*image).c_str());
				Features features = detectAndDescribeFeatures(
						argv[2] + string("/") + (*image));

				string descriptorFileName(argv[3]);
				descriptorFileName += "/"
						+ (*image).substr(0, (*image).size() - 4) + ".key";
				writeFeaturesToFile(descriptorFileName, features);
			}
		}
	} else if (string(argv[1]).compare("-featsel") == 0) {
		string keypointsFolderPath(argv[2]);
		string masksFolderPath(argv[3]);
		string outputFolderPath(argv[4]);

		vector<string> maskFiles;

		Features features, selectedFeatures;

		for (string filename : folderFiles) {
			if (filename.find(".key") != string::npos) {
				string keypointFilepath = keypointsFolderPath + "/" + filename;
				readKeypoints(keypointFilepath.c_str(), features.keypoints,
						features.descriptors);

				StringUtils::split(filename.c_str(), '/').back();
				filename.resize(filename.size() - 4);

				string maskPath = masksFolderPath + "/" + filename
						+ MASK_FILE_EXTENSION;

				vector<Point2f> polygon = readMask(maskPath.c_str());

				printf("Selecting features\n");
				int count = 0;
				selectedFeatures.keypoints.clear();
				for (KeyPoint p : features.keypoints) {
					int inCont = pointPolygonTest(polygon, p.pt, false);
					if (inCont != -1) {
						selectedFeatures.keypoints.push_back(p);
						selectedFeatures.descriptors.push_back(
								features.descriptors.row(count));
					}
					count++;
				}
				printf("  Selected [%d] features\n",
						(int) selectedFeatures.keypoints.size());

				string outputFeaturesPath = outputFolderPath + "/" + filename
						+ KEYPOINT_FILE_EXTENSION;

				writeFeaturesToFile(outputFeaturesPath, selectedFeatures);
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
				Mat img = imread(imgPath, CV_LOAD_IMAGE_COLOR);

				drawKeypoints(img, features.keypoints, img, cvScalar(255, 0, 0),
						DrawMatchesFlags::DEFAULT);
				string imgWithKeysPath = outputImagesFolderPath + "/"
						+ StringUtils::parseImgFilename(filename, "_with_keys");
				printf("Writing image [%s]\n", imgWithKeysPath.c_str());
				cv::imwrite(imgWithKeysPath, img);

			}
		}
	}

	return EXIT_SUCCESS;
}

void writeFeaturesToFile(string& outputFilepath, Features& features) {
	ofstream outputFile;
	printf("Writing feature descriptors to [%s]\n", outputFilepath.c_str());
	outputFile.open(outputFilepath.c_str(), ios::out | ios::trunc);
	outputFile << (int) features.keypoints.size() << " 128" << endl;
	for (int i = 0; i < (int) features.keypoints.size(); ++i) {
		outputFile << (float) features.keypoints[i].pt.y << " "
				<< (float) features.keypoints[i].pt.x << " "
				<< (float) features.keypoints[i].size << " "
				<< (float) features.keypoints[i].angle << endl << " ";
		for (int j = 0; j < features.descriptors.cols; ++j) {
			outputFile << (int) round(features.descriptors.at<float>(i, j))
					<< " ";
			if ((j + 1) % 20 == 0) {
				outputFile << endl << " ";
			}
		}
		outputFile << endl;
	}
	outputFile.close();
}
