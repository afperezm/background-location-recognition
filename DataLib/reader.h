/*
 * reader.h
 *
 *  Created on: May 14, 2013
 *      Author: andresf
 */

#ifndef READER_H_
#define READER_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <stdlib.h>
#include <vector>
#include <string>
#include <map>

#include <VocabLib/keys2.h>

using std::vector;
using std::string;
using std::map;
using cv::Mat;
using cv::KeyPoint;
using cv::Point2f;

/**
 * Read all the keypoints and descriptors stored in the file specified
 * by <b>filename</b> and stores them in the data structures specified
 * by <b>keypoints</b> and <b>descriptors</b> accordingly.
 *
 * @param filename Path to a keypoints file
 * @param keypoints Reference to a vector of keypoints
 * @param descriptors Reference to a matrix of descriptors where ith row corresponds to ith keypoint
 */
void readKeypoints(const char *filename, vector<KeyPoint>& keypoints,
		Mat& descriptors);

/**
 *
 * @param folderName
 * @param files
 * @return
 */
map<string, vector<KeyPoint> > readDescriptorFiles(const char* folderName, const vector<string>& files);

/**
 *
 * @param fileName
 * @return
 */
int readOriginalDescriptors(char* fileName);

#endif /* READER_H_ */
