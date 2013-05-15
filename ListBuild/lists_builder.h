/*
 * lists_builder.h
 *
 *  Created on: May 15, 2013
 *      Author: andresf
 */

#ifndef LISTS_BUILDER_H_
#define LISTS_BUILDER_H_

#include <vector>
#include <string>

using std::ios;
using std::vector;
using std::string;

void createListDbTxt(const char* folderName,
		const vector<string>& geometryFiles, string& keypointsFilename,
		const vector<string>& queryKeypointFiles,
		bool appendLandmarkId = false);

vector<string> createListQueriesTxt(const char* folderName,
		const vector<string>& geometryFiles, string& keypointsFilename,
		bool appendLandmarkId = false);

#endif /* LISTS_BUILDER_H_ */
