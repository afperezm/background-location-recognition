#include "StringUtils.h"

#include <sstream>
#include <string>
#include <vector>

#include "Constants.h"

using std::string;
using std::vector;

vector<string> StringUtils::split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

vector<string> &StringUtils::split(const string &s, char delim,
		vector<string> &elems) {
	std::stringstream ss(s);
	string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

string StringUtils::parseLandmarkName(vector<string>::const_iterator fileName) {
	string landmarkName("");
	vector<string> fileNameSplitted = StringUtils::split((*fileName), '_');
	landmarkName = string(fileNameSplitted[0]);
	for (int var = 1; var < (int) fileNameSplitted.size() - 2; ++var) {
		landmarkName = landmarkName + "_" + fileNameSplitted[var];
	}
	return landmarkName;
}

/**
 * Transforms a key filename to an image filename
 *
 * @param keyFilename String holding the path to a keyfile since the data set root folder.
 * @return Parsed image name
 */
string StringUtils::parseImgFilename(const string keyFilename, string prefix) {
	string imgFilename = StringUtils::split(keyFilename.c_str(), '/').back();
	imgFilename.resize(imgFilename.size() - 4);
	return imgFilename + (!prefix.empty() ? prefix : "") + IMAGE_FILE_EXTENSION;
}
