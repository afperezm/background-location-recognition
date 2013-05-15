#include "FileUtils.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <VocabLib/keys2.h>

#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>

#include "Constants.h"
#include "StringUtils.h"

using cv::KeyPoint;
using cv::Point2f;

using std::string;
using std::vector;
using std::sort;

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
		fprintf(stdout, "  Found [%d] files\n", (int) files.size());
	} else {
		fprintf(stderr, "  Could not open directory [%s]", folderName);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

void FileUtils::getKeypointFilePath(string& keyfilesFolder, string& filepath) {
	filepath.resize(filepath.size() - 4);
	filepath += KEYPOINT_FILE_EXTENSION;
}
