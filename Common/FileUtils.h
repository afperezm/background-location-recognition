#ifndef __file_utils_h__
#define __file_utils_h__

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <string>
#include <vector>
#include <map>
#include <VocabLib/keys2.h>

using std::string;
using std::vector;
using std::map;
using cv::KeyPoint;

namespace FileUtils {
int readFolder(const char* folderName, vector<string>& files);
int readDescriptorFiles(const char* folderName, const vector<string>& files,
		map<string, vector<KeyPoint> >& images);
int readOriginalDescriptors(char* fileName);
void getKeypointFilePath(string& keyfilesFolder, string& filepath);
} // namespace FileUtils

#endif
