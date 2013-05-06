#ifndef __file_utils_h__
#define __file_utils_h__

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <string>
#include <vector>
#include <map>
#include <VocabLib/keys2.h>

#define KEYPOINT_FILE_EXTENSION ".key"
#define IMAGE_FILE_EXTENSION ".jpg"

using std::string;
using std::vector;
using std::map;
using cv::KeyPoint;

class FileUtils {
public:
	static int readFolder(const char* folderName, vector<string>& files);
	static int readDescriptorFiles(const char* folderName,
			const vector<string>& files,
			map<string, vector<KeyPoint> >& images);
//	static void createListDbTxt(const char* folderName,
//			vector<string>::const_iterator fileName, vector<string>* objects,
//			bool appendLandmarkId);
//	static void createListQueriesTxt(const char* folderName,
//			vector<string>::const_iterator fileName, vector<string>* objects,
//			bool appendLandmarkId);
	static int readOriginalDescriptors(char* fileName);
	static void getKeypointFilePath(string& keyfilesFolder, string& filepath);
private:
};

#endif
