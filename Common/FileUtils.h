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
/**
 * Opens a directory and saves all the files names onto a vector of strings, it
 * returns a status flag for reporting any error during the opening of the folder.
 *
 * @param folderName Path to the folder to be opened
 * @param files Reference to a vector where all the files names will be saved
 * @return Status flag, 0 for success and 1 for failure
 */
int readFolder(const char* folderfolderPath, vector<string>& files);
} // namespace FileUtils

#endif
