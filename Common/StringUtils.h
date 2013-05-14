#ifndef __string_utils_h__
#define __string_utils_h__

#include <string>
#include <vector>

using std::string;
using std::vector;

namespace StringUtils {

vector<string> split(const string &s, char delim);
string parseLandmarkName(vector<string>::const_iterator fileName);
string parseImgFilename(const string keyFilename);
vector<string> &split(const string &s, char delim, vector<string> &elems);

} // namespace StringUtils

#endif
