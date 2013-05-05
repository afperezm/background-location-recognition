#ifndef __string_utils_h__
#define __string_utils_h__

#include <string>
#include <vector>


using std::string;
using std::vector;

class StringUtils {
public:
	static vector<string> split(const string &s, char delim);
	static string parseLandmarkName(vector<string>::const_iterator fileName);
	static string parseImgFilename(const string keyFilename);
private:
	static vector<string> &split(const string &s, char delim,
			vector<string> &elems);
};

#endif
