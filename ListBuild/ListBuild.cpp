/*
 * ListBuild.cpp
 *
 *  Created on: Jul 10, 2013
 *      Author: andresf
 */

#include "ListBuild/lists_builder.h"
#include "Common/FileUtils.h"

int main(int argc, char **argv) {

//"Usage: %s {output specification} <path_to_geometry_files_folder>"
//"\nOUTPUT:\n  -db: create list of SIFT feature files of database images.\n  -dbld: create list of SIFT feature files of database images and corresponding landmark ID.\n  -q: create list of SIFT feature files of query images.\n  -gt: create list of SIFT feature files of query images and corresponding landmark ID.\n  -cf: compute image features.\n"

//oxford5k_recognition -lists $GROUND_TRUTH_FOLDER ./
//oxford5k_recognition -gt $GROUND_TRUTH_FOLDER ./

	if (argc < 4) {
		printf(
				"\nUsage: %s [-lists|-gt] <in.ground.truth.folder> <out.lists.folder>\n\nOptions:\n"
						"\t-lists:\tcreate list of SIFT feature files of query and database images.\n"
						"\t-gt:\tcreate list of SIFT feature files of query and database images with corresponding landmark ID.\n\n",
				argv[0]);
		return EXIT_FAILURE;
	}

	vector<string> folderFiles;
	int result = FileUtils::readFolder(argv[2], folderFiles);
	if (result == EXIT_FAILURE) {
		return result;
	}

	if (string(argv[1]).compare("-lists") == 0) {

		string keypointsFilename = string(argv[3]) + "/list_queries.txt";
		vector<string> queryKeypointFiles = createListQueriesTxt(argv[2],
				folderFiles, keypointsFilename);

		keypointsFilename = string(argv[3]) + "/list_db.txt";
		createListDbTxt(argv[2], folderFiles, keypointsFilename,
				queryKeypointFiles);

	} else if (string(argv[1]).compare("-gt") == 0) {

		string keypointsFilename = string(argv[3]) + "/list_gt.txt";
		vector<string> queryKeypointFiles = createListQueriesTxt(argv[2],
				folderFiles, keypointsFilename, true);

		keypointsFilename = string(argv[3]) + "/list_db_ld.txt";
		createListDbTxt(argv[2], folderFiles, keypointsFilename,
				queryKeypointFiles, true);

	}

	return EXIT_SUCCESS;
}
