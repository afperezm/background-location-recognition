/*
 * GeomVerify.cpp
 *
 *  Created on: Jul 10, 2013
 *      Author: andresf
 */
#include <fstream>
#include <stdlib.h>
#include <opencv2/highgui/highgui.hpp>

#include "Common/StringUtils.h"
#include "GeomVerify/matcher.h"

using cv::DataType;
using cv::Mat;

using std::ofstream;
using std::endl;

int main(int argc, char **argv) {

	if (argc != 8 && argc != 9) {
		printf(
				"\nUsage:\n"
						"\t%s -gvc <in.imgs.folder> <in.keys.root.folder> <in.ranked.candidates> <out.geom.ranked.candidates> <out.geom.ranked.candidates.inliers> <in.reprojection.threshold> <in.similarity.threshold>\n\n"
						"\t%s -gv <in.imgs.folder> <in.keys.root.folder> <in.template.img> <in.src.img> <reprj.threshold> <sim.threshold>\n\n"
						"Options:\n"
						"\t-gvc:\t\n"
						"\t-gv:\t\n\n", argv[0], argv[0]);
		return EXIT_FAILURE;
	}

	if (string(argv[1]).compare("-gvc") == 0) {

		string images_folder_in(argv[2]);
		string keys_root_folder_in(argv[3]);
		string candidates_file_in(argv[4]);

		std::ifstream f_candidates_in(candidates_file_in, std::fstream::in);
		ofstream f_candidates_gv_out, f_candidates_inliers_out;

		double ransacReprojThreshold, similarityThreshold;

		if (argc >= 6) {
			f_candidates_gv_out.open(argv[5], std::fstream::out);
		} else {
			f_candidates_gv_out.open("geom_ranked_candidates.txt",
					std::fstream::out);
		}
		if (argc >= 7) {
			f_candidates_inliers_out.open(argv[6], std::fstream::out);
		} else {
			f_candidates_inliers_out.open("geom_ranked_candidates_inliers.txt",
					std::fstream::out);
		}
		if (argc >= 8) {
			ransacReprojThreshold = atof(argv[7]);
		} else {
			ransacReprojThreshold = 10.0;
		}
		if (argc >= 9) {
			similarityThreshold = atof(argv[8]);
		} else {
			similarityThreshold = 0.8;
		}

		string line, templateFilename, sourceFilename;
		Mat candidates_inliers, candidates_inliers_idx;

		while (std::getline(f_candidates_in, line)) {
			candidates_inliers = Mat::zeros(1, 50, DataType<int>::type);
			candidates_inliers_idx = Mat::zeros(1, 50, DataType<int>::type);

			vector<string> splitted_line = StringUtils::split(line.c_str(),
					' ');
			templateFilename = splitted_line[0];
			// TODO Check that templateImgFilepath is a valid image filepath starting from the images folder as root
			string templateImgFilepath(
					images_folder_in + "/"
							+ StringUtils::parseImgFilename(templateFilename));

			for (int i = 1; i < (int) splitted_line.size(); ++i) {
				sourceFilename = splitted_line[i];

				// TODO Check that templateFilename is a valid template keypoints filepath starting from the keypoints folder as root
				string templateKeypointsFilepath(
						keys_root_folder_in + "/" + templateFilename);

				// TODO Check that sourceImgFilepath is a valid image filepath starting from the images folder as root
				string sourceImgFilepath(
						images_folder_in + "/"
								+ StringUtils::parseImgFilename(
										sourceFilename));

				// TODO Check that templateFilename is a valid template keypoints filepath starting from the keypoints folder as root
				string sourceKeypointsFilepath(
						keys_root_folder_in + "/" + sourceFilename);

				int num_inliers = geometricVerification(templateImgFilepath,
						templateKeypointsFilepath, sourceImgFilepath,
						sourceKeypointsFilepath, ransacReprojThreshold,
						similarityThreshold);

				candidates_inliers.at<int>(i - 1) = num_inliers;
			}
			sortIdx(candidates_inliers, candidates_inliers_idx,
					CV_SORT_DESCENDING);
			// Print number of inliers for each candidate
			// candidatesGvFile << templateFilename << " " << candidates_inliers << endl;
			// Print indexes of ordered Mat of candidates inliers
			// candidatesGvFile << templateFilename << " " << candidates_inliers_idx << endl;
			f_candidates_gv_out << templateFilename;
			f_candidates_inliers_out << templateFilename;
			for (int j = 0; j < candidates_inliers_idx.cols; ++j) {
				// Print index of ordered element at j position
				// candidatesGvFile << " " << candidates_inliers.at<int>( candidates_inliers_idx.at<int>(j));
				// Print ordered element at j+1 position since the first element of candidate's line is the query name
				f_candidates_gv_out << " "
						<< splitted_line[candidates_inliers_idx.at<int>(j) + 1];
				f_candidates_inliers_out << " "
						<< candidates_inliers.at<int>(
								candidates_inliers_idx.at<int>(j));
			}
			f_candidates_gv_out << endl;
			f_candidates_inliers_out << endl;
		}
		f_candidates_gv_out.close();
		f_candidates_inliers_out.close();
		f_candidates_in.close();

	} else if (string(argv[1]).compare("-gv") == 0) {

		string imagesFolderpath(argv[2]);
		// TODO Check if imagesFolderpath exists
		string keysFolderpath(argv[3]);
		// TODO Check if keysFolderpath exists

		string templateFilename(argv[4]);
		string sourceFilename(argv[5]);

		double ransacReprojThreshold, similarityThreshold;
		if (argc >= 7) {
			ransacReprojThreshold = atof(argv[6]);
		} else {
			ransacReprojThreshold = 10.0;
		}
		if (argc >= 8) {
			similarityThreshold = atof(argv[7]);
		} else {
			similarityThreshold = 0.8;
		}

		string templateImgFilepath(
				imagesFolderpath + "/"
						+ StringUtils::parseImgFilename(templateFilename));
		// TODO Check that templateImgFilepath is a valid image filepath starting from the images folder as root

		string templateKeypointsFilepath(
				keysFolderpath + "/" + templateFilename);
		// TODO Check that templateFilename is a valid template keypoints filepath starting from the keypoints folder as root

		string sourceImgFilepath(
				imagesFolderpath + "/"
						+ StringUtils::parseImgFilename(sourceFilename));
		// TODO Check that sourceImgFilepath is a valid image filepath starting from the images folder as root

		string sourceKeypointsFilepath(keysFolderpath + "/" + sourceFilename);
		// TODO Check that templateFilename is a valid template keypoints filepath starting from the keypoints folder as root

		int result = geometricVerification(templateImgFilepath,
				templateKeypointsFilepath, sourceImgFilepath,
				sourceKeypointsFilepath, ransacReprojThreshold,
				similarityThreshold);

		return result == -1 ? EXIT_FAILURE : EXIT_SUCCESS;
	}

	return EXIT_SUCCESS;
}
