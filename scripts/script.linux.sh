#!/bin/bash
export IMAGES_FOLDER=$HOME/oxford_buildings_dataset/oxbuild_images/
export GROUND_TRUTH_FOLDER=$HOME/oxford_buildings_dataset/gt_files_170407/
export PATH=$HOME/workspace-opencv/VocabTree2/VocabLearn/:$HOME/workspace-opencv/VocabTree2/VocabBuildDB/:$HOME/workspace-opencv/VocabTree2/VocabMatch/:$HOME/workspace-opencv/oxford5k_recognition/Default/:$PATH

# Compute database features, key file are written to the same directory where the image are located
oxford5k_recognition -cf $IMAGES_FOLDER ./

# Compute images thumbnails
thumbnailer.sh $IMAGES_FOLDER ./

# Move query key and jpeg thumb files from ground truth folder to the queries folder
cat $GROUND_TRUTH_FOLDER/*query* | cut -d" " -f1 | cut -b6-256 | sort | uniq | xargs -I {} mv ./{}.key ./queries/
cat $HOME/oxford_buildings_dataset/gt_files_170407/*query* | cut -d" " -f1 | cut -b6-256 | sort | uniq | xargs -I {} mv ./{}.thumb.jpg ./queries/

# Move db key and jpeg thumb files from ground truth folder to the db folder
mv ./*.key ./db/
mv ./*.thumb.jpg ./db/

# Filter key files using Matlab generated masks and copying to new queries folder
mkdir queries_feature_selected
oxford5k_recognition -featsel ./queries/ ./queries_masks/ ./queries_feature_selected/

# Copy thumb images to the new queries folder
cp ./queries/*.jpg ./queries_feature_selected/

# Create list files of database and query keypoints
oxford5k_recognition -lists $GROUND_TRUTH_FOLDER ./

# Create list files of database and query keypoints for the feature selection dataset
cat list_queries.txt | cut -d"/" -f2 | xargs -I {} echo queries_feature_selected/{} > list_queries_featsel.txt 
cat list_gt.txt | cut -d"/" -f2 | xargs -I {} echo queries_feature_selected/{} > list_gt_featsel.txt

# Create list files of database and query keypoints with its ground truth landmark id
oxford5k_recognition -gt $GROUND_TRUTH_FOLDER ./

# Build vocabulary tree and match & score candidate images
VocabLearn list_db.txt 10 6 1 tree_10_6.out
VocabBuildDB list_db.txt tree_10_6.out tree_db_10_6.out
VocabMatch tree_db_10_6.out list_db_ld.txt list_queries.txt 50 matches.txt output.html ranked_candidates.txt

# Scoring candidate iamges using selected features
VocabMatch tree_db_10_6.out list_db_ld.txt list_queries_featsel.txt 50 matches_featsel.txt output_featsel.html ranked_candidates_featsel.txt

# Re-rank list of candidates using geometric criterion
oxford5k_recognition -gvc $IMAGES_FOLDER ./ ranked_candidates.txt geom_ranked_candidates.txt geom_ranked_candidates_inliers.txt

# Pre vs Post
oxford5k_recognition -perf list_gt.txt list_db_ld.txt ranked_candidates.txt occurrence_matrix_pregv.txt voted_landmarks_pregv.txt
oxford5k_recognition -perf list_gt.txt list_db_ld.txt geom_ranked_candidates.txt occurrence_matrix_postgv.txt voted_landmarks_postgv.txt

# Variation of geometric verification parameters
#oxford5k_recognition -perf list_gt.txt list_db_ld.txt geom_ranked_candidates_3_100_0.5.txt occurrence_matrix_3_100_0.5.txt voted_landmarks_3_100_0.5.txt
oxford5k_recognition -gvc $IMAGES_FOLDER ./ ranked_candidates.txt geom_ranked_candidates_3_auto_0.5.txt geom_ranked_candidates_3_auto_0.5_inliers.txt 3 0.5

# Generate matrices of voting and candidate occurrences for performance evaluation both for pre and post geometric verification
oxford5k_recognition -perf list_gt_featsel.txt list_db_ld.txt ranked_candidates_featsel.txt occurrence_matrix_featsel.txt voted_landmarks_featsel.txt

