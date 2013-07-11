#!/bin/bash
export IMAGES_FOLDER=$HOME/oxford_buildings_dataset/oxbuild_images/
export GROUND_TRUTH_FOLDER=$HOME/oxford_buildings_dataset/gt_files_170407/
export DATASET_ROOT=$HOME/workspace-opencv/oxford5k/
export PATH=$HOME/workspace-opencv/VocabTree2/VocabLearn/:$HOME/workspace-opencv/VocabTree2/VocabBuildDB/:$HOME/workspace-opencv/VocabTree2/VocabMatch/:$HOME/workspace-opencv/LocationRecognition/FeatureExtractSelect/:$HOME/workspace-opencv/LocationRecognition/GeomVerify:$HOME/workspace-opencv/LocationRecognition/ListBuild/:$HOME/workspace-opencv/LocationRecognition/PerfEval/:$PATH

# Compute database features, key file are written to the same directory where the image are located
FeatureExtractSelect -cf $IMAGES_FOLDER $DATASET_ROOT

# Compute images thumbnails
thumbnailer.sh $IMAGES_FOLDER $DATASET_ROOT

# Move query key and jpeg thumb files from ground truth folder to the queries folder
mkdir $DATASET_ROOT/queries
cat $GROUND_TRUTH_FOLDER/*query* | cut -d" " -f1 | cut -b6-256 | sort | uniq | xargs -I {} mv $DATASET_ROOT/{}.key $DATASET_ROOT/queries/
cat $HOME/oxford_buildings_dataset/gt_files_170407/*query* | cut -d" " -f1 | cut -b6-256 | sort | uniq | xargs -I {} mv $DATASET_ROOT/{}.thumb.jpg $DATASET_ROOT/queries/

# Move db key and jpeg thumb files from ground truth folder to the db folder
mv $DATASET_ROOT/db
mv $DATASET_ROOT/*.key $DATASET_ROOT/db/
mv $DATASET_ROOT/*.thumb.jpg $DATASET_ROOT/db/

# Computing masks for query images
octave --silent --eval "addpath('$DATASET_ROOT/../finding-long-straight-lines/');addpath('$DATASET_ROOT/../vanishing-points/');addpath('$DATASET_ROOT/../LocationRecognition/scripts');masker('$IMAGES_FOLDER')"
mv $IMAGES_FOLDER/*.mask $DATASET_ROOT/queries_masks/

# Filter key files using Matlab generated masks and copying to new queries folder
mkdir $DATASET_ROOT/queries_feature_selected
FeatureExtractSelect -featsel $DATASET_ROOT/queries/ $DATASET_ROOT/queries_masks/ $DATASET_ROOT/queries_feature_selected/

# Copy thumb images to the new queries folder
cp $DATASET_ROOT/queries/*.jpg $DATASET_ROOT/queries_feature_selected/

# Create list files of database and query keypoints
ListBuild -lists $GROUND_TRUTH_FOLDER $DATASET_ROOT
# Create list files of database and query keypoints with its ground truth landmark id
ListBuild -gt $GROUND_TRUTH_FOLDER $DATASET_ROOT

# Create list files of database and query keypoints for the feature selection dataset
cat list_queries.txt | cut -d"/" -f2 | xargs -I {} echo queries_feature_selected/{} > list_queries_featsel.txt 
cat list_gt.txt | cut -d"/" -f2 | xargs -I {} echo queries_feature_selected/{} > list_gt_featsel.txt

# Build vocabulary tree and db
VocabLearn list_db.txt 10 6 1 tree_10_6.out
VocabBuildDB list_db.txt tree_10_6.out tree_db_10_6.out

# Match & score candidate images
VocabMatch tree_db_10_6.out list_db_ld.txt list_queries.txt 50 matches.txt output.html ranked_candidates.txt
# Match & score candidate images using selected features
VocabMatch tree_db_10_6.out list_db_ld.txt list_queries_featsel.txt 50 matches_featsel.txt output_featsel.html ranked_candidates_featsel.txt

# Re-rank list of candidates using geometric criterion
GeomVerify -gvc $IMAGES_FOLDER $DATASET_ROOT ranked_candidates.txt geom_ranked_candidates.txt geom_ranked_candidates_inliers.txt
# Variation of geometric verification parameters
GeomVerify -gvc $IMAGES_FOLDER $DATASET_ROOT ranked_candidates.txt geom_ranked_candidates_3_auto_0.5.txt geom_ranked_candidates_3_auto_0.5_inliers.txt 3 0.5

# Generate matrices of voting and candidate occurrences for performance evaluation
PerfEval list_gt.txt list_db_ld.txt ranked_candidates.txt occurrence_matrix_pregv.txt voted_landmarks_pregv.txt
PerfEval list_gt.txt list_db_ld.txt geom_ranked_candidates.txt occurrence_matrix_postgv.txt voted_landmarks_postgv.txt
#PerfEval list_gt.txt list_db_ld.txt geom_ranked_candidates_3_100_0.5.txt occurrence_matrix_3_100_0.5.txt voted_landmarks_3_100_0.5.txt
PerfEval list_gt.txt list_db_ld.txt geom_ranked_candidates_3_auto_0.5.txt occurrence_matrix_3_auto_0.5.txt voted_landmarks_3_auto_0.5.txt
PerfEval list_gt_featsel.txt list_db_ld.txt ranked_candidates_featsel.txt occurrence_matrix_featsel.txt voted_landmarks_featsel.txt

