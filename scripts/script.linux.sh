#!/bin/bash
images_folder=~/oxford_buildings_dataset/oxbuild_images/
ground_truth_folder=~/oxford_buildings_dataset/gt_files_170407/

export PATH=/home/andresf/workspace-opencv/VocabTree2/VocabLearn/:/home/andresf/workspace-opencv/VocabTree2/VocabBuildDB/:/home/andresf/workspace-opencv/VocabTree2/VocabMatch/:/home/andresf/workspace-opencv/oxford5k_recognition/Default/:$PATH

# Compute database features, key file are written to the same directory where the image are located
oxford5k_recognition -cf ~/Documents/POLIMI_maestria/Computer_Vision/Project/oxbuild_images/
mv ~/Documents/POLIMI_maestria/Computer_Vision/Project/oxbuild_images/*.key ./

# Compute images thumbnails
thumbnailer.sh $images_folder ./

# Move query key and jpeg thumb files from ground truth folder to the queries folder
cat $ground_truth_folder/*query* | cut -d" " -f1 | cut -b6-256 | sort | uniq | xargs -I {} mv $images_folder/{}.key ./queries/
cat ~/oxford_buildings_dataset/gt_files_170407/*query* | cut -d" " -f1 | cut -b6-256 | sort | uniq | xargs -I {} mv ./{}.thumb.jpg ./queries/

# Move db key and jpeg thumb files from ground truth folder to the db folder
mv $images_folder/*.key ./db/
mv ./*.thumb.jpg ./db/

# Create list files of database and query keypoints
#cat "$ground_truth_folder"*1_ok* "$ground_truth_folder"*1_good* "$ground_truth_folder"*1_junk* | sort | uniq | xargs -I {} echo db/{}.key > ./list_db.txt
#cat "$ground_truth_folder"*query* | cut -d" " -f1 | cut -b6-256 | sort | uniq | xargs -I {} echo queries/{}.key > ./list_queries.txt
oxford5k_recognition -lists $ground_truth_folder ./

# Create list files of database and query keypoints with its ground truth landmark id
oxford5k_recognition -gt $ground_truth_folder ./

# Build vocabulary tree and match & score candidate images
VocabLearn list_db.txt 10 6 1 tree_10_6.out
VocabBuildDB list_db.txt tree_10_6.out tree_db_10_6.out
VocabMatch tree_db_10_6.out list_db_ld.txt list_queries.txt 50 matches.txt output.html

# Generate matrix of candidate occurrences and voting matrix for performance evaluation
oxford5k_recognition -perf list_gt.txt list_db_ld.txt candidates.txt
