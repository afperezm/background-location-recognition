#!/bin/bash

if [ "$#" -ne "2" ]; then
	echo -e "\nUsage:\n\t$0 <images.folder> <output.thumbs.folder>\n"
	exit 1
fi

for file in $1/*.jpg
do
	filename=$(basename $file)
	filename="${filename%.*}"
	echo "Creating thumbnail for file [$file]"
	/usr/bin/convert -thumbnail 200 $file $2/$filename.thumb.jpg
done

