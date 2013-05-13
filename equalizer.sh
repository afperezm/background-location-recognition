#!/bin/bash

if [ "$#" -ne "2" ]; then
echo "Usage: $0 <images.folder> <output.equalize.folder>"
exit 1
fi

for file in $1/*.jpg
do
filename=$(basename $file)
filename="${filename%.*}"
echo "Equalizing image [$file]"
/usr/bin/convert -equalize $file $2/$filename.jpg
done
