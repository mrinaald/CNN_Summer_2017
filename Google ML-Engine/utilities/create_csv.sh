#!/bin/bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

for i in $(gsutil ls gs://dslab/mrinaal/data/HikeImages/v5/train); do
	tag=$(echo $i | rev | cut -d/ -f2 | rev)
	for j in $(gsutil ls $i); do
		echo "${j},${tag}" >> ~/Documents/train_set.csv
	done
done
echo "train_set.csv Done"

for i in $(gsutil ls gs://dslab/mrinaal/data/HikeImages/v5/validate); do
	tag=$(echo $i | rev | cut -d/ -f2 | rev)
	for j in $(gsutil ls $i); do
		echo "${j},${tag}" >> ~/Documents/eval_set.csv
	done
done
echo "eval_set.csv Done"

IFS=$SAVEIFS
