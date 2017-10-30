#!/bin/bash

version=$(echo -e "0.25 \n0.50 \n0.75 \n1.0")
size=$(echo -e "128\n160\n192\n224")

for v in ${version}; do
	for s in ${size}; do
		if [[ ${v} == "1.0" ]] && [[ ${s} == "160" ]]; then
			continue
		fi
		arch="mobilenet_${v}_${s}"
		output_dir="/tmp/MobileNet_output/${arch}"
		output_file="${output_dir}/accuracy.txt"
		mkdir -p ${output_dir}
		
		# echo ${output_dir}
		python retrain.py --image_dir /tmp/data --output_graph "${arch}.pb" --output_dir ${output_dir} --output_labels labels.txt --summaries_dir retrain_logs --flip_left_right --random_scale 30 --random_brightness 30 --architecture ${arch} >> ${output_file}
	done
done
