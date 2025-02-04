#!/bin/bash

parent_dir_processed_data=$1
story_folder=$2
subfolder_name=$3
threshold=$4
run_entire_analysis=$5


path_to_ts_dir=${parent_dir_processed_data}/${story_folder}/processed_data/ts_output #input path top
isc_output_dir=${parent_dir_processed_data}/${story_folder}/processed_data/ISC_output

ml math matlab;

echo "running ${subfolder_name} for ${story_folder}";
echo "path to ts dir: ${path_to_ts_dir}"
echo "path to isc_output_dir: ${isc_output_dir}"

cd ${parent_dir_processed_data};
matlab -nodesktop -r "addpath(genpath(pwd));addpath(genpath('/oak/stanford/groups/menon/projects/tongshan/2024_SL_TS/scripts/taskfmri/ISC_analysis/')); runISC_SL_folder('${path_to_ts_dir}/','${subfolder_name}/','${isc_output_dir}/',${threshold},${run_entire_analysis});exit();"
