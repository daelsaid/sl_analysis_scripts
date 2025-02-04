#!/bin/bash

results_path=$1
story_folder_prefix=$2
subfolder_name=$3
threshold=$4
run_entire_analysis=$5
t_test_output_foldername=$6
story_folder_suffix=$7

ml math matlab;

cd ${results_path};

story1_ts_output_path=${results_path}/${story_folder_prefix}1_asd${story_folder_suffix}/processed_data/ts_output
story2_ts_output_path=${results_path}/${story_folder_prefix}2_asd${story_folder_suffix}/processed_data/ts_output
story3_ts_output_path=${results_path}/${story_folder_prefix}3_asd${story_folder_suffix}/processed_data/ts_output
#story4_ts_output_path=${results_path}/${story_folder_prefix}4/processed_data/ts_output

story1_isc_output_path=${results_path}/${story_folder_prefix}1_asd${story_folder_suffix}/processed_data/ISC_output
story2_isc_output_path=${results_path}/${story_folder_prefix}2_asd${story_folder_suffix}/processed_data/ISC_output
story3_isc_output_path=${results_path}/${story_folder_prefix}3_asd${story_folder_suffix}/processed_data/ISC_output
#story4_isc_output_path=${results_path}/${story_folder_prefix}4/processed_data/ISC_output

t_test_outputpath=${results_path}/${t_test_output_foldername}/${subfolder_name}

echo "results_path: ${results_path}, running ${subfolder_name} for ${t_test_output_foldername}";
echo "path to ts dir: ${results_path}"
echo "path to t_test results: ${t_test_outputpath},${results_path}/${t_test_output_foldername}/${subfolder_name}"

cd ${results_path};

matlab -nodesktop -r "addpath(genpath(pwd));addpath(genpath('/oak/stanford/groups/menon/projects/tongshan/2024_SL_TS/scripts/taskfmri/ISC_analysis/'));  input_path_top_list = {'${story1_ts_output_path}', '${story2_ts_output_path}','${story3_ts_output_path}'};output_path_top_list =  {'${story1_isc_output_path}','${story2_isc_output_path}','${story3_isc_output_path}'}; runISC_SL_folder_collapse(input_path_top_list, '${subfolder_name}', output_path_top_list,'${t_test_outputpath}',${threshold},${run_entire_analysis});exit();"
