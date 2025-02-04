01_remove_vols_from_start_and_end.sh#!/bin/bash

story_dir=$1

results_path='/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/'

output_dir=${results_path}/${story_dir}/processed_data/ts_output
listener_fmri_data_dir=${results_path}/${story_dir}/processed_data/listener_data
speaker_fmri_data_dir=${results_path}/${story_dir}/processed_data/speaker_data


ml math matlab;

cd ${results_path}
matlab -nodesktop -r "addpath(genpath(pwd));addpath(genpath('/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/time_shift'));time_shift_sl_data_parallel('${speaker_fmri_data_dir}','${listener_fmri_data_dir}','${output_dir}');exit();"
