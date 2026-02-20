#!/bin/bash


group=$1
condition=$2
parent_dir=$3

usage() {
    echo "Usage: bash sbatch_time_shift.sh <data_folder> <memory> <queue>"
    echo "  <group>  - ASD or TD (parent dir)"
	echo "  <condition>  - story dir (story1/2/3/4"
    echo "  <parent_dir>  - parent folder containing group/condition data"
    echo "Example: bash sbatch_time_shift.sh asd story1 /scratch/users/daelsaid/updated_results"
    exit 1
}

# chck # arguments are provided
if [ "$#" -ne 3 ]; then
    usage
fi


results_path=${parent_dir}/${group}/${condition}

output_dir=${results_path}/time-shift
listener_fmri_data_dir=${results_path}
speaker_fmri_data_dir=${results_path}/speaker_data

ml math matlab;

cd ${results_path}
matlab -nodesktop -r "addpath(genpath(pwd));addpath(genpath('/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/time_shift'));time_shift_sl_data_parallel('${speaker_fmri_data_dir}','${listener_fmri_data_dir}','${output_dir}');exit();"
