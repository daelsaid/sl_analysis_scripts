#!/bin/bash


#bash 03_01_submit_collaped_isc_analysis_parallel.sh 32 0.01 0 sl_collapsed 

mem=$1
thresh=$2
entire_analysis=$3
t_test_ouput_foldername=$4 
asd_or_td=$5
story_folder_suffix=$6

usage() {
    echo "Usage: bash 03_01_submit_collaped_isc_analysis_parallel.sh <memory> <p_threshold> <run_analysis> <output_folder> <group>"
    echo "  <memory>        - Memory to assign to job (in GB)"
    echo "  <p_threshold>   - P-value threshold for statistical analysis"
    echo "  <run_analysis>  - Run full analysis (1) or only generate t-test results if ISC output exists (0)"
    echo "  <output_folder> - Folder where results will be written"
    echo "  <group>         - Group label (asd or td)"
    echo "  <story_folder_suffix> - if there is an ending like _hpf_trimmed, add it with _ if not, leave blank"
    echo "Example: bash 03_01_submit_collaped_isc_analysis_parallel.sh 64 0.01 0 sl_collapsed_asd asd _hpf_trimmed"
    exit 1
}

if [ "$#" -ne 6 ]; then
    usage
fi

file_with_isc_analysis_inputs='isc_analysis_inputs.txt'

for folder in `cat ${file_with_isc_analysis_inputs}`; do
    # sl_story_folder=`echo sl_story${folder}`;
    # subfolder_path=`realpath sl_story${folder}/processed_data/ts_output/${folder}`;
    subfoldername=`echo ${folder}`
    echo $subfoldername;
    log_path=`realpath ${t_test_ouput_foldername}/processed_data/ts_output/`
    # echo $subfolder_path;
    # echo $subfoldername;
    if [ "$asd_or_td" == "td" ]; then
	echo "bash 03_run_isc_analysis_collapsed_parallel.sh /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain sl_story ${subfoldername} ${thresh} ${entire_analysis} ${t_test_ouput_foldername} ${story_folder_suffix}";
    	echo '#!/bin/bash' > tmp.sbatch;
    	echo "bash 03_run_isc_analysis_collapsed_parallel.sh /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain sl_story ${subfoldername} ${thresh} ${entire_analysis} ${t_test_ouput_foldername} ${story_folder_suffix}" >> tmp.sbatch;
    	sbatch -p owners,menon -c 16 --mem=${mem}G -o ${log_path}/ISC_collapsed_${subfoldername}_log.txt tmp.sbatch;
    	rm tmp.sbatch;
    elif [ "$asd_or_td" == "asd" ]; then
        echo "bash 03_run_isc_analysis_collapsed_parallel_asd.sh /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain sl_story ${subfoldername} ${thresh} ${entire_analysis} ${t_test_ouput_foldername} ${story_folder_suffix}";
        echo '#!/bin/bash' > tmp.sbatch;
        echo "bash 03_run_isc_analysis_collapsed_parallel_asd.sh /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain sl_story ${subfoldername} ${thresh} ${entire_analysis} ${t_test_ouput_foldername} ${story_folder_suffix}" >> tmp.sbatch;
        sbatch -p owners,menon -c 16 --mem=${mem}G -o ${log_path}/ISC_collapsed_${subfoldername}_log.txt tmp.sbatch;
        rm tmp.sbatch;
    else
    	echo "Error: Invalid value for asd_or_td. Use 'td' or 'asd'."
    	exit 1
    fi
done
