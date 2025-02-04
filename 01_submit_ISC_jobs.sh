#!/bin/bash

#bash 01_submit_ISC_jobs.sh isc_analysis_inputs.txt 
mem=$1
thresh=$2
entire_analysis=$3
story_num=$4
story_suffix=$5

usage() {
    echo "Usage: bash 01_submit_ISC_jobs.sh <memory> <p_threshold> <run_analysis> <story_suffix>"
    echo "  <memory>        - Memory to assign to job (in GB)"
    echo "  <p_threshold>   - P-value threshold for statistical analysis"
    echo "  <run_analysis>  - Run full analysis (1) or only generate t-test results if ISC output exists (0)"
    echo "  <story_suffix>  - Story identifier (1,2,3,4,1_asd,2_asd,3_asd)"
    echo "Example: bash 01_submit_ISC_jobs.sh 32 0.01 1 4"
    exit 1
}

if [ "$#" -ne 5 ]; then
    usage
fi


file_with_isc_analysis_inputs='isc_analysis_inputs.txt'

for folder in `cat ${file_with_isc_analysis_inputs}`; do
        sl_story_folder=`echo sl_story${story_suffix}`;
        subfolder_path=`realpath sl_story${story_suffix}/processed_data/ts_output/${folder}`; 
        subfoldername=`echo $(basename $subfolder_path)`
        log_path=`realpath sl_story${story_suffix}/processed_data/ts_output/`
        # echo $subfolder_path;
        # echo $subfoldername;
        #bash 02_run_isc_analysis_parallel.sh /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain ${sl_story_folder} ${subfoldername};
        echo '#!/bin/bash' > tmp.sbatch;
        echo "bash 02_run_isc_analysis_parallel.sh /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain ${sl_story_folder} ${subfoldername} ${thresh} ${entire_analysis} ${story_suffix}" >> tmp.sbatch;
        sbatch -p owners,menon -c 16 --mem=${mem}G -o ${log_path}/ISC_${subfoldername}_log.txt tmp.sbatch;
        rm tmp.sbatch;
done
