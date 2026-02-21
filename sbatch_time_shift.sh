#!/bin/bash

group=$1
condition=$2 
parent_dir=$3
mem=$4
queue=$5

usage() {
    echo "Usage: bash sbatch_time_shift.sh <data_folder> <memory> <queue>"
    echo "  <group>  - ASD or TD (parent dir)"
	echo "  <condition>  - story dir (story1/2/3/4)"
    echo "  <parent_dir>  - parent folder containing group/condition data"
    echo "  <memory>       - Memory to assign to the job (just numbers in GB)"
    echo "  <queue>        - Queue to send the job to (-p argument)"
    echo "Example: bash sbatch_time_shift.sh asd story1 /scratch/users/daelsaid/updated_results 64 owners,menon"
    exit 1
}

# chck # arguments are provided
if [ "$#" -ne 5 ]; then
    usage
fi


echo '#!/bin/bash' > timeshift.sbatch; 
echo "bash /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/scripts_and_subjlists/submit_time_shift_sl_data_parallel.sh ${group} ${condition} ${parent_dir}" >> timeshift.sbatch; sbatch -p ${queue} -c 16 --mem=${mem}G -o ${parent_dir}/${group}/${condition}/time-shift/time-shift_${group}_${condition}_%j.log -e ${parent_dir}/${group}/${condition}/time-shift/time-shift_${group}_${condition}_%j.err timeshift.sbatch; rm timeshift.sbatch