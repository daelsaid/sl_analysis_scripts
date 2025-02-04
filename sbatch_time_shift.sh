#!/bin/bash


story_dir=$1
mem=$2
queue=$3

usage() {
    echo "Usage: bash sbatch_time_shift.sh <data_folder> <memory> <queue>"
    echo "  <data_folder>  - Name of the folder containing data to time shift"
    echo "  <memory>       - Memory to assign to the job (just numbers in GB)"
    echo "  <queue>        - Queue to send the job to (-p argument)"
    echo "Example: bash sbatch_time_shift.sh sl_story2_asd 64 owners,menon"
    exit 1
}

# chck # arguments are provided
if [ "$#" -ne 3 ]; then
    usage
fi


echo '#!/bin/bash' > tmp.sbatch; echo "bash submit_time_shift_sl_data_parallel.sh ${story_dir}" >> tmp.sbatch; sbatch -p ${queue} -c 16 --mem=${mem}G tmp.sbatch; rm tmp.sbatch
