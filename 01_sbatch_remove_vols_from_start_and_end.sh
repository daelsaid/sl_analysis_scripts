#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: bash $0 <data_path> <script_dir>"
    echo ""
    echo "This script submits SLURM batch jobs to remove volumes from fMRI NIfTI files."
    echo ""
    echo "Arguments:"
    echo "  <data_path>    Path to the directory containing the NIfTI files."
    echo "  <script_dir>   Path to the directory containing '01_remove_vols_from_start_and_end.sh'."
    echo ""
    echo "Example:"
    echo "  bash $0 /scratch/users/daelsaid/post_processed_wholebrain /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain"
    echo ""
    echo "This script:"
    echo "  - Searches for all '*filtered.nii' files in the given data directory."
    echo "  - Submits SLURM batch jobs for each file."
    echo "  - Calls '01_remove_vols_from_start_and_end.sh' from the specified script directory."
    echo ""
    echo "Ensure that:"
    echo "  - <data_path> exists and contains the necessary NIfTI files."
    echo "  - <script_dir> contains '01_remove_vols_from_start_and_end.sh'."
    echo ""
    exit 1
}

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    usage
fi

data_path=$1
script_dir=$2

cd ${data_path}
echo $script_dir;
#for nii in `find . -name '*filtered.nii' -type f`; do
for nii in `cat pth_to_data.txt`; do 
    echo $nii
    prefix=`echo $(basename ${nii}) | cut -d. -f1`;
    echo "running 01_remove_vols_from_start_and_end.sh on ${nii}";
    echo '#!/bin/bash' > remove_vols.sbatch;
    echo "bash ${script_dir}/01_remove_vols_from_start_and_end.sh $nii ${data_path}" >> remove_vols.sbatch;
    #echo "sbatch -p owners,menon -c 4 --mem=16G -o ${data_path}/01_remove_vols_from_start_and_end_log_${prefix}.txt remove_vols.sbatch"; 
    sbatch -p owners,menon -c 4 --mem=16G -o ${data_path}/01_remove_vols_from_start_and_end_log_${prefix}.txt remove_vols.sbatch; 
    rm remove_vols.sbatch;
done
