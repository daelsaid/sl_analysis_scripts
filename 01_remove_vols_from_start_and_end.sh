 #!/bin/bash

nii=$1
path_to_data=$2

# Function to display usage
usage() {
    echo "Usage: bash $0 <nii_file> <path_to_data>"
    echo ""
    echo "This script removes the first 10 and last volumes from a given fMRI NIfTI file."
    echo ""
    echo "Arguments:"
    echo "  <nii_file>      Path to the NIfTI file to process."
    echo "  <path_to_data>  Path to the directory containing the NIfTI file."
    echo ""
    echo "Example:"
    echo "  bash $0 /scratch/users/daelsaid/post_processed_wholebrain/subject1/filtered.nii /scratch/users/daelsaid/post_processed_wholebrain"
    echo ""
    exit 1
}

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    usage
fi


cd ${path_to_data}

ml biology fsl; 
ml biology freesurfer;


echo $nii; 
story=`echo $nii | cut -d/ -f2`; 
echo $story;

source_dir="${story}/processed_data"
rel_path=`echo $nii | cut -d/ -f3-4`;
hpf_trimmed="${story}_hpf_trimmed/${rel_path}"
bpf_trimmed="${story}_bpf_trimmed/${rel_path}"
# echo $hpf_trimmed $bpf_trimmed;

# 
mkdir -p "${hpf_trimmed}"
mkdir -p "${bpf_trimmed}"

nframes=`mri_info ${nii} | grep nframes | cut -d':' -f2 | sed 's/^ //' `; 
keep_frames=$((nframes-10-3));
prefix=`echo $(basename ${nii}) | cut -d. -f1`;
new_file_name=`echo ${prefix}_trimmed.nii.gz`;

echo $prefix ${new_file_name}

echo "removing vols from ${nii}"
echo "original vol #: ${nframes}, removing first 10 until ${keep_frames}";

# process hpfiltered data
if [[ $nii == *_hpfiltered.nii ]]; then
    echo "HPF file: ${nii}"
    # echo "fslroi ${nii} ${hpf_trimmed}/${prefix}_trimmed 10 $keep_frames";
    fslroi ${nii} ${hpf_trimmed}/${prefix}_trimmed 10 $keep_frames;
    nframes_new=`mri_info ${hpf_trimmed}/${prefix}_trimmed.nii.gz | grep nframes | cut -d':' -f2 | sed 's/^ //' `; 
    echo "New file:${hpf_trimmed}/${prefix}_trimmed.nii.gz (Number of frames:${nframes_new}" 
    gunzip ${hpf_trimmed}/${prefix}_trimmed.nii.gz;
# process bpfiltered data
elif [[ $nii == *_filtered.nii ]]; then
    echo "BPF file: ${nii}"
    # echo "fslroi ${nii} ${bpf_trimmed}/${prefix}_trimmed 10 $keep_frames";
    fslroi ${nii} ${bpf_trimmed}/${prefix}_trimmed 10 $keep_frames;
    nframes_new=`mri_info ${bpf_trimmed}/${prefix}_trimmed.nii.gz | grep nframes | cut -d':' -f2 | sed 's/^ //' `; 
    echo "New file:${bpf_trimmed}/${prefix}_trimmed.nii.gz (Number of frames:${nframes_new}" 
    gunzip ${bpf_trimmed}/${prefix}_trimmed.nii.gz;
fi


