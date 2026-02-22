# sl_analysis_scripts
Scripts for speaker–listener (SL) neural coupling analysis using naturalistic fMRI paradigms.  

* Scripts using two analytical approaches to measuring coupling, along with preprocessing, subject list generation, and result visualization.  
    - **ISC-based (MATLAB + Bash)** — voxelwise inter-subject correlation   
    - **GLM-based (Python / nilearn)** — uses the speaker's preprocessed timeseries as a voxelwise regressor in each listener's first-level GLM, alongside motion regressors (using on nilearn's FirstLevelModel). Generates beta and t-stat maps per subject per cond (see isc_analysis_inputs for temporal offsets)  
* Both approaches are run across 21 temporal offset conditions (see `isc_analysis_inputs.txt`):
    * Listener_Precedes_Speaker_1–10 — listener brain leads speaker by 1–10 TRs
    * No_Shift — simultaneous
    * Speaker_Precedes_Listener_1–10 — speaker leads listener by 1–10 TRs



### scripts

```
01_remove_vols_from_start_and_end.sh                trim volumes for magnetization stabilization
01_sbatch_remove_vols_from_start_and_end.sh
01_submit_ISC_jobs.sh                               SLURM submission for ISC
02_run_isc_analysis_parallel.sh                     runs ISC via MATLAB
03_01_submit_collapsed_isc_analysis_parallel.sh
03_run_isc_analysis_collapsed_parallel.sh           collapsed ISC across stories
03_run_isc_analysis_collapsed_parallel_asd.sh
asd_speech_sherlock_expanded_timeshifts_tstats.py   GLM-based SL coupling (nilearn)
create_subject_lists_based_on_usability_SL.py       subject list generation by group/task
sbatch_time_shift.sh
submit_time_shift_sl_data_parallel.sh
time_shift_sl_data.m                                generates time-shifted data
create_height_thresholded_nifti_from_isc.m          thresholds ISC output maps
export_clust_size_of_nii_in_folder.m                cluster size extraction
check_numbers_of_niis.sh                            QC: checks expected file counts
isc_analysis_inputs.txt                             time-shift condition list
make_gifs.ipynb                                     result visualization
```

### Dependencies
**Python:**nilearn, nibabel, numpy, pandas, statsmodels, matplotlib, tqdm
**MATLAB:** SPM12, updated ISC toolbox
**HPC:** SLURM (Stanfords HPC computing cluster)'



### to note
Data paths are hardcoded to local/cluster paths and will need updating. Subject lists are generated from QC output and diagnosis mappings pre-generated via `create_subject_lists_based_on_usability_SL.py`.