#!/usr/bin/env python
# coding: utf-8

# In[1]:

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    import matplotlib.pyplot as plt
    plt.ion()  # Use interactive mode for standalone scripts

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

def install(name):
    import subprocess
    subprocess.call(['pip', 'install',name])
    
install('spectral')

import spectral



import sys
sys.path.append('/Users/daelsaid/tools/miniconda3/env/lib/python3.7/site-packages')

from glob import glob
from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix
)
from nilearn.image import load_img, math_img
# from nilearn.plotting import find_xyz_cut_coords
from nilearn.plotting import plot_epi,plot_roi,plot_connectome,plot_glass_brain,plot_img,plot_stat_map,plot_prob_atlas,plot_design_matrix
import numpy as np
# from nilearn.glm.first_level import FirstLevelModel
from IPython.display import Image
import nibabel as nib
import tqdm


# #listener data;coords
# #speaker time series coordinate --> same coordinate in listener + regressor of speaker for that coordinate
# 
# #add_reg
# 
# #Additional predictors, like subject motion, can be specified using the add_reg parameter. Look at the function definition for available arguments.
# 
# contrasts =1,0
# movement regressors = 6
# 7 total regressors 
# 
# driftmodel=polynomial 
# for x in y in z pythonapi An instance of ()
# 
# 
# get mni152 brain mask --> 1 = brain 0 hen its not 
# make a list of coordinates where brain mask = 0 and iterate through that list 

# loaded the preprocessed story for speaker

# In[ ]:


import argparse
parser = argparse.ArgumentParser(description='Description of your program')
# Initialize parser

# Adding arguments
parser.add_argument('-l', '--listener_prefix', type=str)
# parser.add_argument('-o_path','-output_path',type=str)
parser.add_argument('-l_story_runname', '--listener_story_runname', type=str)
parser.add_argument('-story_num', '--story_number', type=int)
parser.add_argument('-nii_prefix', type=str)
parser.add_argument('-nii_story', type=str)
parser.add_argument('-listener_story_last_frame',type=int)
parser.add_argument('-l_last_framenum',type=int)
parser.add_argument('-s_n_scan_length',type=int)

parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')

# Parse arguments
args = parser.parse_args()

listener = args.listener_prefix
listener_story_runname = args.listener_story_runname
subject_file_name_prefix=listener+'_'+listener_story_runname
story = args.story_number
nii_prefix = args.nii_prefix
nii_story = args.nii_story
verbose_mode = args.verbose
shortened_listener_story_last_frame=args.l_last_framenum
n_scans=args.s_n_scan_length

    # speaker_story_varname=row['speaker_story_varname']


# In[2]:


preprocessed_path='/Volumes/de_encryptd/asd_sl_analysis/swar'
# output_file_path=args.output_path
output_file_path = '/Volumes/de_encryptd/asd_sl_analysis/results/individualstats/hrf_model/beta_and_tvals/updated_time_intervals'

if story == 1:
    sl_run2_story1_12016_1_4_swar=load_img(os.path.join(preprocessed_path,'sl_run2/12016_visit1_session4_fmri_sl_run2_swar_spm12_swarI.nii.gz')) #290 frames
    s_data_temp=sl_run2_story1_12016_1_4_swar

if story == 2:
    sl_run1_story2_12008_1_1_swar=load_img(os.path.join(preprocessed_path,'sl_run1/12008_visit1_session1_fmri_sl_run1_swar_spm12_swarI.nii.gz')) 
    s_data_temp=sl_run1_story2_12008_1_1_swar
if story == 3: 
    sl_run1_story3_12015_1_2_swar=load_img(os.path.join(preprocessed_path,'sl_run1/12015_visit1_session2_fmri_sl_run1_swar_spm12_swarI.nii.gz')) #267 frames
    s_data_temp=sl_run1_story3_12015_1_2_swar
if story == 4:
    sl_run2_story4_12008_1_1_swar=load_img(os.path.join(preprocessed_path,'sl_run2/12008_visit1_session1_fmri_sl_run2_swar_spm12_swarI.nii.gz')) #366 frames
    s_data_temp=sl_run2_story4_12008_1_1_swar

# In[ ]:


# # sl_run1_story2_12008_1_1_swgcaor=load_img('/Users/daelsaid/scratch/12008_1_1/sl_run1/swgcaor_spm12/swgcaorI.nii.gz')
# sl_run1_story2_12008_1_1_swarr=load_img('/Volumes/de_encryptd/asd_sl_analysis//Users/daelsaid/scratch/12008_1_1/sl_run1/swgcaor_spm12/swgcaorI.nii.gz')

# sl_listener_11036_v2_session1_story2_swgcaor=load_img('/Volumes/de_encryptd/asd_sl_analysis/11036/visit2/session1/fmri/story2_3_r5/swgcaor_spm12/swgcaorI.nii.gz')

# sl_listener_11012_v2_session1_story2_swgcaor=load_img('/Volumes/de_encryptd/asd_sl_analysis/data/11012_visit2_session1_fmri_story2_3_r1_unnormalized_I.nii.gz')

# sl_listener_12505_v1_session1_story2_swar=load_img('/Volumes/de_encryptd/asd_sl_analysis/12505/visit1/session1/fmri/story2_3_r4/swar_spm12/swarI.nii.gz')


# Load MNI Template brain mask and get coordinates of locations where vals !=0
# iterate through the coordinates from mni brain mask above and use NiftiSpheresmasker to extract time series. list has time series for each voxel coordinate to be used as regressors

# In[14]:

mni_2mm_brain_path='/Users/daelsaid/tools/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz'

mni_2mm_brain_binarized_img = load_img(mni_2mm_brain_path)

#numpy array
mni_2mm_brainmask_data = mni_2mm_brain_binarized_img.get_fdata()
mni_2mm_brainmask_non_zero_coords = np.argwhere(mni_2mm_brainmask_data != 0)
print(len(mni_2mm_brainmask_non_zero_coords))
#mni_2mm_brainmask_non_zero_coords= 2D numpy array


# ### story2

# ###### functions that load listener nifti and load listener movement stats

# n_scans_story1_speaker=290  
# n_scans_story1_listener=290  
# n_scans_story2_speaker=289  
# n_scans_story2_listener=250  
# n_scans_story3_speaker=267  
# n_scans_story3__listener=257  
# n_scans_story4_speaker  
# n_scans_story3__listener  

# In[4]:


# sl_listener_11036_v2_session1_story2_swar=load_img(os.path.join(preprocessed_path,'story2/11036_visit2_session1_fmri_story2_3_r5_swar_spm12_swarI.nii.gz'))
print(preprocessed_path,f'story{story}',f'{subject_file_name_prefix}_swar_spm12_swarI.nii.gz')
def load_listener_img_swar(subject_file_name_prefix, story):
    file_path = os.path.join(preprocessed_path,f'story{story}',
                             f'{subject_file_name_prefix}_swar_spm12_swarI.nii.gz')
    fmri_image = load_img(file_path)
    return fmri_image

def load_listener_movement_swar_rp_I(subject_file_name_prefix, story):
    file_path = os.path.join(preprocessed_path, 'rp_i',
                             f'{subject_file_name_prefix}_swar_spm12_rp_I.txt')
    rp_I = np.loadtxt(file_path)
    return rp_I


# ####### read subjectlist that has listener_prefix,story_run_name

# In[117]:


# story2_slist=pd.read_csv('/Users/daelsaid/Downloads/sl_analysis_story_2_list.csv',dtype=str)


# In[6]:


# listener='11051_visit2_session1_fmri'
# listener_story_runname='story2_3_r1'
# subject_file_name_prefix=listener+'_'+listener_story_runname
# story='2'



# ##### create empty lists to store beta values/parameter estimates
# 

##speaker


betas_con001=[]
betas_con002=[]
betas_con003=[]
betas_con004=[]
betas_con005=[]
betas_con006=[]
betas_con007=[]
betas_con008=[]
betas_con009=[]
betas_con010=[]
betas_con011=[]
betas_con012=[]
betas_con013=[]
betas_con014=[]
betas_con015=[]

tvalues_con001=[]
tvalues_con002=[]
tvalues_con003=[]
tvalues_con004=[]
tvalues_con005=[]
tvalues_con006=[]
tvalues_con007=[]
tvalues_con008=[]
tvalues_con009=[]
tvalues_con010=[]
tvalues_con011=[]
tvalues_con012=[]
tvalues_con013=[]
tvalues_con014=[]
tvalues_con015=[]

### listener
betas_con0001=[]
betas_con0002=[]
betas_con0003=[]
betas_con0004=[]
betas_con0005=[]
betas_con0006=[]
betas_con0007=[]
betas_con0008=[]
betas_con0009=[]
betas_con0010=[]
betas_con0011=[]
betas_con0012=[]
betas_con0013=[]
betas_con0014=[]
betas_con0015=[]

tvalues=[]
tvalues_con0001=[]
tvalues_con0002=[]
tvalues_con0003=[]
tvalues_con0004=[]
tvalues_con0005=[]
tvalues_con0006=[]
tvalues_con0007=[]
tvalues_con0008=[]
tvalues_con0009=[]
tvalues_con0010=[]
tvalues_con0011=[]
tvalues_con0012=[]
tvalues_con0013=[]
tvalues_con0014=[]
tvalues_con0015=[]

# In[7]:

betas=[]
betas_con1=[]
betas_con2=[]
betas_con3=[]
betas_con4=[]
betas_con5=[]
betas_con6=[]
betas_con7=[]
betas_con8=[]
betas_con9=[]
betas_con10=[]
betas_con11=[]
betas_con12=[]

tvalues=[]
tvalues_con1=[]
tvalues_con2=[]
tvalues_con3=[]
tvalues_con4=[]
tvalues_con5=[]
tvalues_con6=[]
tvalues_con7=[]
tvalues_con8=[]
tvalues_con9=[]
tvalues_con10=[]
tvalues_con11=[]
tvalues_con12=[]

# In[126]:


# 11012,2,1
# 11036,2,1
# 11036,2,2
# 11051,2,1
# 11051,2,2
# 12501,1,1
# 12501,1,2
# 12502,1,1
# 12503,1,1
# 12504,1,1
# 12505,1,1
# 12505,1,2
# 12506,1,1
# 12506,1,2
# 12511,1,1
# 9317,4,1
# 9317,4,2
# 9409,4,1
# 9409,4,2
# 9488,2,1


# ###### load speaker data, listere data (endog_l) and listener motion params (rp_i_l)

# In[9]:

sl_listener_11051_v2_session1_story2_swar=load_listener_img_swar(subject_file_name_prefix, story)
endog_l=sl_listener_11051_v2_session1_story2_swar.get_fdata()
rp_i_l=load_listener_movement_swar_rp_I(subject_file_name_prefix, story)
s_data=s_data_temp.get_fdata()


# ###### set TR and n_scans 

# In[10]:

t_r=.8
# n_scans=289
# shortened_listener_story_last_frame=250


# In[11]:


l_data=endog_l
# motionparameters_s=np.loadtxt('/Users/daelsaid/scratch/12008_1_1/sl_run1/swgcaor_spm12/rp_I.txt')
motionparameters_l=rp_i_l

frametimes = np.linspace(0, (n_scans - 1) * t_r, n_scans)
print(n_scans,shortened_listener_story_last_frame)
for coord in tqdm.tqdm(mni_2mm_brainmask_non_zero_coords):
    s_voxel_time_series = s_data[coord[0], coord[1], coord[2], :]
    s_seed_time_series = s_voxel_time_series.reshape(-1, 1) # (n_scans, 1) for dimension
    s_shortened_seed_time_series=s_seed_time_series[:shortened_listener_story_last_frame]
    # print(f_s_seed_time_series,s_seed_time_series)
    # s_shortened_seed_time_series_ortho=spectral.orthogonalize(s_shortened_seed_time_series)
    # print(s_seed_time_series)
    # debu_orth = debug_orthogonalize(s_shortened_seed_time_series)
    # print(debu_orth)
    
    #speaker precedes (-6.4-3.2)
    # -6.4 (/.8 = -8)

    #10 vols SP1
    array_temp=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    sp1_temp_ts=s_shortened_seed_time_series[10:]
    
    avg=sp1_temp_ts.mean()
    stdev=sp1_temp_ts.std()
    normalized_sp1_shortened_seed_time_series_minus_10 = (sp1_temp_ts - avg) / stdev
    normalized_sp1_shortened_seed_time_series_minus_10 = normalized_sp1_shortened_seed_time_series_minus_10.reshape(-1, 1) 
    
    sp1_shortened_seed_time_series_minus_10 = np.concatenate((normalized_sp1_shortened_seed_time_series_minus_10, array_temp), axis=0)


    #9 vols SP2
    array_temp=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    sp2_temp_ts=s_shortened_seed_time_series[9:]
    
    avg=sp2_temp_ts.mean()
    stdev=sp2_temp_ts.std()
    normalized_sp2_shortened_seed_time_series_minus_9 = (sp2_temp_ts - avg) / stdev
    normalized_sp2_shortened_seed_time_series_minus_9 = normalized_sp2_shortened_seed_time_series_minus_9.reshape(-1, 1) 
    
    sp2_shortened_seed_time_series_minus_9 = np.concatenate((normalized_sp2_shortened_seed_time_series_minus_9, array_temp), axis=0)
    
    #8 vols SP3
    array_temp=np.array([[0, 0, 0, 0, 0, 0, 0, 0]]).T
    sp3_temp_ts=s_shortened_seed_time_series[8:]
    
    avg=sp3_temp_ts.mean()
    stdev=sp3_temp_ts.std()
    normalized_sp3_shortened_seed_time_series_minus_8 = (sp3_temp_ts - avg) / stdev
    normalized_sp3_shortened_seed_time_series_minus_8 = normalized_sp3_shortened_seed_time_series_minus_8.reshape(-1, 1)
     
    sp3_shortened_seed_time_series_minus_8 = np.concatenate((normalized_sp3_shortened_seed_time_series_minus_8, array_temp), axis=0)
  
    ### 7 vols SP4
    array_temp=np.array([[0, 0, 0, 0, 0, 0, 0]]).T
    sp4_temp_ts=s_shortened_seed_time_series[7:]
    
    avg=sp4_temp_ts.mean()
    stdev=sp4_temp_ts.std()
    normalized_sp4_shortened_seed_time_series_minus_7 = (sp4_temp_ts - avg) / stdev
    normalized_sp4_shortened_seed_time_series_minus_7 = normalized_sp4_shortened_seed_time_series_minus_7.reshape(-1, 1) 

    sp4_shortened_seed_time_series_minus_7 = np.concatenate((normalized_sp4_shortened_seed_time_series_minus_7, array_temp), axis=0)
    
    # 6 vols SP5
    # -6.4 + 1.6 = -4.8 (/.8 = -6)
    array_temp=np.array([[0, 0, 0, 0, 0, 0]]).T
    sp5_temp_ts=s_shortened_seed_time_series[6:]
    
    avg=sp5_temp_ts.mean()
    stdev=sp5_temp_ts.std()

    normalized_sp5_shortened_seed_time_series_minus_6 = (sp5_temp_ts - avg) / stdev
    normalized_sp5_shortened_seed_time_series_minus_6 = normalized_sp5_shortened_seed_time_series_minus_6.reshape(-1, 1) 
    
    sp5_shortened_seed_time_series_minus_6 = np.concatenate((normalized_sp5_shortened_seed_time_series_minus_6,array_temp), axis=0)
    
    ### 5 vols SP6
    array_temp=np.array([[0, 0, 0, 0, 0]]).T
    sp6_temp_ts=s_shortened_seed_time_series[5:]
    
    avg=sp6_temp_ts.mean()
    stdev=sp6_temp_ts.std()
    
    normalized_sp6_shortened_seed_time_series_minus_5 = (sp6_temp_ts - avg) / stdev
    normalized_sp6_shortened_seed_time_series_minus_5 = normalized_sp6_shortened_seed_time_series_minus_5.reshape(-1, 1) 

    sp6_shortened_seed_time_series_minus_5 = np.concatenate((normalized_sp6_shortened_seed_time_series_minus_5, array_temp), axis=0)
    
    #4 vols SP7
    # -4.8 + 1.6 = -3.2 (/.8 = - 4)
    array_temp=np.array([[0, 0, 0, 0]]).T
    sp7_temp_ts=s_shortened_seed_time_series[4:]
    avg=sp7_temp_ts.mean()
    stdev=sp7_temp_ts.std()
    
    normalized_sp7_shortened_seed_time_series_minus_4 = (sp7_temp_ts - avg) / stdev
    normalized_sp7_shortened_seed_time_series_minus_4 = normalized_sp7_shortened_seed_time_series_minus_4.reshape(-1, 1) 
    
    sp7_shortened_seed_time_series_minus_4 = np.concatenate((normalized_sp7_shortened_seed_time_series_minus_4,array_temp), axis=0)
    
    #3 vols SP8
    array_temp=np.array([[0, 0, 0]]).T
    sp8_ts=s_shortened_seed_time_series[3:]
    avg=sp8_ts.mean()
    stdev=sp8_ts.std()
    
    normalized_sp8_shortened_seed_time_series_minus_3 = (sp8_ts - avg) / stdev
    normalized_sp8_shortened_seed_time_series_minus_3 = normalized_sp8_shortened_seed_time_series_minus_3.reshape(-1, 1) 
    
    sp8_shortened_seed_time_series_minus_3 = np.concatenate((normalized_sp8_shortened_seed_time_series_minus_3,array_temp), axis=0)
    
    #2 vols SP9
    array_temp=np.array([[0, 0]]).T
    sp9_temp_ts=s_shortened_seed_time_series[2:]
    avg=sp9_temp_ts.mean()
    stdev=sp9_temp_ts.std()
    
    normalized_sp9_shortened_seed_time_series_minus_2 = (sp9_temp_ts - avg) / stdev
    normalized_sp9_shortened_seed_time_series_minus_2 = normalized_sp9_shortened_seed_time_series_minus_2.reshape(-1, 1) 
   
    sp9_shortened_seed_time_series_minus_2 = np.concatenate((normalized_sp9_shortened_seed_time_series_minus_2,array_temp), axis=0)
    
    #1 vols SP10
    array_temp=np.array([[0]]).T
    sp10_temp_ts=s_shortened_seed_time_series[1:]
    avg=sp10_temp_ts.mean()
    stdev=sp10_temp_ts.std()
    
    normalized_sp10_shortened_seed_time_series_minus1 = (sp10_temp_ts - avg) / stdev
    normalized_sp10_shortened_seed_time_series_minus1 = normalized_sp10_shortened_seed_time_series_minus1.reshape(-1, 1) 
   
    sp10_shortened_seed_time_series_minus_1 = np.concatenate((normalized_sp10_shortened_seed_time_series_minus1,array_temp), axis=0)
    
    # listener precedes (3.2-6.4)
    
    # + 1 vols
    array_temp=np.array([[0]]).T
    
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-1)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_lp1_shortened_seed_time_series_pos_1 = (temp_ts - avg) / stdev
    normalized_lp1_shortened_seed_time_series_pos_1 = normalized_lp1_shortened_seed_time_series_pos_1.reshape(-1, 1) 
    lp1_shortened_seed_time_series_pos_1 = np.concatenate((array_temp,normalized_lp1_shortened_seed_time_series_pos_1), axis=0) 
    
    
    # + 2 vols
    array_temp=np.array([[0, 0]]).T
    
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-2)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_lp2_shortened_seed_time_series_pos_2 = (temp_ts - avg) / stdev
    normalized_lp2_shortened_seed_time_series_pos_2 = normalized_lp2_shortened_seed_time_series_pos_2.reshape(-1, 1) 
    lp2_shortened_seed_time_series_pos_2 = np.concatenate((array_temp,normalized_lp2_shortened_seed_time_series_pos_2), axis=0)
    
    # + 3 vols
    array_temp=np.array([[0, 0, 0]]).T
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-3)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_lp3_shortened_seed_time_series_pos_3 = (temp_ts - avg) / stdev
    normalized_lp3_shortened_seed_time_series_pos_3 = normalized_lp3_shortened_seed_time_series_pos_3.reshape(-1, 1) 
    lp3_shortened_seed_time_series_pos_3 = np.concatenate((array_temp,normalized_lp3_shortened_seed_time_series_pos_3), axis=0) #246
 

    # + 4 vols
    # 3.2 (/.8 = 4)
    array_temp=np.array([[0, 0, 0, 0]]).T
    
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-4)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    normalized_lp4_shortened_seed_time_series_pos_4 = (temp_ts - avg) / stdev
    normalized_lp4_shortened_seed_time_series_pos_4 = normalized_lp4_shortened_seed_time_series_pos_4.reshape(-1, 1) 
    lp4_shortened_seed_time_series_pos_4 = np.concatenate((array_temp,normalized_lp4_shortened_seed_time_series_pos_4), axis=0) #246

    # + 5 vols
    # 3.2 (/.8 = 4)
    array_temp=np.array([[0, 0, 0, 0, 0]]).T
    
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-5)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_lp5_shortened_seed_time_series_pos_5 = (temp_ts - avg) / stdev
    normalized_lp5_shortened_seed_time_series_pos_5 = normalized_lp5_shortened_seed_time_series_pos_5.reshape(-1, 1) 
    lp5_shortened_seed_time_series_pos_5 = np.concatenate((array_temp,normalized_lp5_shortened_seed_time_series_pos_5), axis=0) #246
    
    ## 6 vols
    # 3.2 +1.6 = 4.8 (/.8 = 6)
    array_temp=np.array([[0, 0, 0, 0, 0, 0]]).T
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-6)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_lp6_shortened_seed_time_series_pos_6 = (temp_ts - avg) / stdev
    normalized_lp6_shortened_seed_time_series_pos_6 = normalized_lp6_shortened_seed_time_series_pos_6.reshape(-1, 1) 
    lp6_shortened_seed_time_series_pos_6 = np.concatenate((array_temp,normalized_lp6_shortened_seed_time_series_pos_6), axis=0) #244
    
    ## 7 vols
    array_temp=np.array([[0, 0, 0, 0, 0, 0, 0]]).T
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-7)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_lp7_shortened_seed_time_series_pos_7 = (temp_ts - avg) / stdev
    normalized_lp7_shortened_seed_time_series_pos_7 = normalized_lp7_shortened_seed_time_series_pos_7.reshape(-1, 1) 
    
    lp7_shortened_seed_time_series_pos_7 = np.concatenate((array_temp,normalized_lp7_shortened_seed_time_series_pos_7), axis=0) #244

    ## 8 vols
    # 4.8 + 1.6 = 6.4 (/.8 = 8)
    array_temp=np.array([[0, 0, 0, 0, 0, 0, 0, 0]]).T
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-8)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_lp8_shortened_seed_time_series_pos_8 = (temp_ts - avg) / stdev
    normalized_lp8_shortened_seed_time_series_pos_8 = normalized_lp8_shortened_seed_time_series_pos_8.reshape(-1, 1) 
    
    lp8_shortened_seed_time_series_pos_8 = np.concatenate((array_temp,normalized_lp8_shortened_seed_time_series_pos_8), axis=0)#242
    
    ## 9 vols
    array_temp=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-9)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_lp9_shortened_seed_time_series_pos_9 = (temp_ts - avg) / stdev
    normalized_lp9_shortened_seed_time_series_pos_9 = normalized_lp9_shortened_seed_time_series_pos_9.reshape(-1, 1) 
    
    lp9_shortened_seed_time_series_pos_9 = np.concatenate((array_temp,normalized_lp9_shortened_seed_time_series_pos_9), axis=0)
    
    ## 10 vols
    array_temp=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-10)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_lp10_shortened_seed_time_series_pos_10 = (temp_ts - avg) / stdev
    normalized_lp10_shortened_seed_time_series_pos_10 = normalized_lp10_shortened_seed_time_series_pos_10.reshape(-1, 1) 
    
    lp10_shortened_seed_time_series_pos_10 = np.concatenate((array_temp,normalized_lp10_shortened_seed_time_series_pos_10), axis=0)
    
    
    #### synchronous -1.6 - 1.6
    #2 vols
    # -1.6 (/.8 = -2)
    array_temp=np.array([[0, 0]]).T
    temp_ts=s_shortened_seed_time_series[2:]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_synchronous_1_shortened_seed_time_series_minus2 = (temp_ts - avg) / stdev
    normalized_synchronous_1_shortened_seed_time_series_minus2 = normalized_synchronous_1_shortened_seed_time_series_minus2.reshape(-1, 1) 
    
    synchronous_1_shortened_seed_time_series_minus2 = np.concatenate((normalized_synchronous_1_shortened_seed_time_series_minus2,array_temp), axis=0)
    
    ### - 1 vol
    array_temp=np.array([[0]]).T
    temp_ts=s_shortened_seed_time_series[1:]
    avg=temp_ts.mean()
    stdev=temp_ts.std()
    
    normalized_synchronous_2_shortened_seed_time_series_minus1 = (temp_ts - avg) / stdev
    normalized_synchronous_2_shortened_seed_time_series_minus1 = normalized_synchronous_2_shortened_seed_time_series_minus1.reshape(-1, 1)
    
    synchronous_2_shortened_seed_time_series_minus1 = np.concatenate((normalized_synchronous_2_shortened_seed_time_series_minus1,array_temp), axis=0)

    #0 vols 
    # -1.6+1.6 = 0 (/.8 = 0)
    synchronous_3_shortened_seed_time_series_0 = s_shortened_seed_time_series
    
    avg=synchronous_3_shortened_seed_time_series_0.mean()
    stdev=synchronous_3_shortened_seed_time_series_0.std()
    normalized_synchronous_3_shortened_seed_time_series_0 = (synchronous_3_shortened_seed_time_series_0 - avg) / stdev
    normalized_synchronous_3_shortened_seed_time_series_0 = normalized_synchronous_3_shortened_seed_time_series_0.reshape(-1, 1) 
    
    synchronous_3_shortened_seed_time_series_0= normalized_synchronous_3_shortened_seed_time_series_0
    
    # +1 vol 
    array_temp=np.array([[0]]).T
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-1)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()

    normalized_synchronous_4_shortened_seed_time_series_pos_1 = (temp_ts - avg) / stdev
    normalized_synchronous_4_shortened_seed_time_series_pos_1 = normalized_synchronous_4_shortened_seed_time_series_pos_1.reshape(-1, 1) 
    synchronous_4_shortened_seed_time_series_pos_1 = np.concatenate((array_temp,normalized_synchronous_4_shortened_seed_time_series_pos_1), axis=0)

    #+2 vol
    # 0 + 1.6 = 1.6 (/.8 = 2)
    array_temp=np.array([[0, 0]]).T
    temp_ts=s_shortened_seed_time_series[:(shortened_listener_story_last_frame-2)]
    avg=temp_ts.mean()
    stdev=temp_ts.std()

    normalized_synchronous_5_shortened_seed_time_series_pos_2 = (temp_ts - avg) / stdev
    normalized_synchronous_5_shortened_seed_time_series_pos_2 = normalized_synchronous_5_shortened_seed_time_series_pos_2.reshape(-1, 1) 
    synchronous_5_shortened_seed_time_series_pos_2 = np.concatenate((array_temp,normalized_synchronous_5_shortened_seed_time_series_pos_2), axis=0)

    # add_reg_full = np.concatenate([sp1_shortened_seed_time_series_minus_10,sp2_shortened_seed_time_series_minus_9,sp3_shortened_seed_time_series_minus_8, sp4_shortened_seed_time_series_minus_7, sp5_shortened_seed_time_series_minus_6, sp6_shortened_seed_time_series_minus_5, sp7_shortened_seed_time_series_minus_4,sp8_shortened_seed_time_series_minus_3,sp9_shortened_seed_time_series_minus_2, sp10_shortened_seed_time_series_minus_1,lp1_shortened_seed_time_series_pos_1, lp2_shortened_seed_time_series_pos_2, lp3_shortened_seed_time_series_pos_3,lp4_shortened_seed_time_series_pos_4,lp5_shortened_seed_time_series_pos_5, lp6_shortened_seed_time_series_pos_6, lp7_shortened_seed_time_series_pos_7, lp8_shortened_seed_time_series_pos_8, lp9_shortened_seed_time_series_pos_9, lp10_shortened_seed_time_series_pos_10, synchronous_1_shortened_seed_time_series_minus2,synchronous_2_shortened_seed_time_series_minus1,synchronous_3_shortened_seed_time_series_0, synchronous_4_shortened_seed_time_series_pos_1, synchronous_5_shortened_seed_time_series_pos_2, motionparameters_l], axis=1)
    
    add_reg_full = np.concatenate([sp1_shortened_seed_time_series_minus_10,sp2_shortened_seed_time_series_minus_9,sp3_shortened_seed_time_series_minus_8, sp4_shortened_seed_time_series_minus_7, sp5_shortened_seed_time_series_minus_6, sp6_shortened_seed_time_series_minus_5, sp7_shortened_seed_time_series_minus_4,sp8_shortened_seed_time_series_minus_3,sp9_shortened_seed_time_series_minus_2, sp10_shortened_seed_time_series_minus_1,lp1_shortened_seed_time_series_pos_1, lp2_shortened_seed_time_series_pos_2, lp3_shortened_seed_time_series_pos_3,lp4_shortened_seed_time_series_pos_4,lp5_shortened_seed_time_series_pos_5, lp6_shortened_seed_time_series_pos_6, lp7_shortened_seed_time_series_pos_7, lp8_shortened_seed_time_series_pos_8, lp9_shortened_seed_time_series_pos_9, lp10_shortened_seed_time_series_pos_10, synchronous_3_shortened_seed_time_series_0, motionparameters_l], axis=1)

    
    add_regs=add_reg_full
    add_reg_names = ["sp1_shortened_seed_time_series_minus_10"] + ["sp2_shortened_seed_time_series_minus_9"] + ["sp3_shortened_seed_time_series_minus_8"] + ["sp4_shortened_seed_time_series_minus_7"] + ["sp5_shortened_seed_time_series_minus_6"] + ["sp6_shortened_seed_time_series_minus_5"] + ["sp7_shortened_seed_time_series_minus_4"] + ["sp8_shortened_seed_time_series_minus_3"] + ["sp9_shortened_seed_time_series_minus_2"] + ["sp10_shortened_seed_time_series_minus_1"] + ["lp1_shortened_seed_time_series_pos_1"] + ["lp2_shortened_seed_time_series_pos_2"] + ["lp3_shortened_seed_time_series_pos_3"] + ["lp4_shortened_seed_time_series_pos_4"] + ["lp5_shortened_seed_time_series_pos_5"] + ["lp6_shortened_seed_time_series_pos_6"] + ["lp7_shortened_seed_time_series_pos_7"] + ["lp8_shortened_seed_time_series_pos_8"] + ["lp9_shortened_seed_time_series_pos_9"]+ ["lp10_shortened_seed_time_series_pos_10"] + ["synchronous_3_shortened_seed_time_series_0"] + ["motion_param_" + str(i) for i in range(1, motionparameters_l.shape[1] + 1)]

    # add_reg_names = ["sp1_shortened_seed_time_series_minus_10"] + ["sp2_shortened_seed_time_series_minus_9"] + ["sp3_shortened_seed_time_series_minus_8"] + ["sp4_shortened_seed_time_series_minus_7"] + ["sp5_shortened_seed_time_series_minus_6"] + ["sp6_shortened_seed_time_series_minus_5"] + ["sp7_shortened_seed_time_series_minus_4"] + ["sp8_shortened_seed_time_series_minus_3"] + ["sp9_shortened_seed_time_series_minus_2"] + ["sp10_shortened_seed_time_series_minus_1"] + ["lp1_shortened_seed_time_series_pos_1"] + ["lp2_shortened_seed_time_series_pos_2"] + ["lp3_shortened_seed_time_series_pos_3"] + ["lp4_shortened_seed_time_series_pos_4"] + ["lp5_shortened_seed_time_series_pos_5"] + ["lp6_shortened_seed_time_series_pos_6"] + ["lp7_shortened_seed_time_series_pos_7"] + ["lp8_shortened_seed_time_series_pos_8"] + ["lp9_shortened_seed_time_series_pos_9"]+ ["lp10_shortened_seed_time_series_pos_10"] + ["synchronous_1_shortened_seed_time_series_minus2"] + ["synchronous_2_shortened_seed_time_series_minus1"] + ["synchronous_3_shortened_seed_time_series_0"] + ["synchronous_4_shortened_seed_time_series_pos_1"] + ["synchronous_5_shortened_seed_time_series_pos_2"] + ["motion_param_" + str(i) for i in range(1, motionparameters_l.shape[1] + 1)]

    design_matrix = make_first_level_design_matrix(
    frametimes[:shortened_listener_story_last_frame],
    hrf_model="None",
    add_regs=add_regs,
    add_reg_names=add_reg_names,
    drift_model='polynomial')
    
    #listener
    listener_voxel_time_series=endog_l[coord[0], coord[1], coord[2], :]
    listener_seed_time_series = listener_voxel_time_series.reshape(-1, 1) # (n_scans, 1) for dimension
    model = sm.OLS(listener_seed_time_series,design_matrix)
    results = model.fit()
    betas.append(results.params)

    vox_contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns[0:29])
    }
    # print(basic_contrasts)

    contrasts={
        "speaker_precedes_vs_listener_precedes": 
            (basic_contrasts['sp3_shortened_seed_time_series_minus_8'] * 1/5 + basic_contrasts['sp4_shortened_seed_time_series_minus_7'] * 1/5 + basic_contrasts['sp5_shortened_seed_time_series_minus_6'] * 1/5 + basic_contrasts['sp6_shortened_seed_time_series_minus_5'] * 1/5 + basic_contrasts['sp7_shortened_seed_time_series_minus_4'] * 1/5) -
            (basic_contrasts['lp4_shortened_seed_time_series_pos_4'] * 1/5 + basic_contrasts['lp5_shortened_seed_time_series_pos_5'] * 1/5 + basic_contrasts['lp6_shortened_seed_time_series_pos_6'] * 1/5 + basic_contrasts['lp7_shortened_seed_time_series_pos_7'] * 1/5 +basic_contrasts['lp8_shortened_seed_time_series_pos_8'] * 1/5),
        "listener_precedes_vs_speaker_precedes": 
            (basic_contrasts['lp4_shortened_seed_time_series_pos_4']* 1/5 + basic_contrasts['lp5_shortened_seed_time_series_pos_5']* 1/5 + basic_contrasts['lp6_shortened_seed_time_series_pos_6']* 1/5 + basic_contrasts['lp7_shortened_seed_time_series_pos_7']* 1/5 + basic_contrasts['lp8_shortened_seed_time_series_pos_8']* 1/5) - 
            (basic_contrasts['sp3_shortened_seed_time_series_minus_8'] * 1/5 + basic_contrasts['sp4_shortened_seed_time_series_minus_7'] * 1/5 + basic_contrasts['sp5_shortened_seed_time_series_minus_6'] * 1/5 + basic_contrasts['sp6_shortened_seed_time_series_minus_5'] * 1/5 + basic_contrasts['sp7_shortened_seed_time_series_minus_4'] * 1/5),
        "listener_precedes_vs_baseline": 
            (basic_contrasts['lp4_shortened_seed_time_series_pos_4']* 1/5 + basic_contrasts['lp5_shortened_seed_time_series_pos_5']* 1/5 + basic_contrasts['lp6_shortened_seed_time_series_pos_6']* 1/5 + basic_contrasts['lp7_shortened_seed_time_series_pos_7']* 1/5 + basic_contrasts['lp8_shortened_seed_time_series_pos_8']* 1/5),
        "speaker_precedes_vs_baseline": 
            (basic_contrasts['sp3_shortened_seed_time_series_minus_8'] * 1/5 + basic_contrasts['sp4_shortened_seed_time_series_minus_7'] * 1/5 + basic_contrasts['sp5_shortened_seed_time_series_minus_6'] * 1/5 + basic_contrasts['sp6_shortened_seed_time_series_minus_5'] * 1/5 + basic_contrasts['sp7_shortened_seed_time_series_minus_4'] * 1/5),
        # "synchronous_minus_baseline": 
        #     (basic_contrasts['synchronous_1_shortened_seed_time_series_minus2'] * 1/5 + basic_contrasts['synchronous_2_shortened_seed_time_series_minus1'] * 1/5 + basic_contrasts['synchronous_3_shortened_seed_time_series_0'] * 1/5 + basic_contrasts['synchronous_4_shortened_seed_time_series_pos_1'] * 1/5 + basic_contrasts['synchronous_5_shortened_seed_time_series_pos_2']*1/5),
        # "speaker_precedes_vs_synchronous": 
        #     (basic_contrasts['lp4_shortened_seed_time_series_pos_4']* 1/5 + basic_contrasts['lp5_shortened_seed_time_series_pos_5']* 1/5 + basic_contrasts['lp6_shortened_seed_time_series_pos_6']* 1/5 + basic_contrasts['lp7_shortened_seed_time_series_pos_7']* 1/5 + basic_contrasts['lp8_shortened_seed_time_series_pos_8']* 1/5) - 
        #     (basic_contrasts['synchronous_1_shortened_seed_time_series_minus2'] * 1/5 + basic_contrasts['synchronous_2_shortened_seed_time_series_minus1'] * 1/5 + basic_contrasts['synchronous_3_shortened_seed_time_series_0'] * 1/5 + basic_contrasts['synchronous_4_shortened_seed_time_series_pos_1'] * 1/5 + basic_contrasts['synchronous_5_shortened_seed_time_series_pos_2']*1/5),
            
        # "synchronous_vs_speaker_precedes": 
        #     (basic_contrasts['synchronous_1_shortened_seed_time_series_minus2'] * 1/5 + basic_contrasts['synchronous_2_shortened_seed_time_series_minus1'] * 1/5 + basic_contrasts['synchronous_3_shortened_seed_time_series_0'] * 1/5 + basic_contrasts['synchronous_4_shortened_seed_time_series_pos_1'] * 1/5 + basic_contrasts['synchronous_5_shortened_seed_time_series_pos_2']*1/5) - 
        #     (basic_contrasts['sp3_shortened_seed_time_series_minus_8'] * 1/5 + basic_contrasts['sp4_shortened_seed_time_series_minus_7'] * 1/5 + basic_contrasts['sp5_shortened_seed_time_series_minus_6'] * 1/5 + basic_contrasts['sp6_shortened_seed_time_series_minus_5'] * 1/5 + basic_contrasts['sp7_shortened_seed_time_series_minus_4'] * 1/5),
            
        # "listener_precedes_vs_synchronous": 
        #     (basic_contrasts['lp4_shortened_seed_time_series_pos_4']* 1/5 + basic_contrasts['lp5_shortened_seed_time_series_pos_5']* 1/5 + basic_contrasts['lp6_shortened_seed_time_series_pos_6']* 1/5 + basic_contrasts['lp7_shortened_seed_time_series_pos_7']* 1/5 + basic_contrasts['lp8_shortened_seed_time_series_pos_8']* 1/5) - 
        #     (basic_contrasts['synchronous_1_shortened_seed_time_series_minus2'] * 1/5 + basic_contrasts['synchronous_2_shortened_seed_time_series_minus1'] * 1/5 + basic_contrasts['synchronous_3_shortened_seed_time_series_0'] * 1/5 + basic_contrasts['synchronous_4_shortened_seed_time_series_pos_1'] * 1/5 + basic_contrasts['synchronous_5_shortened_seed_time_series_pos_2']*1/5),
            
        # "synchronous_vs_listener_precedes":
        #     (basic_contrasts['synchronous_1_shortened_seed_time_series_minus2'] * 1/5 + basic_contrasts['synchronous_2_shortened_seed_time_series_minus1'] * 1/5 + basic_contrasts['synchronous_3_shortened_seed_time_series_0'] * 1/5 + basic_contrasts['synchronous_4_shortened_seed_time_series_pos_1'] * 1/5 + basic_contrasts['synchronous_5_shortened_seed_time_series_pos_2']*1/5) - 
        #     (basic_contrasts['lp4_shortened_seed_time_series_pos_4']* 1/5 + basic_contrasts['lp5_shortened_seed_time_series_pos_5']* 1/5 +  basic_contrasts['lp6_shortened_seed_time_series_pos_6']* 1/5 +  basic_contrasts['lp7_shortened_seed_time_series_pos_7']* 1/5 +  basic_contrasts['lp8_shortened_seed_time_series_pos_8']* 1/5),
            
        "synchronous_time0_vs_baseline": 
            (basic_contrasts['synchronous_3_shortened_seed_time_series_0']-0)
    }
    # print(basic_contrasts)


    contrast_01_beta_score=np.dot(contrasts["speaker_precedes_vs_listener_precedes"],results.params)
    contrast_03_beta_score=np.dot(contrasts["listener_precedes_vs_speaker_precedes"],results.params)
    contrast_05_beta_score=np.dot(contrasts["listener_precedes_vs_baseline"],results.params)
    contrast_06_beta_score=np.dot(contrasts["speaker_precedes_vs_baseline"],results.params)
    # contrast_07_beta_score=np.dot(contrasts["synchronous_minus_baseline"],results.params)
    # contrast_08_beta_score=np.dot(contrasts["speaker_precedes_vs_synchronous"],results.params)
    # contrast_09_beta_score=np.dot(contrasts["synchronous_vs_speaker_precedes"],results.params)
    # contrast_10_beta_score=np.dot(contrasts["listener_precedes_vs_synchronous"],results.params)
    # contrast_11_beta_score=np.dot(contrasts["synchronous_vs_listener_precedes"],results.params)
    contrast_12_beta_score=np.dot(contrasts["synchronous_time0_vs_baseline"],results.params)

    betas_con1.append(contrast_01_beta_score)
    betas_con3.append(contrast_03_beta_score)
    betas_con5.append(contrast_05_beta_score)
    betas_con6.append(contrast_06_beta_score)
    # betas_con7.append(contrast_07_beta_score)
    # betas_con8.append(contrast_08_beta_score)
    # betas_con9.append(contrast_09_beta_score)
    # betas_con10.append(contrast_10_beta_score)
    # betas_con11.append(contrast_11_beta_score)
    betas_con12.append(contrast_12_beta_score)

    contrast_01_tvalues=np.dot(contrasts["speaker_precedes_vs_listener_precedes"],results.tvalues)
    contrast_03_tvalues=np.dot(contrasts["listener_precedes_vs_speaker_precedes"],results.tvalues)
    contrast_05_tvalues=np.dot(contrasts["listener_precedes_vs_baseline"],results.tvalues)
    contrast_06_tvalues=np.dot(contrasts["speaker_precedes_vs_baseline"],results.tvalues)
    # contrast_07_tvalues=np.dot(contrasts["synchronous_minus_baseline"],results.tvalues)
    # contrast_08_tvalues=np.dot(contrasts["speaker_precedes_vs_synchronous"],results.tvalues)
    # contrast_09_tvalues=np.dot(contrasts["synchronous_vs_speaker_precedes"],results.tvalues)
    # contrast_10_tvalues=np.dot(contrasts["listener_precedes_vs_synchronous"],results.tvalues)
    # contrast_11_tvalues=np.dot(contrasts["synchronous_vs_listener_precedes"],results.tvalues)
    contrast_12_tvalues=np.dot(contrasts["synchronous_time0_vs_baseline"],results.tvalues)

    tvalues_con1.append(contrast_01_tvalues)
    tvalues_con3.append(contrast_03_tvalues)
    tvalues_con5.append(contrast_05_tvalues)
    tvalues_con6.append(contrast_06_tvalues)
    # tvalues_con7.append(contrast_07_tvalues)
    # tvalues_con8.append(contrast_08_tvalues)
    # tvalues_con9.append(contrast_09_tvalues)
    # tvalues_con10.append(contrast_10_tvalues)
    # tvalues_con11.append(contrast_11_tvalues)
    tvalues_con12.append(contrast_12_tvalues)
    
    #speaker
    contrast_001_beta_score=np.dot(basic_contrasts["sp1_shortened_seed_time_series_minus_10"],results.params)
    contrast_002_beta_score=np.dot(basic_contrasts["sp2_shortened_seed_time_series_minus_9"],results.params)
    contrast_003_beta_score=np.dot(basic_contrasts["sp3_shortened_seed_time_series_minus_8"],results.params)
    contrast_004_beta_score=np.dot(basic_contrasts["sp4_shortened_seed_time_series_minus_7"],results.params)
    contrast_005_beta_score=np.dot(basic_contrasts["sp5_shortened_seed_time_series_minus_6"],results.params)
    contrast_006_beta_score=np.dot(basic_contrasts["sp6_shortened_seed_time_series_minus_5"],results.params)
    contrast_007_beta_score=np.dot(basic_contrasts["sp7_shortened_seed_time_series_minus_4"],results.params)
    contrast_008_beta_score=np.dot(basic_contrasts["sp8_shortened_seed_time_series_minus_3"],results.params)
    contrast_009_beta_score=np.dot(basic_contrasts["sp9_shortened_seed_time_series_minus_2"],results.params)
    contrast_010_beta_score=np.dot(basic_contrasts["sp10_shortened_seed_time_series_minus_1"],results.params)
    # contrast_011_beta_score=np.dot(basic_contrasts["synchronous_1_shortened_seed_time_series_minus2"],results.params)
    # contrast_012_beta_score=np.dot(basic_contrasts["synchronous_2_shortened_seed_time_series_minus1"],results.params)
    contrast_013_beta_score=np.dot(basic_contrasts["synchronous_3_shortened_seed_time_series_0"],results.params)
    # contrast_014_beta_score=np.dot(basic_contrasts["synchronous_4_shortened_seed_time_series_pos_1"],results.params)
    # contrast_015_beta_score=np.dot(basic_contrasts["synchronous_5_shortened_seed_time_series_pos_2"],results.params)

 
    betas_con001.append(contrast_001_beta_score)
    betas_con002.append(contrast_002_beta_score)
    betas_con003.append(contrast_003_beta_score)
    betas_con004.append(contrast_004_beta_score)
    betas_con005.append(contrast_005_beta_score)
    betas_con006.append(contrast_006_beta_score)
    betas_con007.append(contrast_007_beta_score)
    betas_con008.append(contrast_008_beta_score)
    betas_con009.append(contrast_009_beta_score)
    betas_con010.append(contrast_010_beta_score)
    # betas_con011.append(contrast_011_beta_score)
    # betas_con012.append(contrast_012_beta_score)
    betas_con013.append(contrast_013_beta_score)
    # betas_con014.append(contrast_014_beta_score)
    # betas_con015.append(contrast_015_beta_score)

    
    #listener
    contrast_0001_beta_score=np.dot(basic_contrasts["lp1_shortened_seed_time_series_pos_1"],results.params)
    contrast_0002_beta_score=np.dot(basic_contrasts["lp2_shortened_seed_time_series_pos_2"],results.params)
    contrast_0003_beta_score=np.dot(basic_contrasts["lp3_shortened_seed_time_series_pos_3"],results.params)
    contrast_0004_beta_score=np.dot(basic_contrasts["lp4_shortened_seed_time_series_pos_4"],results.params)
    contrast_0005_beta_score=np.dot(basic_contrasts["lp5_shortened_seed_time_series_pos_5"],results.params)
    contrast_0006_beta_score=np.dot(basic_contrasts["lp6_shortened_seed_time_series_pos_6"],results.params)
    contrast_0007_beta_score=np.dot(basic_contrasts["lp7_shortened_seed_time_series_pos_7"],results.params)
    contrast_0008_beta_score=np.dot(basic_contrasts["lp8_shortened_seed_time_series_pos_8"],results.params)
    contrast_0009_beta_score=np.dot(basic_contrasts["lp9_shortened_seed_time_series_pos_9"],results.params)
    contrast_0010_beta_score=np.dot(basic_contrasts["lp10_shortened_seed_time_series_pos_10"],results.params)
    # contrast_0011_beta_score=np.dot(basic_contrasts["synchronous_1_shortened_seed_time_series_minus2"],results.params)
    # contrast_0012_beta_score=np.dot(basic_contrasts["synchronous_2_shortened_seed_time_series_minus1"],results.params)
    contrast_0013_beta_score=np.dot(basic_contrasts["synchronous_3_shortened_seed_time_series_0"],results.params)
    # contrast_0014_beta_score=np.dot(basic_contrasts["synchronous_4_shortened_seed_time_series_pos_1"],results.params)
    # contrast_0015_beta_score=np.dot(basic_contrasts["synchronous_5_shortened_seed_time_series_pos_2"],results.params)


    betas_con0001.append(contrast_0001_beta_score)
    betas_con0002.append(contrast_0002_beta_score)
    betas_con0003.append(contrast_0003_beta_score)
    betas_con0004.append(contrast_0004_beta_score)
    betas_con0005.append(contrast_0005_beta_score)
    betas_con0006.append(contrast_0006_beta_score)
    betas_con0007.append(contrast_0007_beta_score)
    betas_con0008.append(contrast_0008_beta_score)
    betas_con0009.append(contrast_0009_beta_score)
    betas_con0010.append(contrast_0010_beta_score)
    # betas_con0011.append(contrast_0011_beta_score)
    # betas_con0012.append(contrast_0012_beta_score)
    betas_con0013.append(contrast_0013_beta_score)
    # betas_con0014.append(contrast_0014_beta_score)    
    # betas_con0015.append(contrast_0015_beta_score)

    #speaker
    contrast_001_tvalues_score=np.dot(basic_contrasts["sp1_shortened_seed_time_series_minus_10"],results.tvalues)
    contrast_002_tvalues_score=np.dot(basic_contrasts["sp2_shortened_seed_time_series_minus_9"],results.tvalues)
    contrast_003_tvalues_score=np.dot(basic_contrasts["sp3_shortened_seed_time_series_minus_8"],results.tvalues)
    contrast_004_tvalues_score=np.dot(basic_contrasts["sp4_shortened_seed_time_series_minus_7"],results.tvalues)
    contrast_005_tvalues_score=np.dot(basic_contrasts["sp5_shortened_seed_time_series_minus_6"],results.tvalues)
    contrast_006_tvalues_score=np.dot(basic_contrasts["sp6_shortened_seed_time_series_minus_5"],results.tvalues)
    contrast_007_tvalues_score=np.dot(basic_contrasts["sp7_shortened_seed_time_series_minus_4"],results.tvalues)
    contrast_008_tvalues_score=np.dot(basic_contrasts["sp8_shortened_seed_time_series_minus_3"],results.tvalues)
    contrast_009_tvalues_score=np.dot(basic_contrasts["sp9_shortened_seed_time_series_minus_2"],results.tvalues)
    contrast_010_tvalues_score=np.dot(basic_contrasts["sp10_shortened_seed_time_series_minus_1"],results.tvalues)
    # contrast_011_tvalues_score=np.dot(basic_contrasts["synchronous_1_shortened_seed_time_series_minus2"],results.tvalues)
    # contrast_012_tvalues_score=np.dot(basic_contrasts["synchronous_2_shortened_seed_time_series_minus1"],results.tvalues)
    contrast_013_tvalues_score=np.dot(basic_contrasts["synchronous_3_shortened_seed_time_series_0"],results.tvalues)
    # contrast_014_tvalues_score=np.dot(basic_contrasts["synchronous_4_shortened_seed_time_series_pos_1"],results.tvalues)
    # contrast_015_tvalues_score=np.dot(basic_contrasts["synchronous_5_shortened_seed_time_series_pos_2"],results.tvalues)

 
    tvalues_con001.append(contrast_001_tvalues_score)
    tvalues_con002.append(contrast_002_tvalues_score)
    tvalues_con003.append(contrast_003_tvalues_score)
    tvalues_con004.append(contrast_004_tvalues_score)
    tvalues_con005.append(contrast_005_tvalues_score)
    tvalues_con006.append(contrast_006_tvalues_score)
    tvalues_con007.append(contrast_007_tvalues_score)
    tvalues_con008.append(contrast_008_tvalues_score)
    tvalues_con009.append(contrast_009_tvalues_score)
    tvalues_con010.append(contrast_010_tvalues_score)
    # tvalues_con011.append(contrast_011_tvalues_score)
    # tvalues_con012.append(contrast_012_tvalues_score)
    tvalues_con013.append(contrast_013_tvalues_score)
    # tvalues_con014.append(contrast_014_tvalues_score)
    # tvalues_con015.append(contrast_015_tvalues_score)



    #listener tvalues 
    contrast_0001_tvalues_score=np.dot(basic_contrasts["lp1_shortened_seed_time_series_pos_1"],results.tvalues)
    contrast_0002_tvalues_score=np.dot(basic_contrasts["lp2_shortened_seed_time_series_pos_2"],results.tvalues)
    contrast_0003_tvalues_score=np.dot(basic_contrasts["lp3_shortened_seed_time_series_pos_3"],results.tvalues)
    contrast_0004_tvalues_score=np.dot(basic_contrasts["lp4_shortened_seed_time_series_pos_4"],results.tvalues)
    contrast_0005_tvalues_score=np.dot(basic_contrasts["lp5_shortened_seed_time_series_pos_5"],results.tvalues)
    contrast_0006_tvalues_score=np.dot(basic_contrasts["lp6_shortened_seed_time_series_pos_6"],results.tvalues)
    contrast_0007_tvalues_score=np.dot(basic_contrasts["lp7_shortened_seed_time_series_pos_7"],results.tvalues)
    contrast_0008_tvalues_score=np.dot(basic_contrasts["lp8_shortened_seed_time_series_pos_8"],results.tvalues)
    contrast_0009_tvalues_score=np.dot(basic_contrasts["lp9_shortened_seed_time_series_pos_9"],results.tvalues)
    contrast_0010_tvalues_score=np.dot(basic_contrasts["lp10_shortened_seed_time_series_pos_10"],results.tvalues)
    # contrast_0011_tvalues_score=np.dot(basic_contrasts["synchronous_1_shortened_seed_time_series_minus2"],results.tvalues)
    # contrast_0012_tvalues_score=np.dot(basic_contrasts["synchronous_2_shortened_seed_time_series_minus1"],results.tvalues)
    contrast_0013_tvalues_score=np.dot(basic_contrasts["synchronous_3_shortened_seed_time_series_0"],results.tvalues)
    # contrast_0014_tvalues_score=np.dot(basic_contrasts["synchronous_4_shortened_seed_time_series_pos_1"],results.tvalues)
    # contrast_0015_tvalues_score=np.dot(basic_contrasts["synchronous_5_shortened_seed_time_series_pos_2"],results.tvalues)


    tvalues_con0001.append(contrast_0001_tvalues_score)
    tvalues_con0002.append(contrast_0002_tvalues_score)
    tvalues_con0003.append(contrast_0003_tvalues_score)
    tvalues_con0004.append(contrast_0004_tvalues_score)
    tvalues_con0005.append(contrast_0005_tvalues_score)
    tvalues_con0006.append(contrast_0006_tvalues_score)
    tvalues_con0007.append(contrast_0007_tvalues_score)
    tvalues_con0008.append(contrast_0008_tvalues_score)
    tvalues_con0009.append(contrast_0009_tvalues_score)
    tvalues_con0010.append(contrast_0010_tvalues_score)
    # tvalues_con0011.append(contrast_0011_tvalues_score)
    # tvalues_con0012.append(contrast_0012_tvalues_score)
    tvalues_con0013.append(contrast_0013_tvalues_score)
    # tvalues_con0014.append(contrast_0014_tvalues_score)    
    # tvalues_con0015.append(contrast_0015_tvalues_score)
    
plot_design_matrix(design_matrix)
plt.show()


blank_brain_betas_contrast_001=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_002=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_003=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_004=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_005=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_006=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_007=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_008=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_009=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_010=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_013=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_betas_contrast_014=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_betas_contrast_015=np.zeros(mni_2mm_brainmask_data.shape)

for i, coord in enumerate(mni_2mm_brainmask_non_zero_coords):
    blank_brain_betas_contrast_001[coord[0], coord[1], coord[2]] = betas_con001[i]
    blank_brain_betas_contrast_002[coord[0], coord[1], coord[2]] = betas_con002[i]
    blank_brain_betas_contrast_003[coord[0], coord[1], coord[2]] = betas_con003[i]
    blank_brain_betas_contrast_004[coord[0], coord[1], coord[2]] = betas_con004[i]
    blank_brain_betas_contrast_005[coord[0], coord[1], coord[2]] = betas_con005[i]
    blank_brain_betas_contrast_006[coord[0], coord[1], coord[2]] = betas_con006[i]
    blank_brain_betas_contrast_007[coord[0], coord[1], coord[2]] = betas_con007[i]
    blank_brain_betas_contrast_008[coord[0], coord[1], coord[2]] = betas_con008[i]
    blank_brain_betas_contrast_009[coord[0], coord[1], coord[2]] = betas_con009[i]
    blank_brain_betas_contrast_010[coord[0], coord[1], coord[2]] = betas_con010[i]
    blank_brain_betas_contrast_013[coord[0], coord[1], coord[2]] = betas_con013[i]
    


    # blank_brain_betas_contrast_005[coord[0], coord[1], coord[2]] = betas_con005[i]
    # blank_brain_betas_contrast_006[coord[0], coord[1], coord[2]] = betas_con006[i]
    # blank_brain_betas_contrast_007[coord[0], coord[1], coord[2]] = betas_con007[i]
    # blank_brain_betas_contrast_008[coord[0], coord[1], coord[2]] = betas_con008[i]
    # blank_brain_betas_contrast_009[coord[0], coord[1], coord[2]] = betas_con009[i]
    # blank_brain_betas_contrast_010[coord[0], coord[1], coord[2]] = betas_con010[i]
    # # blank_brain_betas_contrast_011[coord[0], coord[1], coord[2]] = betas_con011[i]
    # # blank_brain_betas_contrast_012[coord[0], coord[1], coord[2]] = betas_con012[i]
    # blank_brain_betas_contrast_013[coord[0], coord[1], coord[2]] = betas_con013[i]
    # # blank_brain_betas_contrast_014[coord[0], coord[1], coord[2]] = betas_con014[i]
    # # blank_brain_betas_contrast_015[coord[0], coord[1], coord[2]] = betas_con015[i]
    

beta_img_nifti_contrast_001 = nib.Nifti1Image(blank_brain_betas_contrast_001, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_002 = nib.Nifti1Image(blank_brain_betas_contrast_002, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_003 = nib.Nifti1Image(blank_brain_betas_contrast_003, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_004 = nib.Nifti1Image(blank_brain_betas_contrast_004, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_005 = nib.Nifti1Image(blank_brain_betas_contrast_005, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_006 = nib.Nifti1Image(blank_brain_betas_contrast_006, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_007 = nib.Nifti1Image(blank_brain_betas_contrast_007, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_008 = nib.Nifti1Image(blank_brain_betas_contrast_008, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_009 = nib.Nifti1Image(blank_brain_betas_contrast_009, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_010 = nib.Nifti1Image(blank_brain_betas_contrast_010, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_013 = nib.Nifti1Image(blank_brain_betas_contrast_013, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_014 = nib.Nifti1Image(blank_brain_betas_contrast_014, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_015 = nib.Nifti1Image(blank_brain_betas_contrast_015, affine=mni_2mm_brain_binarized_img.affine)


blank_brain_tvalues_contrast_001=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_002=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_003=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_004=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_005=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_006=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_007=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_008=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_009=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_010=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_013=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_tvalues_contrast_014=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_tvalues_contrast_015=np.zeros(mni_2mm_brainmask_data.shape)

for i, coord in enumerate(mni_2mm_brainmask_non_zero_coords):
    blank_brain_tvalues_contrast_001[coord[0], coord[1], coord[2]] = tvalues_con001[i]
    blank_brain_tvalues_contrast_002[coord[0], coord[1], coord[2]] = tvalues_con002[i]
    blank_brain_tvalues_contrast_003[coord[0], coord[1], coord[2]] = tvalues_con003[i]
    blank_brain_tvalues_contrast_004[coord[0], coord[1], coord[2]] = tvalues_con004[i]
    blank_brain_tvalues_contrast_005[coord[0], coord[1], coord[2]] = tvalues_con005[i]
    blank_brain_tvalues_contrast_006[coord[0], coord[1], coord[2]] = tvalues_con006[i]
    blank_brain_tvalues_contrast_007[coord[0], coord[1], coord[2]] = tvalues_con007[i]
    blank_brain_tvalues_contrast_008[coord[0], coord[1], coord[2]] = tvalues_con008[i]
    blank_brain_tvalues_contrast_009[coord[0], coord[1], coord[2]] = tvalues_con009[i]
    blank_brain_tvalues_contrast_010[coord[0], coord[1], coord[2]] = tvalues_con010[i]
    blank_brain_tvalues_contrast_013[coord[0], coord[1], coord[2]] = tvalues_con013[i]
    


tvalues_img_nifti_contrast_001 = nib.Nifti1Image(blank_brain_tvalues_contrast_001, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_002 = nib.Nifti1Image(blank_brain_tvalues_contrast_002, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_003 = nib.Nifti1Image(blank_brain_tvalues_contrast_003, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_004 = nib.Nifti1Image(blank_brain_tvalues_contrast_004, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_005 = nib.Nifti1Image(blank_brain_tvalues_contrast_005, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_006 = nib.Nifti1Image(blank_brain_tvalues_contrast_006, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_007 = nib.Nifti1Image(blank_brain_tvalues_contrast_007, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_008 = nib.Nifti1Image(blank_brain_tvalues_contrast_008, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_009 = nib.Nifti1Image(blank_brain_tvalues_contrast_009, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_010 = nib.Nifti1Image(blank_brain_tvalues_contrast_010, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_013 = nib.Nifti1Image(blank_brain_tvalues_contrast_013, affine=mni_2mm_brain_binarized_img.affine)
# tvalues_img_nifti_contrast_014 = nib.Nifti1Image(blank_brain_tvalues_contrast_014, affine=mni_2mm_brain_binarized_img.affine)
# tvalues_img_nifti_contrast_015 = nib.Nifti1Image(blank_brain_tvalues_contrast_015, affine=mni_2mm_brain_binarized_img.affine)



blank_brain_betas_contrast_0001=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0002=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0003=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0004=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0005=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0006=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0007=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0008=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0009=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0010=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_0013=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_betas_contrast_0014=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_betas_contrast_0015=np.zeros(mni_2mm_brainmask_data.shape)

for i, coord in enumerate(mni_2mm_brainmask_non_zero_coords):
    blank_brain_betas_contrast_0001[coord[0], coord[1], coord[2]] = betas_con001[i]
    blank_brain_betas_contrast_0002[coord[0], coord[1], coord[2]] = betas_con002[i]
    blank_brain_betas_contrast_0003[coord[0], coord[1], coord[2]] = betas_con003[i]
    blank_brain_betas_contrast_0004[coord[0], coord[1], coord[2]] = betas_con004[i]
    blank_brain_betas_contrast_0005[coord[0], coord[1], coord[2]] = betas_con005[i]
    blank_brain_betas_contrast_0006[coord[0], coord[1], coord[2]] = betas_con006[i]
    blank_brain_betas_contrast_0007[coord[0], coord[1], coord[2]] = betas_con007[i]
    blank_brain_betas_contrast_0008[coord[0], coord[1], coord[2]] = betas_con008[i]
    blank_brain_betas_contrast_0009[coord[0], coord[1], coord[2]] = betas_con009[i]
    blank_brain_betas_contrast_0010[coord[0], coord[1], coord[2]] = betas_con010[i]
    blank_brain_betas_contrast_0013[coord[0], coord[1], coord[2]] = betas_con013[i]
    

beta_img_nifti_contrast_0001 = nib.Nifti1Image(blank_brain_betas_contrast_0001, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0002 = nib.Nifti1Image(blank_brain_betas_contrast_0002, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0003 = nib.Nifti1Image(blank_brain_betas_contrast_0003, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0004 = nib.Nifti1Image(blank_brain_betas_contrast_0004, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0005 = nib.Nifti1Image(blank_brain_betas_contrast_0005, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0006 = nib.Nifti1Image(blank_brain_betas_contrast_0006, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0007 = nib.Nifti1Image(blank_brain_betas_contrast_0007, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0008 = nib.Nifti1Image(blank_brain_betas_contrast_0008, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0009 = nib.Nifti1Image(blank_brain_betas_contrast_0009, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0010 = nib.Nifti1Image(blank_brain_betas_contrast_0010, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_0013 = nib.Nifti1Image(blank_brain_betas_contrast_0013, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_0014 = nib.Nifti1Image(blank_brain_betas_contrast_0014, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_0015 = nib.Nifti1Image(blank_brain_betas_contrast_0015, affine=mni_2mm_brain_binarized_img.affine)


blank_brain_tvalues_0001=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0002=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0003=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0004=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0005=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0006=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0007=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0008=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0009=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0010=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_0013=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_tvalues_0014=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_tvalues_0015=np.zeros(mni_2mm_brainmask_data.shape)

for i, coord in enumerate(mni_2mm_brainmask_non_zero_coords):
    blank_brain_tvalues_0001[coord[0], coord[1], coord[2]] = betas_con001[i]
    blank_brain_tvalues_0002[coord[0], coord[1], coord[2]] = betas_con002[i]
    blank_brain_tvalues_0003[coord[0], coord[1], coord[2]] = betas_con003[i]
    blank_brain_tvalues_0004[coord[0], coord[1], coord[2]] = betas_con004[i]
    blank_brain_tvalues_0005[coord[0], coord[1], coord[2]] = betas_con005[i]
    blank_brain_tvalues_0006[coord[0], coord[1], coord[2]] = betas_con006[i]
    blank_brain_tvalues_0007[coord[0], coord[1], coord[2]] = betas_con007[i]
    blank_brain_tvalues_0008[coord[0], coord[1], coord[2]] = betas_con008[i]
    blank_brain_tvalues_0009[coord[0], coord[1], coord[2]] = betas_con009[i]
    blank_brain_tvalues_0010[coord[0], coord[1], coord[2]] = betas_con010[i]
    blank_brain_tvalues_0013[coord[0], coord[1], coord[2]] = betas_con013[i]
    

tvalues_img_nifti_contrast_0001 = nib.Nifti1Image(blank_brain_tvalues_0001, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0002 = nib.Nifti1Image(blank_brain_tvalues_0002, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0003 = nib.Nifti1Image(blank_brain_tvalues_0003, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0004 = nib.Nifti1Image(blank_brain_tvalues_0004, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0005 = nib.Nifti1Image(blank_brain_tvalues_0005, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0006 = nib.Nifti1Image(blank_brain_tvalues_0006, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0007 = nib.Nifti1Image(blank_brain_tvalues_0007, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0008 = nib.Nifti1Image(blank_brain_tvalues_0008, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0009 = nib.Nifti1Image(blank_brain_tvalues_0009, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0010 = nib.Nifti1Image(blank_brain_tvalues_0010, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_0013 = nib.Nifti1Image(blank_brain_tvalues_0013, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_0014 = nib.Nifti1Image(blank_brain_tvalues_0014, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_0015 = nib.Nifti1Image(blank_brain_tvalues_0015, affine=mni_2mm_brain_binarized_img.affine)


# In[15]:
blank_brain_betas=np.zeros(mni_2mm_brainmask_data.shape)

blank_brain_betas_contrast_01=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_03=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_05=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_06=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_betas_contrast_07=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_betas_contrast_08=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_betas_contrast_09=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_betas_contrast_10=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_betas_contrast_11=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_betas_contrast_12=np.zeros(mni_2mm_brainmask_data.shape)

for i, coord in enumerate(mni_2mm_brainmask_non_zero_coords):

    blank_brain_betas_contrast_01[coord[0], coord[1], coord[2]] = betas_con1[i]
    blank_brain_betas_contrast_03[coord[0], coord[1], coord[2]] = betas_con3[i]
    blank_brain_betas_contrast_05[coord[0], coord[1], coord[2]] = betas_con5[i]
    blank_brain_betas_contrast_06[coord[0], coord[1], coord[2]] = betas_con6[i]
    # blank_brain_betas_contrast_07[coord[0], coord[1], coord[2]] = betas_con7[i]
    # blank_brain_betas_contrast_08[coord[0], coord[1], coord[2]] = betas_con8[i]
    # blank_brain_betas_contrast_09[coord[0], coord[1], coord[2]] = betas_con9[i]
    # blank_brain_betas_contrast_10[coord[0], coord[1], coord[2]] = betas_con10[i]
    # blank_brain_betas_contrast_11[coord[0], coord[1], coord[2]] = betas_con11[i]
    blank_brain_betas_contrast_12[coord[0], coord[1], coord[2]] = betas_con12[i]

beta_img_nifti = nib.Nifti1Image(blank_brain_betas, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_01 = nib.Nifti1Image(blank_brain_betas_contrast_01, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_03 = nib.Nifti1Image(blank_brain_betas_contrast_03, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_05 = nib.Nifti1Image(blank_brain_betas_contrast_05, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_06 = nib.Nifti1Image(blank_brain_betas_contrast_06, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_07 = nib.Nifti1Image(blank_brain_betas_contrast_07, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_08 = nib.Nifti1Image(blank_brain_betas_contrast_08, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_09 = nib.Nifti1Image(blank_brain_betas_contrast_09, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_10 = nib.Nifti1Image(blank_brain_betas_contrast_10, affine=mni_2mm_brain_binarized_img.affine)
# beta_img_nifti_contrast_11 = nib.Nifti1Image(blank_brain_betas_contrast_11, affine=mni_2mm_brain_binarized_img.affine)
beta_img_nifti_contrast_12 = nib.Nifti1Image(blank_brain_betas_contrast_12, affine=mni_2mm_brain_binarized_img.affine)


# fixed effects at the individual level (all stories)
# story 1/2/3/4


# In[50]:

blank_brain_tvalues=np.zeros(mni_2mm_brainmask_data.shape)

blank_brain_tvalues_contrast_01=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_03=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_05=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_06=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_tvalues_contrast_07=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_tvalues_contrast_08=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_tvalues_contrast_09=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_tvalues_contrast_10=np.zeros(mni_2mm_brainmask_data.shape)
# blank_brain_tvalues_contrast_11=np.zeros(mni_2mm_brainmask_data.shape)
blank_brain_tvalues_contrast_12=np.zeros(mni_2mm_brainmask_data.shape)

for i, coord in enumerate(mni_2mm_brainmask_non_zero_coords):

    blank_brain_tvalues_contrast_01[coord[0], coord[1], coord[2]] =tvalues_con1[i]
    blank_brain_tvalues_contrast_03[coord[0], coord[1], coord[2]] =tvalues_con3[i]
    blank_brain_tvalues_contrast_05[coord[0], coord[1], coord[2]] =tvalues_con5[i]
    blank_brain_tvalues_contrast_06[coord[0], coord[1], coord[2]] =tvalues_con6[i]
    # blank_brain_tvalues_contrast_07[coord[0], coord[1], coord[2]] =tvalues_con7[i]
    # blank_brain_tvalues_contrast_08[coord[0], coord[1], coord[2]] =tvalues_con8[i]
    # blank_brain_tvalues_contrast_09[coord[0], coord[1], coord[2]] =tvalues_con9[i]
    # blank_brain_tvalues_contrast_10[coord[0], coord[1], coord[2]] =tvalues_con10[i]
    # blank_brain_tvalues_contrast_11[coord[0], coord[1], coord[2]] =tvalues_con11[i]
    blank_brain_tvalues_contrast_12[coord[0], coord[1], coord[2]] =tvalues_con12[i]

tvalues_img_nifti = nib.Nifti1Image(blank_brain_tvalues, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_01 = nib.Nifti1Image(blank_brain_tvalues_contrast_01, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_03 = nib.Nifti1Image(blank_brain_tvalues_contrast_03, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_05 = nib.Nifti1Image(blank_brain_tvalues_contrast_05, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_06 = nib.Nifti1Image(blank_brain_tvalues_contrast_06, affine=mni_2mm_brain_binarized_img.affine)
# tvalues_img_nifti_contrast_07 = nib.Nifti1Image(blank_brain_tvalues_contrast_07, affine=mni_2mm_brain_binarized_img.affine)
# tvalues_img_nifti_contrast_08 = nib.Nifti1Image(blank_brain_tvalues_contrast_08, affine=mni_2mm_brain_binarized_img.affine)
# tvalues_img_nifti_contrast_09 = nib.Nifti1Image(blank_brain_tvalues_contrast_09, affine=mni_2mm_brain_binarized_img.affine)
# tvalues_img_nifti_contrast_10 = nib.Nifti1Image(blank_brain_tvalues_contrast_10, affine=mni_2mm_brain_binarized_img.affine)
# tvalues_img_nifti_contrast_11 = nib.Nifti1Image(blank_brain_tvalues_contrast_11, affine=mni_2mm_brain_binarized_img.affine)
tvalues_img_nifti_contrast_12 = nib.Nifti1Image(blank_brain_tvalues_contrast_12, affine=mni_2mm_brain_binarized_img.affine)


# In[16]:


# nii_prefix='11051_2_1_'
# nii_story='story2_swar'


# ideal contrast order  
#     contrast_01_beta_score=np.dot(contrasts["speaker_precedes_vs_listener_precedes"],results.params)     
#     contrast_02_beta_score=np.dot(contrasts["listener_precedes_vs_speaker_precedes"],results.params)   
#     contrast_04_beta_score=np.dot(contrasts["listener_precedes_vs_baseline"],results.params)  
#     contrast_05_beta_score=np.dot(contrasts["baseline_vs_listener_precedes"],results.params)  
#     contrast_06_beta_score=np.dot(contrasts["speaker_precedes_vs_baseline"],results.params)  
#     contrast_07_beta_score=np.dot(contrasts["baseline_vs_speaker_precedes"],results.params)  
#     contrast_08_beta_score=np.dot(contrasts["synchronous_vs_baseline"],results.params)  
#     contrast_09_beta_score=np.dot(contrasts["baseline_vs_synchronous"],results.params)  
#     contrast_10_beta_score=np.dot(contrasts["speaker_precedes_vs_synchronous"],results.params)  
#     contrast_11_beta_score=np.dot(contrasts["synchronous_vs_speaker_precedes"],results.params)  
#     contrast_12_beta_score=np.dot(contrasts["listener_precedes_vs_synchronous"],results.params)   
#     contrast_13_beta_score=np.dot(contrasts["synchronous_vs_listener_precedes"],results.params) 

contrasts = [
    # Beta contrasts
    {"img": beta_img_nifti_contrast_001, "title": "Speaker Precedes beta params - 10 Volumes", "suffix": "sp1_minus_10_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_003, "title": "Speaker Precedes beta params - 9 Volumes", "suffix": "sp2_minus_9_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_003, "title": "Speaker Precedes beta params - 8 Volumes", "suffix": "sp3_minus_8_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_004, "title": "Speaker Precedes beta params - 7 Volumes", "suffix": "sp4_minus_7_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_005, "title": "Speaker Precedes beta params - 6 Volumes", "suffix": "sp5_minus_6_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_006, "title": "Speaker Precedes beta params - 5 Volumes", "suffix": "sp6_minus_5_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_007, "title": "Speaker Precedes beta params - 4 Volumes", "suffix": "sp7_minus_4_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_008, "title": "Speaker Precedes beta params - 3 Volumes", "suffix": "sp8_minus_3_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_009, "title": "Speaker Precedes beta params - 2 Volumes", "suffix": "sp9_minus_2_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_010, "title": "Speaker Precedes beta params - 1 Volume", "suffix": "sp10_minus_1_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_013, "title": "Synchronous at time 0 beta params", "suffix": "synchronous_3_0_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0001, "title": "Listener Precedes beta params - 1 Volume", "suffix": "lp1_pos_1_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0003, "title": "Listener Precedes beta params - 2 Volumes", "suffix": "lp2_pos_2_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0003, "title": "Listener Precedes beta params - 3 Volumes", "suffix": "lp3_pos_3_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0004, "title": "Listener Precedes beta params - 4 Volumes", "suffix": "lp4_pos_4_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0005, "title": "Listener Precedes beta params - 5 Volumes", "suffix": "lp5_pos_5_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0006, "title": "Listener Precedes beta params - 6 Volumes", "suffix": "lp6_pos_6_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0007, "title": "Listener Precedes beta params - 7 Volumes", "suffix": "lp7_pos_7_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0008, "title": "Listener Precedes beta params - 8 Volumes", "suffix": "lp8_pos_8_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0009, "title": "Listener Precedes beta params - 9 Volumes", "suffix": "lp9_pos_9_con.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_0010, "title": "Listener Precedes beta params - 10 Volumes", "suffix": "lp10_pos_10_con.nii.gz", "type": "beta"},
    # T-value contrasts
    {"img": tvalues_img_nifti_contrast_001, "title": "Speaker Precedes tvalues - 10 Volume", "suffix": "sp1_minus_10_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_002, "title": "Speaker Precedes tvalues - 9 Volumes", "suffix": "sp2_minus_9_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_003, "title": "Speaker Precedes tvalues - 8 Volumes", "suffix": "sp3_minus_8_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_004, "title": "Speaker Precedes tvalues - 7 Volumes", "suffix": "sp4_minus_7_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_005, "title": "Speaker Precedes tvalues - 6 Volumes", "suffix": "sp5_minus_6_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_006, "title": "Speaker Precedes tvalues - 5 Volumes", "suffix": "sp6_minus_5_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_007, "title": "Speaker Precedes tvalues - 4 Volumes", "suffix": "sp7_minus_4_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_008, "title": "Speaker Precedes tvalues - 3 Volumes", "suffix": "sp8_minus_3_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_009, "title": "Speaker Precedes tvalues - 2 Volumes", "suffix": "sp9_minus_2_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_010, "title": "Speaker Precedes tvalues - 1 Volume", "suffix": "sp10_minus_1_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_013, "title": "Synchronous at time 0 tvalues", "suffix": "synchronous_3_0_spmT.nii.gz", "type": "beta"},
    {"img": tvalues_img_nifti_contrast_0001, "title": "Listener Precedes tvalues - 1 Volume", "suffix": "lp1_pos_1_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_0002, "title": "Listener Precedes tvalues - 2 Volumes", "suffix": "lp2_pos_2_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_0003, "title": "Listener Precedes tvalues - 3 Volumes", "suffix": "lp3_pos_3_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_0004, "title": "Listener Precedes tvalues - 4 Volumes", "suffix": "lp4_pos_4_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_0005, "title": "Listener Precedes tvalues - 5 Volumes", "suffix": "lp5_pos_5_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_0006, "title": "Listener Precedes tvalues - 6 Volumes", "suffix": "lp6_pos_6_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_0007, "title": "Listener Precedes tvalues - 7 Volumes", "suffix": "lp7_pos_7_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_0008, "title": "Listener Precedes tvalues - 8 Volumes", "suffix": "lp8_pos_8_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_0009, "title": "Listener Precedes tvalues - 9 Volumes", "suffix": "lp9_pos_9_spmT.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_0010, "title": "Listener Precedes tvalues - 10 Volumes", "suffix": "lp10_pos_10_spmT.nii.gz", "type": "tvalue"}]


story_str=str(story)
print(story_str)


for contrast_type in ["beta", "tvalue"]:
    os.makedirs(os.path.join(output_file_path,'story'+story_str+'_normalized','test'), exist_ok=True)
    os.makedirs(os.path.join(output_file_path, f"individuallevel_contrast_{contrast_type}_map_images", 'story'+story_str+'_normalized'),exist_ok=True)

# Loop through contrasts
for contrast in  contrasts:
    print(contrast)
    normalized_dir = os.path.join(output_file_path, 'story'+story_str+'_normalized')
    image_dir = os.path.join(output_file_path, f"individuallevel_contrast_{contrast['type']}_map_images", 'story'+story_str+'_normalized')

    # Plot and save figure
    fig = plot_stat_map(
        contrast["img"],
        bg_img=mni_2mm_brain_path,
        title=contrast["title"],
        display_mode="ortho",
        cut_coords=(0, 0, 0),
        cmap="viridis"
    )
    plt.show()

    # Save figure and NIfTI
    nii_name = nii_prefix + nii_story + '_' + contrast["suffix"]
    fig_path = os.path.join(image_dir, nii_name.split('.')[0] + '.png')
    fig.savefig(fig_path)
    nib.save(contrast["img"], os.path.join(normalized_dir, nii_name))
    
    
    

output_params = [
    # Betas
    {"img": beta_img_nifti_contrast_01, "title": "Speaker Precedes > Listener Precedes", "suffix": "speaker_precedes_minus_listener_precedes_con_0001.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_03, "title": "Listener Precedes > Speaker Precedes", "suffix": "listener_precedes_minus_speaker_precedes_con_0003.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_05, "title": "Listener Precedes > Baseline", "suffix": "listener_precedes_minus_baseline_con_0005.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_06, "title": "Speaker Precedes > Baseline", "suffix": "speaker_precedes_minus_baseline_con_0006.nii.gz", "type": "beta"},
    {"img": beta_img_nifti_contrast_12, "title": "Synchronous at time 0 > Baseline", "suffix": "synchronous_t0_minus_baseline_con_0012.nii.gz", "type": "beta"},
    # T-values
    {"img": tvalues_img_nifti_contrast_01, "title": "Speaker Precedes > Listener Precedes", "suffix": "speaker_precedes_minus_listener_precedes_spmT_0001.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_03, "title": "Listener Precedes > Speaker Precedes", "suffix": "listener_precedes_minus_speaker_precedes_spmT_0003.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_05, "title": "Listener Precedes > Baseline", "suffix": "listener_precedes_minus_baseline_spmT_0005.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_06, "title": "Speaker Precedes > Baseline", "suffix": "speaker_precedes_minus_baseline_spmT_0006.nii.gz", "type": "tvalue"},
    {"img": tvalues_img_nifti_contrast_12, "title": "Synchronous at time 0 > Baseline", "suffix": "synchronous_at_t0_minus_baseline_spmT_0012.nii.gz", "type": "tvalue"}]


    # {"img": tvalues_img_nifti_contrast_07, "title": "Synchronous > Baseline", "suffix": "synchronous_minus_baseline_spmT_0007.nii.gz", "type": "tvalue"},
    # {"img": tvalues_img_nifti_contrast_08, "title": "Speaker Precedes > Synchronous", "suffix": "speaker_precedes_minus_synchronous_spmT_0008.nii.gz", "type": "tvalue"},
    # {"img": tvalues_img_nifti_contrast_09, "title": "Synchronous > Speaker Precedes", "suffix": "synchronous_minus_speaker_precedes_spmT_0009.nii.gz", "type": "tvalue"},
    # {"img": tvalues_img_nifti_contrast_10, "title": "Listener Precedes > Synchronous", "suffix": "listener_precedes_minus_synchronous_spmT_0010.nii.gz", "type": "tvalue"},
    # {"img": tvalues_img_nifti_contrast_11, "title": "Synchronous > Listener Precedes", "suffix": "synchronous_minus_listener_precedes_spmT_0011.nii.gz", "type": "tvalue"},



    # {"img": beta_img_nifti_contrast_07, "title": "Synchronous > Baseline", "suffix": "synchronous_minus_baseline_con_0007.nii.gz", "type": "beta"},
    # {"img": beta_img_nifti_contrast_08, "title": "Speaker Precedes > Synchronous", "suffix": "speaker_precedes_minus_synchronous_con_0008.nii.gz", "type": "beta"},
    # {"img": beta_img_nifti_contrast_09, "title": "Synchronous > Speaker Precedes", "suffix": "synchronous_minus_speaker_precedes_con_0009.nii.gz", "type": "beta"},
    # {"img": beta_img_nifti_contrast_10, "title": "Listener Precedes > Synchronous", "suffix": "listener_precedes_minus_synchronous_con_0010.nii.gz", "type": "beta"},
    # {"img": beta_img_nifti_contrast_11, "title": "Synchronous > Listener Precedes", "suffix": "synchronous_minus_listener_precedes_con_0011.nii.gz", "type": "beta"},



for contrast_type in ["beta", "tvalue"]:
    print(contrast_type)
    
    os.makedirs(os.path.join(output_file_path,'story'+story_str+'_normalized','orig_cons'), exist_ok=True)
    os.makedirs(os.path.join(output_file_path, f"individuallevel_contrast_{contrast_type}_map_images", 'story'+story_str+'_normalized','orig_cons'),exist_ok=True)

for output in output_params:
    print(output)
    normalized_dir = os.path.join(output_file_path, 'story'+story_str+'_normalized','orig_cons')
    image_dir = os.path.join(output_file_path, f"individuallevel_contrast_{output['type']}_map_images", 'story'+story_str+'_normalized','orig_cons')

    fig = plot_stat_map(
        output["img"],
        bg_img=mni_2mm_brain_path,
        title=output["title"],
        display_mode="ortho",
        cut_coords=(0, 0, 0),
        cmap="viridis"
    )
    plt.show()

    nii_name = nii_prefix + nii_story + '_' + output["suffix"]
    fig_path = os.path.join(image_dir, nii_name.split('.')[0] + '.png')
    fig.savefig(fig_path)
    nib.save(output["img"], os.path.join(normalized_dir, nii_name))


# # ### Speaker Precedes - Listener Precedes Con 0001

# # In[17]:


# fig=plot_stat_map(beta_img_nifti_contrast_01, bg_img=mni_2mm_brain_path, title="Speaker Precedes > Listener Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# # if not os.path.exists(output_file_path):
# #     os.mkdir(output_file_path)
# nii_name=nii_prefix+nii_story+'_speaker_precedes_minus_listener_precedes_con_0001.nii.gz'
# fig_path = os.path.join(output_file_path, 'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# fig.savefig(fig_path)

# nib.save(beta_img_nifti_contrast_01, os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # ### Listener Precedes - Speaker Precedes Con 0003

# # In[18]:


# fig=plot_stat_map(beta_img_nifti_contrast_03, bg_img=mni_2mm_brain_path, title="Listener Precedes - Speaker Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# # if not os.path.exists(output_file_path):
# #     os.mkdir(output_file_path)

# nii_name=nii_prefix+nii_story+'_listener_precedes_minus_speaker_precedes_con_0003.nii.gz'
# fig_path = os.path.join(output_file_path, 'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# fig.savefig(fig_path)
# nib.save(beta_img_nifti_contrast_03,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # ### Listener Precedes - Baseline Con 0005

# # In[19]:

# fig=plot_stat_map(beta_img_nifti_contrast_05, bg_img=mni_2mm_brain_path, title="Listener Precedes - Baseline", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# # if not os.path.exists(output_file_path):
# #     os.mkdir(output_file_path)

# nii_name=nii_name=nii_prefix+nii_story+'_listener_precedes_minus_baseline_con_0005.nii.gz'
# fig_path = os.path.join(output_file_path, 'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# fig.savefig(fig_path)

# nib.save(beta_img_nifti_contrast_05, os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # ### Speaker Precedes - Baseline Con 0006

# # In[20]:



# fig=plot_stat_map(beta_img_nifti_contrast_06, bg_img=mni_2mm_brain_path, title="Speaker Precedes - Baseline", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# # if not os.path.exists(output_file_path):
# #     os.mkdir(output_file_path)
# nii_name=nii_name=nii_prefix+nii_story+'_speaker_precedes_minus_baseline_con_0006.nii.gz'
# fig_path = os.path.join(output_file_path,'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# fig.savefig(fig_path)
# nib.save(beta_img_nifti_contrast_06, os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))

# # # ### Synchronous - Baseline Con 0007

# # # In[21]:


# # fig=plot_stat_map(beta_img_nifti_contrast_07, bg_img=mni_2mm_brain_path, title="Synchronous - Baseline", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)

# # nii_name=nii_name=nii_name=nii_prefix+nii_story+'_sychronous_minus_baseline_con_0007.nii.gz'
# # plt.savefig(os.path.join(output_file_path,'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png'))


# # nib.save(beta_img_nifti_contrast_07,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))
# # fig.savefig(fig_path)


# # # ### Speaker precedes - Synchronous Con 0008

# # # In[22]:


# # fig=plot_stat_map(beta_img_nifti_contrast_08, bg_img=mni_2mm_brain_path, title="Speaker Precedes - Synchronous", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)
# # nii_name=nii_name=nii_prefix+nii_story+'_speaker_precedes_minus_sychronous_con_0008.nii.gz'
# # fig_path = os.path.join(output_file_path,'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# # fig.savefig(fig_path)
# # nib.save(beta_img_nifti_contrast_08, os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))

# # # ### Synchronous vs Speaker Precedes Con 0009
# # # 

# # # In[23]:


# # fig=plot_stat_map(beta_img_nifti_contrast_09, bg_img=mni_2mm_brain_path, title="Synchronous - Speaker Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)

# # nii_name=nii_name=nii_prefix+nii_story+'_synchronous_minus_speaker_precedes_con_0009.nii.gz'
# # fig_path = os.path.join(output_file_path, 'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# # fig.savefig(fig_path)
# # nib.save(beta_img_nifti_contrast_09, os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # # ### Listener precedes - Synchronous Con 0010

# # # In[24]:


# # fig=plot_stat_map(beta_img_nifti_contrast_10, bg_img=mni_2mm_brain_path, title="Synchronous - Speaker Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)

# # nii_name=nii_name=nii_prefix+nii_story+'_listener_precedes_minus_synchronous_con_0010.nii.gz'
# # fig_path = os.path.join(output_file_path, 'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# # fig.savefig(fig_path)
# # nib.save(beta_img_nifti_contrast_10, os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # # ### Synchronous vs Listener Precedes Con 0011
# # # 

# # # In[25]:


# # fig=plot_stat_map(beta_img_nifti_contrast_11, bg_img=mni_2mm_brain_path, title="Synchronous - Listener Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)

# # nii_name=nii_name=nii_prefix+nii_story+'_synchronous_minus_speaker_precedes_con_0011.nii.gz'
# # fig_path = os.path.join(output_file_path,'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# # fig.savefig(fig_path)
# # nib.save(beta_img_nifti_contrast_11,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))



# # ### Synchronous time 0 vs baseline Con 0012

# # In[62]:


# fig=plot_stat_map(beta_img_nifti_contrast_12, bg_img=mni_2mm_brain_path, title="Synchronous at time 0 > Baseline", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# if not os.path.exists(output_file_path):
#     os.mkdir(output_file_path)

# nii_name=nii_prefix+nii_story+'_synchronous_t0_minus_baseline_con_0012.nii.gz'
# fig_path = os.path.join(output_file_path,'individuallevel_contrast_beta_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# fig.savefig(fig_path)

# nib.save(beta_img_nifti_contrast_12,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))



# # # Tvalues figures

# # ### Speaker Precedes - Listener Precedes SPMT 0001

# # In[63]:


# fig=plot_stat_map(tvalues_img_nifti_contrast_01, bg_img=mni_2mm_brain_path, title="Speaker Precedes > Listener Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# # if not os.path.exists(output_file_path):
# #     os.mkdir(output_file_path)
# nii_name=nii_prefix+nii_story+'_speaker_precedes_minus_listener_precedes_spmT_0001.nii.gz'
# fig_path = os.path.join(output_file_path,'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')


# fig.savefig(fig_path)

# # Save the NIfTI image to disk
# nib.save(tvalues_img_nifti_contrast_01,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))

# # ### Listener Precedes - Speaker Precedes SPMT 0003
# # 

# # In[ ]:

# fig=plot_stat_map(tvalues_img_nifti_contrast_03, bg_img=mni_2mm_brain_path, title="Listener Precedes > Speaker Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# # if not os.path.exists(output_file_path):
# #     os.mkdir(output_file_path)

# nii_name=nii_prefix+nii_story+'_listener_precedes_minus_speaker_precedes_spmT_0003.nii.gz'
# fig_path = os.path.join(output_file_path,'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# fig.savefig(fig_path)
# nib.save(tvalues_img_nifti_contrast_03,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # ### Listener Precedes - Baseline SPMT 0005

# # In[ ]:
# fig=plot_stat_map(tvalues_img_nifti_contrast_05, bg_img=mni_2mm_brain_path, title="Listener Precedes > Baseline", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# # if not os.path.exists(output_file_path):
# #     os.mkdir(output_file_path)

# nii_name=nii_prefix+nii_story+'_listener_precedes_minus_baseline_spmT_0005.nii.gz'
# fig_path = os.path.join(output_file_path,'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# fig.savefig(fig_path)
# nib.save(tvalues_img_nifti_contrast_05, os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # ### Speaker Precedes - Baseline SPMT 0006

# # In[ ]:
# fig=plot_stat_map(tvalues_img_nifti_contrast_06, bg_img=mni_2mm_brain_path, title="Speaker Precedes > Baseline", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# # if not os.path.exists(output_file_path):
# #     os.mkdir(output_file_path)
# nii_name=nii_prefix+nii_story+'_speaker_precedes_minus_baseline_spmT_0006.nii.gz'
# fig_path = os.path.join(output_file_path,'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# fig.savefig(fig_path)
# nib.save(tvalues_img_nifti_contrast_06,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # # ### Synchronous - Baseline SPMT 0007

# # # In[ ]:

# # fig=plot_stat_map(tvalues_img_nifti_contrast_07, bg_img=mni_2mm_brain_path, title="Synchronous > Baseline", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)
# # nii_name=nii_prefix+nii_story+'_synchronous_minus_baseline_spmT_0007.nii.gz'
# # fig_path = os.path.join(output_file_path,'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# # fig.savefig(fig_path)
# # nib.save(tvalues_img_nifti_contrast_07,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))

# # # ### Speaker precedes - Synchronous SPMT 0008

# # # In[ ]:
# # fig=plot_stat_map(tvalues_img_nifti_contrast_08, bg_img=mni_2mm_brain_path, title="Speaker Precedes > Synchronous", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)
# # nii_name=nii_prefix+nii_story+'_speaker_precedes_minus_sychronous_spmT_0008.nii.gz'
# # fig_path = os.path.join(output_file_path,'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# # fig.savefig(fig_path)


# # # Save the NIfTI image to disk
# # nib.save(tvalues_img_nifti_contrast_08,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))

# # # ### Synchronous vs Speaker Precedes SPMT 0009

# # # In[ ]:
# # fig=plot_stat_map(tvalues_img_nifti_contrast_09, bg_img=mni_2mm_brain_path, title="Synchronous > Speaker Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)

# # nii_name=nii_prefix+nii_story+'_synchronous_minus_speaker_precedes_spmT_0009.nii.gz'
# # fig_path = os.path.join(output_file_path,'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# # fig.savefig(fig_path)
# # nib.save(tvalues_img_nifti_contrast_09,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))

# # # 

# # # ### Listener precedes - Synchronous SPMT 0010
# # # 

# # # In[ ]:
# # fig=plot_stat_map(tvalues_img_nifti_contrast_10, bg_img=mni_2mm_brain_path, title="Synchronous > Speaker Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)

# # nii_name=nii_prefix+nii_story+'_listener_precedes_minus_synchronous_spmT_0010.nii.gz'
# # fig_path = os.path.join(output_file_path,'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')
# # fig.savefig(fig_path)
# # nib.save(tvalues_img_nifti_contrast_10,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # # ### Synchronous vs Listener Precedes SPMT 0011

# # # In[ ]:


# # fig=plot_stat_map(tvalues_img_nifti_contrast_11, bg_img=mni_2mm_brain_path, title="Synchronous > Listener Precedes", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# # plt.show()

# # # if not os.path.exists(output_file_path):
# # #     os.mkdir(output_file_path)

# # nii_name=nii_prefix+nii_story+'_synchronous_minus_listener_precedes_spmT_0011.nii.gz'
# # fig_path = os.path.join(output_file_path, 'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# # fig.savefig(fig_path)
# # nib.save(tvalues_img_nifti_contrast_11, os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))


# # ### Synchronous time 0 vs baseline SPMT 0012

# # In[ ]:

# fig=plot_stat_map(tvalues_img_nifti_contrast_12, bg_img=mni_2mm_brain_path, title="Synchronous at time 0 > baseline", display_mode='ortho', cut_coords=(0,0,0), cmap='viridis')
# plt.show()

# # if not os.path.exists(output_file_path):
# #     os.mkdir(output_file_path)

# nii_name=nii_prefix+nii_story+'_synchronous_at_t0_minus_baseline_spmT_0012.nii.gz'
# fig_path = os.path.join(output_file_path, 'individuallevel_contrast_tvalues_map_images','story'+str(story)+'_normalized',nii_name.split('.')[0] + '.png')

# fig.savefig(fig_path)
# nib.save(tvalues_img_nifti_contrast_12,os.path.join(output_file_path,'story'+str(story)+'_normalized',nii_name))

# # In[ ]:




