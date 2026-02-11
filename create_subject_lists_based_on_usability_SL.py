# %%
import os 
import pandas as pd
from IPython.display import display, Markdown
# from glob import glob
import numpy as np 
pd.set_option('display.max_rows',100)
pd.set_option('display.min_rows', 100)

import os
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool

import datetime

# %%
import warnings
# Filter out DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

# %%
# data=pd.read_csv('/Users/daelsaid/Library/Application Support/Box/Box Edit/Documents/1746974773687/swgcaor_sl_movementstats2025.csv',dtype='str')
# subject_status=pd.read_csv('/Users/daelsaid/Library/Application Support/Box/Box Edit/Documents/1746974773687/subjectlist_asd_td_status.csv',dtype='str')
data=pd.read_csv('/Users/daelsaid/scripts/asd_speech_analysis/2026_usable_subjects/2026_movement.csv',dtype='str')
subject_status=pd.read_csv('/Users/daelsaid/scripts/asd_speech_analysis/2026_usable_subjects/pid_diagnosis_mapping.csv',dtype='str')


# %%
# subjects_merged_with_group=data.merge(subject_status.reset_index(),on='PID',how='left')
subjects_merged_with_group=data.drop_duplicates()
os.chdir('/Users/daelsaid/scripts/asd_speech_analysis')

# %%
display(subjects_merged_with_group.head())

# %%
for task_name, group in subjects_merged_with_group.groupby('TASK'):
    usable = group[group['usability'] == 'usable']
    not_usable = group[group['usability'] == 'not usable']
    asd_not_usable= not_usable[not_usable['group'] == 'ASD']
    td_not_usable = not_usable[not_usable['group'] == 'TD']
    asd_usable = usable[usable['group'] == 'ASD']
    td_usable = usable[usable['group'] == 'TD']
    os.makedirs(f'./2026_usable_subjects/all_subjects/usable', exist_ok=True)
    os.makedirs(f'./2026_usable_subjects/all_subjects/not_usable', exist_ok=True)
    os.makedirs(f'./2026_usable_subjects/asd_usable', exist_ok=True)
    os.makedirs(f'./2026_usable_subjects/asd_not_usable', exist_ok=True)
    os.makedirs(f'./2026_usable_subjects/td_usable', exist_ok=True)
    os.makedirs(f'./2026_usable_subjects/td_not_usable', exist_ok=True)
    #All subjects
    usable.to_csv(f'./2026_usable_subjects/all_subjects/usable/{task_name}_usable.csv', index=False)
    not_usable.to_csv(f'./2026_usable_subjects/all_subjects/not_usable/{task_name}_not_usable.csv', index=False)
    #ASD
    asd_usable.to_csv(f'./2026_usable_subjects/asd_usable/{task_name}_asd_usable.csv', index=False)
    asd_not_usable.to_csv(f'./2026_usable_subjects/asd_not_usable/{task_name}_asd_not_usable.csv', index=False)
    
    #TD 
    td_not_usable.to_csv(f'./2026_usable_subjects/td_not_usable/{task_name}_td_not_usable.csv', index=False)
    td_usable.to_csv(f'./2026_usable_subjects/td_usable/{task_name}_td_usable.csv', index=False) 
    print(f"Saved {task_name}_usable.csv and {task_name}_not_usable.csv" and f"{task_name}_asd_usable.csv and {task_name}_asd_not_usable.csv" and f"{task_name}_td_usable.csv and {task_name}_td_not_usable.csv")

# %%
display(subjects_merged_with_group.columns)


