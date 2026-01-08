import os
import numpy as np
import pandas as pd
from codes.utils import figure3F_images, figure3_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure3_supp1')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 1A
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3_supp', '1A')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '1A', 'tdTomato')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3F_images.figure3f_images(data=data_dict, saving_path=result_folder, formats=['png'])

# 1B GCaMP
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3_supp', '1B_gcamp')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '1B', 'GCaMP')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3F_images.figure3f_images(data=data_dict, saving_path=result_folder, formats=['png'],
                                scale=(-0.015, 0.015), halfrange=0.01)

# 1B GFP
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3_supp', '1B_gfp')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '1B', 'GFP')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3F_images.figure3f_images(data=data_dict, saving_path=result_folder, formats=['png'],
                                scale=(-0.015, 0.015), halfrange=0.01)

# REVIEW FIGURES
# Context transitions
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3', '3F_images')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, 'Reviewing', 'context_switch')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3_supp.wf_timecourse_context_switch(data=data_dict, saving_path=result_folder, formats=['png'])


# 3I supp: d' above 2 for each mouse based on PSTHs
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3', '3F_psths')
df = pd.read_csv(os.path.join(table_path, 'PSTHs_dataset.csv'), index_col=0)
result_folder = os.path.join(fig_folder, 'Reviewing')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3_supp.dprime_by_mouse(df, saving_path=result_folder, name='Figure3H_supp', formats=['png', 'svg'])


# PSTHs whisker trials over block center on transition:
table_path = os.path.join(main_dir, 'data', 'figure3_supp', 'Reviewing')
df = pd.read_pickle(os.path.join(table_path, 'psth_over_blocks.pkl'))
sorted_areas = np.load(os.path.join(table_path, 'sorted_areas.npy'))
result_folder = os.path.join(fig_folder, 'Reviewing')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3_supp.wh_psth_by_trial_index(df, sorted_areas, save_folder=result_folder,
                                    name='wh_over_block', formats=['png', 'svg'], pre_stim=5)

# Difference between W+ auditory hits and whisker hits
aud_data_path = os.path.join(main_dir, 'data', 'figure3', '3E_images')
aud_data_dict = np.load(os.path.join(aud_data_path, 'general_data_dict.npy'), allow_pickle=True).item()

wh_data_path = os.path.join(main_dir, 'data', 'figure3', '3F_images')
wh_data_dict = np.load(os.path.join(wh_data_path, 'general_data_dict.npy'), allow_pickle=True).item()

result_folder = os.path.join(fig_folder, 'Reviewing')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

figure3_supp.wf_timecourse_auditory_to_whisker(aud_data=aud_data_dict,
                                               whisker_data=wh_data_dict,
                                               saving_path=result_folder)

