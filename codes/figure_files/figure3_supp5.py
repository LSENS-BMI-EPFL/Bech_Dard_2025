import os
import numpy as np
import pandas as pd
from codes.utils import figure3_supp

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure3_supp5')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


# 5A
data_path = os.path.join(main_dir, 'data', 'figure3', '3F_images')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '5A')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3_supp.wf_timecourse_context_switch(data=data_dict, saving_path=result_folder, formats=['png'])

# 5BC
table_path = os.path.join(main_dir, 'data', 'figure3_supp', '5BC')
df = pd.read_pickle(os.path.join(table_path, 'psth_over_blocks.pkl'))
sorted_areas = np.load(os.path.join(table_path, 'sorted_areas.npy'))
figure3_supp.wh_psth_by_trial_index(df, sorted_areas, save_folder=fig_folder,
                                    names=['Figure3_supp5B', 'Figure3_supp5C'], formats=['png', 'svg'], pre_stim=5)

