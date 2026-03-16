import os
import numpy as np
from codes.utils import figure3_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure3_supp2')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# Difference between W+ auditory hits and whisker hits
aud_data_path = os.path.join(main_dir, 'data', 'figure3', '3E_images')
aud_data_dict = np.load(os.path.join(aud_data_path, 'general_data_dict.npy'), allow_pickle=True).item()

wh_data_path = os.path.join(main_dir, 'data', 'figure3', '3F_images')
wh_data_dict = np.load(os.path.join(wh_data_path, 'general_data_dict.npy'), allow_pickle=True).item()

figure3_supp.wf_timecourse_auditory_to_whisker(aud_data=aud_data_dict,
                                               whisker_data=wh_data_dict,
                                               saving_path=fig_folder)

