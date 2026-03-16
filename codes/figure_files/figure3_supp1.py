import os
import numpy as np
import pandas as pd
from codes.utils import figure3F_images, figure3_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure3_supp1')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 1A TdTomato
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3_supp', '1A_tdtomato')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '1A', 'tdTomato')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3F_images.figure3f_images(data=data_dict, saving_path=result_folder, formats=['png'])

# # 1B GCaMP
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

