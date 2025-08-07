import os
import numpy as np
from codes.utils import figure3F


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure3_supp2')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 2A
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3_supp', '2A')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '2A')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3F.figure3f(data=data_dict, saving_path=result_folder, formats=['png'])


# 2B
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3_supp', '2B')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '2B')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3F.figure3f(data=data_dict, saving_path=result_folder, formats=['png'], scale=(-0.015, 0.015), halfrange=0.01)

