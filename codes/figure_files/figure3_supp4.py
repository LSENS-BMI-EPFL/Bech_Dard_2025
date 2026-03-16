import os
import pandas as pd
from codes.utils import figure3_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure3_supp4')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

dset_list = ['dlc_jrgeco', 'dlc_gcamp', 'dlc_controls_tdtomato', 'dlc_controls_gfp']
fig_name = 'Figure3_supp4'
side_dlc_list = []
top_dlc_list = []
data_path = os.path.join(main_dir, 'data', 'figure3', '3BC')
for dset in dset_list:
    sub_data_path = os.path.join(data_path, dset)
    side_dlc = pd.read_csv(os.path.join(sub_data_path, 'side_dlc_results.csv'))
    top_dlc = pd.read_csv(os.path.join(sub_data_path, 'top_dlc_results.csv'))
    side_dlc = side_dlc.drop('Unnamed: 0', axis=1)
    top_dlc = top_dlc.drop('Unnamed: 0', axis=1)
    side_dlc_list.append(side_dlc)
    top_dlc_list.append(top_dlc)

full_side_dlc = pd.concat(side_dlc_list)
full_top_dlc = pd.concat(top_dlc_list)
figure3_supp.dlc_psths(full_side_dlc, full_top_dlc, save_folder=fig_folder,
                       name=fig_name,
                       formats=['png', 'svg'])

