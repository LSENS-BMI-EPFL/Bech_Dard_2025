import os
import pandas as pd
import numpy as np
from codes.utils import figure3D, figure3E_images, figure3E_psth, figure3F_images, figure3F_psths, \
    figure3H, figure3G, figure3_supp

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'figure3')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


# 3B
dset_list = ['dlc_jrgeco', 'dlc_gcamp', 'dlc_controls_tdtomato', 'dlc_controls_gfp']
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
                       name='Figure3B',
                       formats=['png', 'svg'])

# 3C
criterion = 3
figure3_supp.dlc_rt(full_side_dlc, threshold=criterion, save_folder=fig_folder,
                    name='Figure3C', formats=['png', 'svg'])


# 3D
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3', '3D')
grid = pd.read_csv(os.path.join(table_path, 'empty_grid.csv'), index_col=0)
df = pd.read_csv(os.path.join(table_path, 'GECO_coordinates_table.csv'), index_col=0)
figure3D.figure3d(data_table=df, grid_template=grid, saving_path=fig_folder)

# 3E images
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3', '3E_images')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '3E')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3E_images.figure3e_images(data=data_dict, saving_path=result_folder, formats=['png'])

# 3E PSTHs
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3', '3E_psths')
df = pd.read_csv(os.path.join(table_path, 'PSTHs_dataset.csv'), index_col=0)
figure3E_psth.figure3e_psth(table=df, saving_path=fig_folder, name='Figure3E_psths', formats=['png', 'svg'])

# 3F images
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3', '3F_images')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '3F')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3F_images.figure3f_images(data=data_dict, saving_path=result_folder, formats=['png'])

# 3F PSTHs
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3', '3F_psths')
df = pd.read_csv(os.path.join(table_path, 'PSTHs_dataset.csv'), index_col=0)
figure3F_psths.figure3f_psth(table=df, saving_path=fig_folder, name='Figure3F_psths', formats=['png', 'svg'])

# 3G
# LOAD DATA :
auditory_path = os.path.join(main_dir, 'data', 'figure3', '3E_psths')
auditory_df = pd.read_csv(os.path.join(auditory_path, 'PSTHs_dataset.csv'), index_col=0)
whisker_path = os.path.join(main_dir, 'data', 'figure3', '3F_psths')
whisker_df = pd.read_csv(os.path.join(whisker_path, 'PSTHs_dataset.csv'), index_col=0)
figure3G.figure3g(auditory_df, whisker_df, saving_path=fig_folder, name='Figure3G', formats=['png', 'svg'])

# 3H barplots
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3', '3H')
df = pd.read_csv(os.path.join(table_path, 'Figure3F_psths_stats.csv'), index_col=0)
figure3H.figure3h_barplots(df, saving_path=fig_folder, name='Figure3H_barplots', formats=['png', 'svg'])

