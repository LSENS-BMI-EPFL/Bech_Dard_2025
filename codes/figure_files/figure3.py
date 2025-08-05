import os
import pandas as pd
import numpy as np
from codes.utils import figure3A, figure3D, figure3E, figure3F, figure3G, figure3I, figure3J

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'figure3')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 3A
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3', '3A')
grid = pd.read_csv(os.path.join(table_path, 'empty_grid.csv'), index_col=0)
df = pd.read_csv(os.path.join(table_path, 'GECO_coordinates_table.csv'), index_col=0)
figure3A.figure3a(data_table=df, grid_template=grid, saving_path=fig_folder)

# 3D
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3', '3D')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '3D')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3D.figure3d(data=data_dict, saving_path=result_folder, formats=['png'])

# 3E
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3', '3E')
df = pd.read_csv(os.path.join(table_path, 'PSTHs_dataset.csv'), index_col=0)
figure3E.figure3e(table=df, saving_path=fig_folder, name='Figure3E', formats=['png', 'svg'])

# 3F
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3', '3F')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '3F')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3F.figure3f(data=data_dict, saving_path=result_folder, formats=['png'])

# 3G
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3', '3G')
df = pd.read_csv(os.path.join(table_path, 'PSTHs_dataset.csv'), index_col=0)
figure3G.figure3g(table=df, saving_path=fig_folder, name='Figure3G', formats=['png', 'svg'])

# 3I
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3', '3I')
df = pd.read_csv(os.path.join(table_path, 'Figure3G_stats.csv'), index_col=0)
figure3I.figure3i(df, saving_path=fig_folder, name='Figure3I', formats=['png', 'svg'])

# 3J
# LOAD DATA :
auditory_path = os.path.join(main_dir, 'data', 'figure3', '3E')
auditory_df = pd.read_csv(os.path.join(auditory_path, 'PSTHs_dataset.csv'), index_col=0)
whisker_path = os.path.join(main_dir, 'data', 'figure3', '3G')
whisker_df = pd.read_csv(os.path.join(whisker_path, 'PSTHs_dataset.csv'), index_col=0)
figure3J.figure3j(auditory_df, whisker_df, saving_path=fig_folder, name='Figure3J', formats=['png', 'svg'])


