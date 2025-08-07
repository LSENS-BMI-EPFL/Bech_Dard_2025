import os
import numpy as np
import pandas as pd
from codes.utils import figure3D, figure3E, figure3F, figure3G, figure3I, figure3J


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure3_supp1')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 1A
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3_supp', '1A')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '1A')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3D.figure3d(data=data_dict, saving_path=result_folder, formats=['png'], scale=(-0.015, 0.015), halfrange=0.01)

# 1B
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3_supp', '1B')
df = pd.read_csv(os.path.join(table_path, 'PSTHs_dataset.csv'), index_col=0)
figure3E.figure3e(table=df, saving_path=fig_folder, name='Figure3_supp1B', formats=['png', 'svg'])

# 1C
# LOAD DATA :
data_path = os.path.join(main_dir, 'data', 'figure3_supp', '1C')
data_dict = np.load(os.path.join(data_path, 'general_data_dict.npy'), allow_pickle=True).item()
result_folder = os.path.join(fig_folder, '1C')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure3F.figure3f(data=data_dict, saving_path=result_folder, formats=['png'], scale=(-0.015, 0.015), halfrange=0.01)

# 1D
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3_supp', '1D')
df = pd.read_csv(os.path.join(table_path, 'PSTHs_dataset.csv'), index_col=0)
figure3G.figure3g(table=df, saving_path=fig_folder, name='Figure3_supp1D', formats=['png', 'svg'])

# 1F
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure3_supp', '1F')
df = pd.read_csv(os.path.join(table_path, 'Figure3_supp1D_stats.csv'), index_col=0)
figure3I.figure3i(df, saving_path=fig_folder, name='Figure3_supp1F', formats=['png', 'svg'])

# 1G
# LOAD DATA :
auditory_path = os.path.join(main_dir, 'data', 'figure3_supp', '1B')
auditory_df = pd.read_csv(os.path.join(auditory_path, 'PSTHs_dataset.csv'), index_col=0)
whisker_path = os.path.join(main_dir, 'data', 'figure3_supp', '1D')
whisker_df = pd.read_csv(os.path.join(whisker_path, 'PSTHs_dataset.csv'), index_col=0)
figure3J.figure3j(auditory_df, whisker_df, saving_path=fig_folder, name='Figure3_supp1G', formats=['png', 'svg'])

