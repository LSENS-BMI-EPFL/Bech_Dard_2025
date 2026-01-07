import os
import pandas as pd
from codes.utils import figure2CDE, figure2F

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'figure2')
supp_fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure2_supp1')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 2CD
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2', '2CDE')
df = pd.read_csv(os.path.join(table_path, 'optogrid_data_table_VGAT.csv'), index_col=0)
trial_df = pd.read_csv(os.path.join(table_path, 'trial_data_table_VGAT.csv'), index_col=0)
figure2CDE.figure2cde(data_table=df, trial_table=trial_df, saving_path=fig_folder, supp_saving_path=supp_fig_folder,
                      saving_formats=['png'])

# 2E
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2', '2F')
df = pd.read_csv(os.path.join(table_path, 'piezo_reaction_time.csv'), index_col=0)
figure2F.figure2f(data_table=df, saving_path=fig_folder)

