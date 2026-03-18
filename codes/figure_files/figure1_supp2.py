import os
import pandas as pd
from codes.utils import figure1_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure1_supp2')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 2A
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1_supp', '2A')
trial_table = pd.read_csv(os.path.join(table_path, 'example_trial_table.csv'), index_col=0)

dlc_table_path = os.path.join(main_dir, 'data', 'figure1_supp', '2A')
dlc_data = pd.read_csv(os.path.join(dlc_table_path, 'example_dlc_data.csv'), index_col=0)

figure1_supp.plot_example_traces(dlc_data, trial_table=trial_table, saving_path=fig_folder, name='Figure1_supp2A',
                                 saving_formats=['png', 'svg'])

