import os
import pandas as pd
from codes.utils import figure2CD, figure2E

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'figure2')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 2CD
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2', '2CD')
df = pd.read_csv(os.path.join(table_path, 'optogrid_data_table_VGAT.csv'), index_col=0)
figure2CD.figure2cd(data_table=df, saving_path=fig_folder, saving_formats=['png'])

# 2E
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2', '2E')
df = pd.read_csv(os.path.join(table_path, 'piezo_reaction_time.csv'), index_col=0)
figure2E.figure2e(data_table=df, saving_path=fig_folder)

