import os
import pandas as pd
from codes.utils import figure3_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure3_supp3')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 3AB
table_path = os.path.join(main_dir, 'data', 'figure3', '3F_psths')
df = pd.read_csv(os.path.join(table_path, 'PSTHs_dataset.csv'), index_col=0)
figure3_supp.figure3_supp3ab(df, saving_path=fig_folder, name='Figure3_supp3', formats=['png', 'svg'])

