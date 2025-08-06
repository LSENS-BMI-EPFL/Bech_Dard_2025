import os
import pandas as pd
from codes.utils import figure2_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure2_supp1')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 1AB
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2_supp', '1AB')
df = pd.read_csv(os.path.join(table_path, 'optogrid_data_table_controls.csv'), index_col=0)
figure2_supp.plot_figure2_supp1ab(data_table=df, saving_path=fig_folder, saving_formats=['png'])

# 1CDE
# LOAD DATA :
muscimol_path = os.path.join(main_dir, 'data', 'figure2_supp', '1CDE', 'muscimol')
ringer_path = os.path.join(main_dir, 'data', 'figure2_supp', '1CDE', 'ringer')
figure2_supp.plot_figure2_supp1cde(muscimol_path, ringer_path, saving_path=fig_folder,
                                   sites=['wS1', 'fpS1', 'RSC'], names=['C', 'D', 'E'], saving_formats=['png', 'svg'])

