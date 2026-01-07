import os
import pandas as pd
from codes.utils import figure2_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure2_supp1')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 1CD
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2_supp', '1CD')
df = pd.read_csv(os.path.join(table_path, 'optogrid_data_table_controls.csv'), index_col=0)
figure2_supp.plot_figure2_supp1cd(data_table=df, saving_path=fig_folder, saving_formats=['png'])

# 1EFG
# LOAD DATA :
muscimol_path = os.path.join(main_dir, 'data', 'figure2_supp', '1EFG', 'muscimol')
ringer_path = os.path.join(main_dir, 'data', 'figure2_supp', '1EFG', 'ringer')
figure2_supp.plot_figure2_supp1efg(muscimol_path, ringer_path, saving_path=fig_folder,
                                   sites=['wS1', 'fpS1', 'RSC'], names=['E', 'F', 'G'], saving_formats=['png', 'svg'])

# 1H
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2_supp', '1H')
df = pd.read_csv(os.path.join(table_path, 'all_trials_bodyparts_psths.csv'), index_col=0)

# 1H jaw trace average
figure2_supp.plot_figure2_supp1h_tc(data_table=df, saving_path=fig_folder, name='Figure2_supp1H_tc',
                                    saving_formats=['png', 'svg'])
# 1H AUC barplots
figure2_supp.plot_figure2_supp1h_auc(data_table=df, saving_path=fig_folder, name='Figure2_supp1H_auc_bar',
                                     saving_formats=['png', 'svg'])
# 1H grid representation
figure2_supp.plot_figure2_supp1h_grid(data_table=df, saving_path=fig_folder, name='Figure2_supp1H_grid',
                                      saving_formats=['png'])
