import os
import pandas as pd
from codes.utils import figure2_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure2_supp1')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


# 1C
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2', '2CDE')
df = pd.read_csv(os.path.join(table_path, 'optogrid_data_table_VGAT.csv'), index_col=0)
result_folder = os.path.join(fig_folder)
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
figure2_supp.figure2supp_dplick_barplots(data_table=df, saving_path=result_folder, name='Figure2_supp1C',
                                         saving_formats=['png', 'svg'])

# 1DE
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2_supp', '1CD')
df = pd.read_csv(os.path.join(table_path, 'optogrid_data_table_controls.csv'), index_col=0)
figure2_supp.plot_figure2_supp1de(data_table=df, saving_path=fig_folder, saving_formats=['png'])





