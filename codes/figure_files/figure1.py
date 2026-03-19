import os
import pandas as pd
from codes.utils import figure1B, figure1C, figure1D, figure1E, figure1FG, figure1H, figure1IJ

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'figure1')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)
fig_supp_folder = os.path.join(main_dir, 'figures', 'supplementary')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 1B
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1', '1B')
df = pd.read_csv(os.path.join(table_path, 'concatenated_bhv_tables.csv'), index_col=0)
figure1B.plot_figure1b(data_table=df, saving_path=fig_folder, name='Figure1B', saving_formats=['png', 'svg'])

# 1C
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1', '1C')
df = pd.read_csv(os.path.join(table_path, 'context_days_full_table.csv'), index_col=0)
figure1C.plot_figure1c(data_table=df, saving_path=fig_folder, name='Figure1C', saving_formats=['png', 'svg'])

# 1D
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1', '1D')
df = pd.read_csv(os.path.join(table_path, 'context_days_full_table.csv'), index_col=0)
figure1D.plot_figure1d(data_table=df, saving_path=fig_folder, name='Figure1D', saving_formats=['png', 'svg'])

# 1E
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1', '1E')
df = pd.read_csv(os.path.join(table_path, 'context_days_full_table.csv'), index_col=0)
figure1E.plot_figure1e(data_table=df, saving_path=fig_folder, name='Figure1E', saving_formats=['png', 'svg'])

# 1F-G
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1', '1FG')
df = pd.read_csv(os.path.join(table_path, 'context_transitions_averaged_table.csv'), index_col=0)
figure1FG.plot_figure1fg(data_table=df, saving_path=fig_folder, name='Figure1FG', saving_formats=['png', 'svg'])

# 1H
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1', '1H')
df = pd.read_csv(os.path.join(table_path, 'whisker_transitions_table.csv'), index_col=0)
figure1H.plot_figure1h(data_table=df, saving_path=fig_folder, name='Figure1H', saving_formats=['png', 'svg'])


# 1IJ
dset_list = ['dlc_jrgeco', 'dlc_gcamp', 'dlc_controls_tdtomato', 'dlc_controls_gfp']
side_dlc_list = []
top_dlc_list = []
data_path = os.path.join(main_dir, 'data', 'figure1', '1IJ')
for dset in dset_list:
    sub_data_path = os.path.join(data_path, dset)
    side_dlc = pd.read_csv(os.path.join(sub_data_path, 'uncentered_side_dlc_results.csv'))
    top_dlc = pd.read_csv(os.path.join(sub_data_path, 'uncentered_top_dlc_results.csv'))
    side_dlc = side_dlc.drop('Unnamed: 0', axis=1)
    top_dlc = top_dlc.drop('Unnamed: 0', axis=1)
    side_dlc_list.append(side_dlc)
    top_dlc_list.append(top_dlc)
full_side_dlc = pd.concat(side_dlc_list)
full_top_dlc = pd.concat(top_dlc_list)

figure1IJ.plot_baseline_differences(side_dlc_data=full_side_dlc,
                                    top_dlc_data=full_top_dlc,
                                    save_path=fig_folder,
                                    supp_save_path=fig_supp_folder,
                                    figname='Figure1',
                                    fig_formats=['png', 'svg'])

