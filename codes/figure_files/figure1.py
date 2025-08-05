import os
import pandas as pd
from codes.utils import figure1B, figure1C, figure1D, figure1E, figure1FG, figure1H

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'figure1')
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
df = pd.read_csv(os.path.join(table_path, 'context_days_full_table.csv'), index_col=0)
figure1H.plot_figure1h(data_table=df, saving_path=fig_folder, name='Figure1H', saving_formats=['png', 'svg'])
