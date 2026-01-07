import os
import pandas as pd
from codes.utils import figure1_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure1_supp1')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 1A
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1_supp', '1ABC')
df = pd.read_excel(os.path.join(table_path, 'context_expert_sessions.xlsx'), index_col=0)
figure1_supp.plot_figure1_supp1a(data_table=df, saving_path=fig_folder, name='Figure1_supp1A',
                                 saving_formats=['png', 'svg'])

# 1B
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1_supp', '1ABC')
df = pd.read_csv(os.path.join(table_path, 'context_block_duration_expert.csv'), index_col=0)
figure1_supp.plot_figure1_supp1b(data_table=df, saving_path=fig_folder, name='Figure1_supp1B',
                                 saving_formats=['png', 'svg'])

# 1C
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1_supp', '1ABC')
df = pd.read_csv(os.path.join(table_path, 'context_block_duration_expert.csv'), index_col=0)
figure1_supp.plot_figure1_supp1c(data_table=df, saving_path=fig_folder, name='Figure1_supp1C',
                                 saving_formats=['png', 'svg'])

# 1D
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1_supp', '1D')
df = pd.read_csv(os.path.join(table_path, 'context_transitions_averaged_table.csv'), index_col=0)
figure1_supp.plot_figure1_supp1d(data_table=df, saving_path=fig_folder, name='Figure1_supp1D',
                                 saving_formats=['png', 'svg'])

# 1E
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1_supp', '1E')
df = pd.read_csv(os.path.join(table_path, 'mouse_averaged_reaction_time.csv'), index_col=0)
figure1_supp.plot_figure1_supp1e(data_table=df, saving_path=fig_folder, name='Figure1_supp1E',
                                 saving_formats=['png', 'svg'])

# D' and criterion
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure1', '1B')
df = pd.read_csv(os.path.join(table_path, 'concatenated_bhv_tables.csv'), index_col=0)
figure1_supp.dprime_criterion(data_table=df, saving_path=fig_folder,
                              name='Figure1D_supp', saving_formats=['png', 'svg'])
