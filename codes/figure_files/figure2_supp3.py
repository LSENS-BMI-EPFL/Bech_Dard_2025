import os
import pandas as pd
from codes.utils import figure2_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure2_supp3')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2_supp', '3ABC')
df = pd.read_csv(os.path.join(table_path, 'all_trials_bodyparts_psths.csv'), index_col=0)

# 3A
figure2_supp.plot_figure2_supp3a(data_table=df, saving_path=fig_folder, name='Figure2_supp3A',
                                 saving_formats=['png', 'svg'])
# 3B
figure2_supp.plot_figure2_supp3b(data_table=df, saving_path=fig_folder, name='Figure2_supp3B',
                                 saving_formats=['png', 'svg'])
# 3C
figure2_supp.plot_figure2_supp3c(data_table=df, saving_path=fig_folder, name='Figure2_supp3C',
                                 saving_formats=['png'])
