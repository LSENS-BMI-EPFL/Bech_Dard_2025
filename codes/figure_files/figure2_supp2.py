import os
import pandas as pd
from codes.utils import figure2_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure2_supp2')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)

# 2A
# LOAD DATA :
table_path = os.path.join(main_dir, 'data', 'figure2_supp', '2')
df = pd.read_csv(os.path.join(table_path, 'all_trials_bodyparts_psths.csv'), index_col=0)
figure2_supp.plot_figure2_supp2a(data_table=df, saving_path=fig_folder, name='Figure2_supp2A',
                                 saving_formats=['png', 'svg'])

# 2B
figure2_supp.plot_figure2_supp2b(data_table=df, saving_path=fig_folder, name='Figure2_supp2B',
                                 saving_formats=['png', 'svg'])

# 2C
figure2_supp.plot_figure2_supp2c(data_table=df, saving_path=fig_folder, name='Figure2_supp2C',
                                 saving_formats=['png'])
