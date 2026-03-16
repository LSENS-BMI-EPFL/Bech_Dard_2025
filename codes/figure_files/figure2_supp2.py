import os
import pandas as pd
from codes.utils import figure2_supp


# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'supplementary', 'figure2_supp2')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


# 1EFG
# LOAD DATA :
muscimol_path = os.path.join(main_dir, 'data', 'figure2_supp', '2ABC', 'muscimol')
ringer_path = os.path.join(main_dir, 'data', 'figure2_supp', '2ABC', 'ringer')
figure2_supp.plot_figure2_supp2abc(muscimol_path, ringer_path, saving_path=fig_folder,
                                   sites=['wS1', 'fpS1', 'RSC'], names=['A', 'B', 'C'], saving_formats=['png', 'svg'])

