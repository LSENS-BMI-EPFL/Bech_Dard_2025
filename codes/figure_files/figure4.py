import os
import pandas as pd
import numpy as np
from codes.utils import figure4A_B, figure4C_G

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'figure3')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


# 4A-B

figure4A_B

# 4C-G
figure4C_G

