import os
import pandas as pd
import numpy as np
from codes.utils import figure4A_B, figure4C_G

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'figure4')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


# 4A-B
data = pd.read_json(os.path.join(main_dir, 'data',"figure4A_B_combined_data.json"))
data['value'] = data.value.apply(lambda x: np.asarray(x, dtype=float))
figure4A_B.main(data, output_path = fig_folder)

# 4C-G
figure4C_G

