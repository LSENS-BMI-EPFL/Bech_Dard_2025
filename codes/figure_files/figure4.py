import os
import pandas as pd
import numpy as np
from codes.utils import figure4A_B, figure4C_G

# Get main data and saving dir
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
fig_folder = os.path.join(main_dir, 'figures', 'figure4')
if not os.path.exists(fig_folder):
    os.makedirs(fig_folder)


# 4A-B
data = pd.read_json(os.path.join(main_dir, 'data', 'figure4', '4AB', 'combined_avg_correlation_results.json'))
data['value'] = data.value.apply(lambda x: np.asarray(x, dtype=float))
figure4A_B.main(data, output_path=fig_folder)

# 4C-G

data_path_4C = os.path.join(main_dir, 'data', 'figure4', '4C')
data_path_4DG = os.path.join(main_dir, 'data', 'figure4', '4DG', 'VGAT')
data_path_4_supp = os.path.join(main_dir, 'data', 'figure4_supp', '2A')
opto_data_path = os.path.join(main_dir, 'data', 'figure2', '2CDE', 'VGAT')

figure4C_G.main(data_path_4C = os.path.join(main_dir, 'data', 'figure4', '4C'),
                data_path_4DG = os.path.join(main_dir, 'data', 'figure4', '4DG', 'VGAT'),
                data_path_4_supp = os.path.join(main_dir, 'data', 'figure4_supp', '2A'), 
                opto_data_path=opto_data_path,
                output_path=fig_folder)

