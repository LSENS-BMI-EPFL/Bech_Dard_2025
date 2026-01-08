import os
import numpy as np
import matplotlib.pyplot as plt
from codes.utils.misc.plot_on_grid import plot_grid_on_allen


def figure3d(data_table, grid_template, saving_path):
    sensory_mass_avg = data_table.copy().drop('Session', axis=1).groupby(['Area', 'Mouse'],
                                                                         as_index=False).agg('mean')
    full_avg = sensory_mass_avg.copy().drop('Mouse', axis=1).groupby(['Area'], as_index=False).agg('mean')
    grid_template.drop(grid_template[(grid_template.x == 5.5) & (grid_template.y == 2.5)].index, inplace=True)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_grid_on_allen(grid_template, outcome='dff0', palette=None, facecolor='white', edgecolor='black',
                       result_path=None, dotsize=500, vmin=-1, vmax=1, norm=None, fig=fig, ax=ax)
    c = {'A1': 'cyan', 'wS1': 'darkorange', 'wS2': 'orange'}
    full_avg = full_avg.rename(columns={'AP': 'y', 'ML': 'x'})
    for area in full_avg['Area'].unique():
        df = full_avg.loc[full_avg['Area'] == area]
        df_plot = df.copy()
        plot_grid_on_allen(df_plot, outcome='dff0', palette=None, facecolor=c.get(area), edgecolor='black',
                           result_path=None, dotsize=80, dotmarker='*', dotzorder=5, vmin=-1, vmax=1, norm=None,
                           fig=fig, ax=ax)
    fig.tight_layout()

    for ext in ['png']:
        fig.savefig(os.path.join(saving_path, f'Figure3D.{ext}'), dpi=400)

    plt.close('all')
