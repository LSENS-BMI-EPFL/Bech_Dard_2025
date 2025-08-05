import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from codes.utils.misc.fig_saving import save_fig


def plot_figure1e(data_table, saving_path, name, saving_formats):
    # Drop useless columns
    data_table = data_table.drop(['day', 'artificial_day', 'context_rwd_str'], axis=1)

    # Average over session for each mouse
    avg_data_table = data_table.copy()
    avg_data_table = avg_data_table.drop('session_id', axis=1)
    avg_data_table = avg_data_table.groupby(['mouse_id', 'context', 'context_background'], as_index=False).agg(
        np.nanmean)

    color_palette = ['sienna', 'pink']
    pplot_size = 5
    scatter_alpha = 0.5
    scatter_size = 3
    fig_width, fig_height = 3, 2

    # Plot
    trials = ['outcome_a', 'outcome_w', 'outcome_n']
    titles = ['Auditory', 'Whisker', 'Catch']
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)
    for idx, trial in enumerate(trials):
        sns.pointplot(avg_data_table, y=trial, hue='context_background', palette=color_palette, legend=False,
                      ax=axes.flatten()[idx], dodge=0.7, markersize=pplot_size)
        sns.stripplot(avg_data_table, y=trial, hue='context_background', palette=color_palette, legend=False,
                      ax=axes.flatten()[idx], dodge=True, s=scatter_size, alpha=scatter_alpha)
        axes.flatten()[idx].set_title(titles[idx])
        axes.flatten()[idx].set_ylabel('Lick probability')
        axes.flatten()[idx].set_ylim(0, 1.05)
        sns.despine()
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)
