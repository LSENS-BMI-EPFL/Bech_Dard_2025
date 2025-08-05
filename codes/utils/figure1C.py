import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.table_saving import save_table


def plot_figure1c(data_table, saving_path, name, saving_formats):
    # Remove useless columns
    data_table = data_table.drop(['day', 'artificial_day', 'context_background'], axis=1)

    # Average over session for each mouse
    avg_data_table = data_table.copy()
    avg_data_table = avg_data_table.drop('session_id', axis=1)
    avg_data_table = avg_data_table.groupby(['mouse_id', 'context', 'context_rwd_str'], as_index=False).agg(np.nanmean)

    # Set plot parameters.
    color_palette = [(129 / 255, 0 / 255, 129 / 255), (0 / 255, 135 / 255, 0 / 255)]
    pplot_size = 5
    scatter_alpha = 0.5
    scatter_size = 3
    fig_width, fig_height = 3, 2

    # Plot
    trials = ['outcome_a', 'outcome_w', 'outcome_n']
    titles = ['Auditory', 'Whisker', 'Catch']
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)
    for idx, trial in enumerate(trials):
        sns.pointplot(avg_data_table, y=trial, hue='context_rwd_str', palette=color_palette, legend=False,
                      ax=axes.flatten()[idx], dodge=0.7, markersize=pplot_size)
        sns.stripplot(avg_data_table, y=trial, hue='context_rwd_str', palette=color_palette, legend=False,
                      ax=axes.flatten()[idx], dodge=True, s=scatter_size, alpha=scatter_alpha)
        axes.flatten()[idx].set_title(titles[idx])
        axes.flatten()[idx].set_ylabel('Lick probability')
        axes.flatten()[idx].set_ylim(0, 1.05)
        sns.despine()
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)

    # Table with averaged values
    gran_average_df = avg_data_table.copy()
    gran_average_df = gran_average_df.drop('mouse_id', axis=1)
    gran_average_df = gran_average_df.groupby(['context', 'context_rwd_str'], as_index=False).agg(np.nanmean)
    gran_average_df.rename(columns={'outcome_a': 'auditory_mean',
                                    'outcome_w': 'whisker_mean',
                                    'outcome_n': 'catch_mean'}, inplace=True)

    gran_std_df = avg_data_table.copy()
    gran_std_df = gran_std_df.drop('mouse_id', axis=1)
    gran_std_df = gran_std_df.groupby(['context', 'context_rwd_str'], as_index=False).agg(np.nanstd)
    gran_std_df.rename(columns={'outcome_a': 'auditory_std',
                                'outcome_w': 'whisker_std',
                                'outcome_n': 'catch_std'}, inplace=True)
    stats_df = pd.merge(gran_average_df, gran_std_df, left_on=['context_rwd_str', 'context'],
                        right_on=['context_rwd_str', 'context'], how='right')
    save_table(df=stats_df, saving_path=saving_path, name='Figure1C_averaged_values', format=['csv'])

