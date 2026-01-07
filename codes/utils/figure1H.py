import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.stats import analyze_both_transitions
from codes.utils.misc.table_saving import df_to_latex, save_table
pd.set_option("display.float_format", "{:.2e}".format)


def plot_figure1h(data_table, saving_path, name, saving_formats):
    # Add a column for time around transition (all possibilities)
    cols_to_merge = ['time_in_reward', 'time_in_non_reward', 'time_to_non_reward', 'time_to_reward']
    data_table['Time around transition'] = data_table[cols_to_merge].bfill(axis=1).iloc[:, 0]

    # Add a 10sec bin for each trial
    transition_trial_times = data_table['Time around transition'].values[:]
    bins = np.arange(-300, 300, 10)
    digitized = np.digitize(transition_trial_times, bins=bins) - 1
    symmetrical_digitized = np.where(transition_trial_times < 0, -(len(bins) // 2 - digitized),
                                     digitized - len(bins) // 2 + 1)
    data_table['time_bin'] = symmetrical_digitized

    # Average and count by time bin
    cols = ['time_bin', 'lick_flag', 'context', 'whisker_trial']
    bin_averaged_full_data = data_table[cols].groupby(['time_bin', 'context', 'whisker_trial'],
                                                      as_index=False).mean(numeric_only=True)

    # Add here dict for time bin and actual time interval
    x_label_dict = {-1: '-10', -2: '-20', -3: '-30', -4: '-40',
                    -5: '-50', -6: '-60', -7: '-70', -8: '-80', -9: '-90', -10: '-100',
                    1: '10', 2: '20', 3: '30', 4: '40',
                    5: '50', 6: '60', 7: '70', 8: '80', 9: '90', 10: '100'}

    # Figure: lick probability over time at whisker trials around transitions both way

    # Set plot parameters.
    color_palette = [(129 / 255, 0 / 255, 129 / 255), (0 / 255, 135 / 255, 0 / 255)]
    hue_order = ['To W-', 'To W+']

    data_to_plot = bin_averaged_full_data.loc[(bin_averaged_full_data.time_bin < 6) &
                                              (bin_averaged_full_data.time_bin > -6)].copy()
    data_to_plot['transition'] = [
        data_to_plot.iloc[i].context.astype(bool) if data_to_plot.iloc[i].whisker_trial == 'first' else not
        data_to_plot.iloc[i].context.astype(bool) for i in range(len(data_to_plot))]
    data_to_plot['transition'] = data_to_plot['transition'].map({True: 'To W+', False: 'To W-'})

    fig, ax0 = plt.subplots(1, 1, figsize=(5, 3))

    sns.pointplot(data=data_to_plot, x='time_bin', y='lick_flag', hue='transition', hue_order=hue_order,
                  palette=color_palette, markers='o', ax=ax0)

    positions = [tick for tick in ax0.get_xticks()]
    new_label = [x_label_dict[int(i.get_text())] for i in ax0.get_xticklabels()]
    ax0.axvline(x=np.median(positions), ymin=0, ymax=1, c='k', linestyle='--')
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel('Lick Probability')
    ax0.xaxis.set_major_locator(FixedLocator(positions))
    ax0.set_xlabel('Time around block switch (s)')
    ax0.set_xticklabels(new_label, rotation=30)
    sns.despine()
    fig.tight_layout()

    results, fit_fig = analyze_both_transitions(data_to_plot, plot=True)

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)
    save_fig(fit_fig, saving_path, figure_name=f'{name}_fit', formats=saving_formats)

    # Convert to LaTeX with booktabs and improved names
    stats_df = data_to_plot.drop(['whisker_trial', 'context'], axis=1)
    stats_df.rename(columns={'lick_flag': 'P (lick)',
                             'transition': 'Transition',
                             'time_bin': 'Time bin'}, inplace=True)
    df_to_latex(df=stats_df, filename=os.path.join(saving_path, 'Figure1H_table.tex'), caption='Figure1H', label='')
    save_table(stats_df, saving_path, f'{name}_stat_results', format=['csv'])
