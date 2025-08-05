import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from codes.utils.misc.fig_saving import save_fig


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
    bin_count_full_data = data_table[cols].groupby(['time_bin', 'context', 'whisker_trial'],
                                                   as_index=False).count()

    # Add here dict for time bin and actual time interval
    x_label_dict = {-1: '10-0', -2: '20-10', -3: '30-20', -4: '40-30',
                    -5: '50-40', -6: '60-50', -7: '70-60', -8: '80-70', -9: '90-80', -10: '100-90',
                    1: '0-10', 2: '10-20', 3: '20-30', 4: '30-40',
                    5: '40-50', 6: '50-60', 7: '60-70', 8: '70-80', 9: '80-90', 10: '90-100'}

    # Figure: lick probability over time at whisker trials around transitions both way

    # Set plot parameters.
    color_palette = [(129 / 255, 0 / 255, 129 / 255), (0 / 255, 135 / 255, 0 / 255)]
    hue_order = ['To W-', 'To W+']
    pplot_size = 5
    lw = 2
    fig_width, fig_height = 3.5, 3.3

    data_to_plot = bin_averaged_full_data.loc[(bin_averaged_full_data.time_bin < 6) &
                                              (bin_averaged_full_data.time_bin > -6)].copy()
    count_to_plot = bin_count_full_data.loc[(bin_count_full_data.time_bin < 6) &
                                            (bin_count_full_data.time_bin > -6)].copy()

    data_to_plot['transition'] = [
        data_to_plot.iloc[i].context.astype(bool) if data_to_plot.iloc[i].whisker_trial == 'first' else not
        data_to_plot.iloc[i].context.astype(bool) for i in range(len(data_to_plot))]
    count_to_plot['transition'] = [
        count_to_plot.iloc[i].context.astype(bool) if count_to_plot.iloc[i].whisker_trial == 'first' else not
        count_to_plot.iloc[i].context.astype(bool) for i in range(len(count_to_plot))]
    data_to_plot['transition'] = data_to_plot['transition'].map({True: 'To W+', False: 'To W-'})
    count_to_plot['transition'] = count_to_plot['transition'].map({True: 'To W+', False: 'To W-'})

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(fig_width, fig_height), height_ratios=[3, 1], sharex=True)

    sns.pointplot(data=data_to_plot, x='time_bin', y='lick_flag', hue='transition', hue_order=hue_order,
                  palette=color_palette, markers='o', ax=ax0)
    sns.barplot(data=count_to_plot, x='time_bin', y='lick_flag', hue='transition', hue_order=hue_order,
                palette=color_palette, legend=False, ax=ax1)

    positions = [tick for tick in ax1.get_xticks()]
    new_label = [x_label_dict[int(i.get_text())] for i in ax1.get_xticklabels()]
    ax0.axvline(x=np.median(positions), ymin=0, ymax=1, c='k', linestyle='--')
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel('Lick Probability')
    ax1.xaxis.set_major_locator(FixedLocator(positions))
    ax1.set_xlabel('Time around transition (s)')
    ax1.set_xticklabels(new_label, rotation=30)
    ax1.set_ylim(0, 1200)
    ax1.set_yticks(np.arange(0, 1200, 200))
    ax1.set_ylabel('Number of trials')
    sns.despine()
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)
