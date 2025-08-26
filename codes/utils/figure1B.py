import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from codes.utils.misc.fig_saving import save_fig


def plot_figure1b(data_table, saving_path, name, saving_formats):
    # Select session :
    session_table = data_table.loc[data_table.session_id == 'RD068_20241115_125119']
    session_table = session_table.loc[session_table.early_lick == 0]
    session_table = session_table.reset_index(drop=True)

    # Get the blocks :
    switches = np.where(np.diff(session_table.context.values[:]))[0]
    if len(switches) <= 1:
        block_length = switches[0] + 1
    else:
        block_length = min(np.diff(switches))

    # Add  trial info
    session_table['trial'] = session_table.index

    # Add block info
    session_table = session_table.assign(block=session_table['trial'] // block_length)

    # Compute performance average per block
    for outcome, new_col in zip(['outcome_w', 'outcome_a', 'outcome_n', 'correct_choice'],
                                ['hr_w', 'hr_a', 'hr_n', 'correct']):
        session_table[new_col] = session_table.groupby(['block', 'opto_stim'], as_index=False)[outcome].transform('mean')

    # Subsample at one value per block for performance plot
    d = session_table.loc[session_table.early_lick == 0][int(block_length / 2)::block_length]

    # Plot
    raster_marker = 2
    marker_width = 0.5
    fig_width, fig_height = 6, 3
    catch_palette = ['grey', 'black']
    auditory_palette = ['cyan', 'blue']
    whisker_palette = ['bisque', 'orange']
    context_palette = ['darkmagenta', 'green']

    # Plot the performance by block average
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    for outcome, palette in [('hr_n', catch_palette), ('hr_a', auditory_palette), ('hr_w', whisker_palette)]:
        if outcome in d.columns and (not np.isnan(d[outcome].values[:]).all()):
            sns.lineplot(data=d, x='trial', y=outcome, color=palette[1], markeredgecolor=palette[1], ax=ax, marker='o',
                         lw=2, legend=False)

    # Plot the context band
    rewarded_bloc_bool = list(d.context.values[:])
    bloc_limites = np.arange(start=0, stop=len(session_table.index), step=block_length)
    bloc_area_color = [context_palette[1] if i == 1 else context_palette[0] for i in rewarded_bloc_bool]
    if bloc_limites[-1] < len(session_table.index):
        bloc_area = [(bloc_limites[i], bloc_limites[i + 1]) for i in range(len(bloc_limites) - 1)]
        bloc_area.append((bloc_limites[-1], len(session_table.index)))
        if len(bloc_area) > len(bloc_area_color):
            bloc_area = bloc_area[0: len(bloc_area_color)]
        for index, coords in enumerate(bloc_area):
            bloc_color = bloc_area_color[index]
            ax.axvspan(coords[0], coords[1], facecolor=bloc_color, zorder=1, alpha=0.5)

    # Plot the single trials :
    for outcome, color_offset, palette in [('outcome_n', 0.1, catch_palette), ('outcome_a', 0.15, auditory_palette),
                                           ('outcome_w', 0.2, whisker_palette)]:
        if outcome in d.columns and (not np.isnan(d[outcome]).all()):
            for lick_flag, color_index in zip([0, 1], [0, 1]):
                lick_subset = session_table.loc[session_table.lick_flag == lick_flag]
                ax.scatter(x=lick_subset['trial'], y=lick_subset[outcome] - lick_flag - color_offset,
                           color=palette[color_index], marker=raster_marker, linewidths=marker_width)
    sns.despine()
    ax.set_ylim(-0.25, 1.05)
    ax.set_xlim(-1, len(session_table) + 0.05)
    ax.set_ylabel('Lick probability')
    ax.set_xlabel('Trial number')
    ax.set_title('Mouse RD068 - Session 20241115 ')
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)


