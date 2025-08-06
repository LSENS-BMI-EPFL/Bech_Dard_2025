import seaborn as sns
import numpy as np


def ax_set(ax, ylim=None, xlabel='Day', ylabel='Lick probability'):
    """
    Sets the limits and labels of the axes, and removes the top and right spines.

    Args:
        ax: The matplotlib axes to modify.
        ylim: The y-axis limits (default is [-0.1, 1.05]).
        xlabel: The label for the x-axis (default is 'Day').
        ylabel: The label for the y-axis (default is 'Lick probability').
    """
    if ylim is None:
        ylim = [-0.1, 1.05]
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine()


def plot_with_point_and_strip(data, x_name, y_name, hue, palette, ax, palette_key, link_mice):
    """
    Create a pointplot and stripplot on the same axes with consistent formatting.

    Args:
        data: DataFrame containing the data to plot.
        x_name: Name of the column to be used for the x-axis.
        y_name: Name of the column to be used for the y-axis.
        hue: Name of the column to be used for hue (color coding).
        palette: Dictionary containing color palettes for different outcomes.
        ax: Axes object to plot on.
        palette_key: Key for the palette in the palette dictionary.
        link_mice: link data point from each mouse
    """

    sns.pointplot(data=data, x=x_name, y=y_name, hue=x_name if hue is None else hue, palette=palette[palette_key],
                  dodge=True, estimator=np.nanmean, errorbar=('ci', 95), n_boot=1000, ax=ax, linewidth=3)
    # Link mice across days.
    if link_mice:
        mice = data.mouse_id.unique()
        for mouse_id in mice:
            mouse_data = data.loc[data.mouse_id == mouse_id]
            sns.pointplot(x=x_name, y=y_name, data=mouse_data, hue=x_name if hue is None else hue,
                          palette=palette[palette_key], marker=None, estimator=np.nanmean, ax=ax, zorder=5, lw=1)
    else:
        sns.stripplot(x=x_name, y=y_name, data=data, hue=x_name if hue is None else hue, palette=palette[palette_key],
                      dodge=True, marker='o', alpha=.5, ax=ax)
