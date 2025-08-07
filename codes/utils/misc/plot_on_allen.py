import os
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm
from skimage.transform import rescale
from codes.utils.misc.plot_on_grid import get_colormap, get_allen_ccf, get_wf_scalebar


# UTILS
def save_fig_and_close(results_path, save_formats, figure, figname):
    """
    Saves a figure in multiple formats and closes the figure.

    Args:
        results_path: Base path where figures will be saved.
        save_formats: List of file formats (e.g., ['png', 'pdf']) to save the figure.
        figure: The figure to save.
        figname: The base name for the saved figure files.
    """
    figure.tight_layout()
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for save_format in save_formats:
        figure.savefig(os.path.join(results_path, f'{figname}.{save_format}'),
                       format=f"{save_format}", dpi=500)
    plt.close('all')


def plot_wf_single_frame(frame, title, figure, ax_to_plot, suptitle, saving_path, save_formats, grid=None,
                         griddotsize=500,
                         colormap='hotcold',
                         vmin=-0.005, vmax=0.02, halfrange=0.04, cbar_shrink=1.0, separated_plots=False, nan_c='white'):
    subfig, ax = plt.subplots(1, 1, figsize=(4, 4), frameon=False)

    bregma = (488, 290)
    scale = 4
    scalebar = get_wf_scalebar(scale=scale)
    iso_mask, atlas_mask, allen_bregma = get_allen_ccf(bregma)
    cmap = get_colormap(colormap)
    cmap.set_bad(color=nan_c)

    single_frame = np.rot90(rescale(frame, scale, anti_aliasing=False))
    single_frame = np.pad(single_frame, [(0, 650 - single_frame.shape[0]), (0, 510 - single_frame.shape[1])],
                          mode='constant', constant_values=np.nan)

    mask = np.pad(iso_mask, [(0, 650 - iso_mask.shape[0]), (0, 510 - iso_mask.shape[1])], mode='constant',
                  constant_values=np.nan)
    single_frame = np.where(mask > 0, single_frame, np.nan)

    if colormap == 'hotcold':
        cmap = get_colormap('hotcold')
        cmap.set_bad(color=nan_c)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax_to_plot.imshow(single_frame, norm=norm, cmap=cmap)
        im_2 = ax.imshow(single_frame, norm=norm, cmap=cmap)
    elif colormap == 'seismic':
        norm = colors.CenteredNorm(halfrange=halfrange)
        cmap.set_bad(color=nan_c)
        im = ax_to_plot.imshow(single_frame, cmap="seismic", norm=norm)
        im_2 = ax.imshow(single_frame, cmap="seismic", norm=norm)
    else:
        im = ax_to_plot.imshow(single_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        im_2 = ax.imshow(single_frame, cmap=cmap, vmin=vmin, vmax=vmax)

    # Put the image on main figure
    ax_to_plot.contour(atlas_mask, levels=np.unique(atlas_mask), colors='gray',
                       linewidths=1)
    ax_to_plot.contour(iso_mask, levels=np.unique(np.round(iso_mask)), colors='black',
                       linewidths=2, zorder=2)
    ax_to_plot.scatter(bregma[0], bregma[1], marker='+', c='k', s=100, linewidths=2,
                       zorder=3)
    if nan_c == 'white':
        ax_to_plot.hlines(25, 25, 25 + scalebar * 3, linewidth=2, color='k')
    else:
        ax_to_plot.hlines(25, 25, 25 + scalebar * 3, linewidth=2, color='white')
    ax_to_plot.set_title(title)
    figure.colorbar(im, ax=ax_to_plot, location='right', shrink=cbar_shrink)
    figure.set_facecolor('white')

    if grid is not None:
        grid['ml_wf'] = bregma[0] - grid['x'] * scalebar
        grid['ap_wf'] = bregma[1] - grid['y'] * scalebar
        sns.scatterplot(data=grid, x='ml_wf', y='ap_wf', s=griddotsize, marker='o', facecolor='white', alpha=0.1,
                        edgecolor='grey',
                        ax=ax_to_plot, zorder=1)

    ax_to_plot.set_axis_off()

    # Put on subfig for individual image
    if separated_plots:
        # subfig.suptitle(suptitle)
        # ax.set_title(title)
        if grid is not None:
            sns.scatterplot(data=grid, x='ml_wf', y='ap_wf', s=griddotsize, marker='o', facecolor='white', alpha=0.1,
                            edgecolor='grey',
                            ax=ax, zorder=1)
        ax.contour(atlas_mask, levels=np.unique(atlas_mask), colors='gray', linewidths=1)
        ax.contour(iso_mask, levels=np.unique(np.round(iso_mask)), colors='black', linewidths=2, zorder=2)
        if nan_c == 'white':
            ax.scatter(bregma[0], bregma[1], marker='+', c='k', s=100, linewidths=2, zorder=3)
            ax.hlines(25, 25, 25 + scalebar * 3, linewidth=2, colors='k')
        else:
            ax.scatter(bregma[0], bregma[1], marker='+', c='white', s=100, linewidths=2, zorder=3)
        ax.hlines(25, 25, 25 + scalebar * 3, linewidth=2, color='white')

        ax.set_axis_off()
        ax.set_xticks(ticks=[], labels=None)
        ax.set_yticks(ticks=[], labels=None)
        subfig.colorbar(im_2, ax=ax, location='right', shrink=cbar_shrink)
        save_fig_and_close(results_path=saving_path, save_formats=save_formats, figure=subfig,
                           figname=f'{suptitle}_{title}' if suptitle != ' ' else f'{title}')


# MAIN
def plot_wf_avg(avg_data, output_path, n_frames_post_stim, n_frames_averaged, key, center_frame,
                figname, save_formats, subdir, c_scale=(-0.005, 0.02), halfrange=0.04, colormap='hotcold'):
    path = os.path.join(output_path, f"{subdir}")
    if not os.path.exists(path):
        os.makedirs(path)

    cutlet_idx = 0
    n_cols = int(np.ceil(n_frames_post_stim / n_frames_averaged))
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(5 * n_cols, 5),
                             frameon=False)
    fig.suptitle(f'{key}')
    for start in range(center_frame, center_frame + n_frames_post_stim, n_frames_averaged):
        if start + n_frames_averaged > center_frame + n_frames_post_stim:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            frame_mean = np.nanmean(avg_data[start:start + n_frames_averaged], axis=0)
        plot_wf_single_frame(frame_mean,
                             title=f'{start - center_frame}-{start - center_frame + n_frames_averaged}',
                             figure=fig,
                             ax_to_plot=axes[cutlet_idx],
                             suptitle=key,
                             colormap=colormap,
                             vmin=c_scale[0],
                             vmax=c_scale[1],
                             halfrange=halfrange,
                             cbar_shrink=0.75,
                             saving_path=path,
                             save_formats=save_formats)
        cutlet_idx += 1
    fig.tight_layout()

    save_fig_and_close(results_path=path, save_formats=save_formats, figure=fig,
                       figname=f'{figname}_wf_avg')
