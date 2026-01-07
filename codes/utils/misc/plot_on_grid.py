import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize
from skimage.transform import rescale
import seaborn as sns


def plot_grid_on_allen(grid, outcome, palette, facecolor, edgecolor, result_path, dotsize=300, dotmarker='o',
                       dotzorder=1, vmin=-1, vmax=1, norm=None, fig=None, ax=None):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, figsize=(5, 5), dpi=200)
        new_fig = True
    else:
        new_fig = False
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.3)

    if norm == 'two_slope':
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = get_colormap('gray')
    cmap.set_bad(color='white')
    bregma = (488, 290)
    scale = 4
    scalebar = get_wf_scalebar(scale=scale)
    iso_mask, atlas_mask, allen_bregma = get_allen_ccf(bregma)

    grid['ml_wf'] = bregma[0] - grid['x'] * scalebar
    grid['ap_wf'] = bregma[1] - grid['y'] * scalebar

    grid = grid.loc[~(((grid.x == 5.5) & (grid.y == 1.5)) | ((grid.x == 4.5) & (grid.y == 2.5)))]

    single_frame = np.rot90(rescale(np.ones([125, 160]), scale, anti_aliasing=False))
    single_frame = np.pad(single_frame, [(0, 650 - single_frame.shape[0]), (0, 510 - single_frame.shape[1])],
                          mode='constant', constant_values=np.nan)
    im = ax.imshow(single_frame, cmap=cmap, vmin=0, vmax=1)
    if palette is not None:
        g = sns.scatterplot(data=grid, x='ml_wf', y='ap_wf', hue=f'{outcome}', edgecolor=edgecolor,
                            hue_norm=norm, s=dotsize, marker=dotmarker, palette=palette, ax=ax, zorder=dotzorder)
    else:
        g = sns.scatterplot(data=grid, x='ml_wf', y='ap_wf', s=dotsize, marker=dotmarker, facecolor=facecolor,
                            edgecolor=edgecolor, ax=ax, zorder=dotzorder)

    ax.contour(atlas_mask, levels=np.unique(atlas_mask), colors='gray',
               linewidths=1)
    ax.contour(iso_mask, levels=np.unique(np.round(iso_mask)), colors='black',
               linewidths=2, zorder=2)
    # ax.scatter(bregma[0], bregma[1], marker='+', c='r', s=300, linewidths=4,
    #            zorder=3)

    ax.set_xticks(grid.ml_wf.unique(), grid.x.unique())
    ax.set_yticks(grid.ap_wf.unique(), grid.y.unique())
    ax.set_aspect(1)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.set_axis_off()
    if palette is not None:
        ax.get_legend().remove()

    ax.hlines(5, 5, 5 + scalebar * 3, linewidth=2, colors='k')

    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm, cax=cax, orientation='horizontal')

    if new_fig and result_path is not None:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        fig.savefig(result_path + ".png")
        fig.savefig(result_path + ".svg")

    if not new_fig:
        return fig, ax


def get_wf_scalebar(scale=1, plot=False, savepath=None):
    x = [62 * scale, 167 * scale]
    y = [162 * scale, 152 * scale]
    c = np.sqrt((x[1] - x[0]) ** 2 + (y[0] - y[1]) ** 2)
    return round(c / 6)


def get_allen_ccf(bregma=(528, 315),
                  root=os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..", "..")),
                                    'data', 'utils', 'Allen_brain')):

    # All images aligned to 240,175 at widefield video alignment, after expanding image, goes to this. Set manually.
    iso_mask = np.load(root + r"\allen_isocortex_tilted_500x640.npy")
    atlas_mask = np.load(root + r"\allen_brain_tilted_500x640.npy")
    bregma_coords = np.load(root + r"\allen_bregma_tilted_500x640.npy")

    displacement_x = int(bregma[0] - np.round(bregma_coords[0] + 20))
    displacement_y = int(bregma[1] - np.round(bregma_coords[1]))

    margin_y = atlas_mask.shape[0] - np.abs(displacement_y)
    margin_x = atlas_mask.shape[1] - np.abs(displacement_x)

    if displacement_y >= 0 and displacement_x >= 0:
        atlas_mask[displacement_y:, displacement_x:] = atlas_mask[:margin_y, :margin_x]
        atlas_mask[:displacement_y, :] *= 0
        atlas_mask[:, :displacement_x] *= 0

        iso_mask[displacement_y:, displacement_x:] = iso_mask[:margin_y, :margin_x]
        iso_mask[:displacement_y, :] *= 0
        iso_mask[:, :displacement_x] *= 0

    elif displacement_y < 0 <= displacement_x:
        atlas_mask[:displacement_y, displacement_x:] = atlas_mask[-margin_y:, :margin_x]
        atlas_mask[displacement_y:, :] *= 0
        atlas_mask[:, :displacement_x] *= 0

        iso_mask[:displacement_y, displacement_x:] = iso_mask[-margin_y:, :margin_x]
        iso_mask[displacement_y:, :] *= 0
        iso_mask[:, :displacement_x] *= 0

    elif displacement_y >= 0 > displacement_x:
        atlas_mask[displacement_y:, :displacement_x] = atlas_mask[:margin_y, -margin_x:]
        atlas_mask[:displacement_y, :] *= 0
        atlas_mask[:, displacement_x:] *= 0

        iso_mask[displacement_y:, :displacement_x] = iso_mask[:margin_y, -margin_x:]
        iso_mask[:displacement_y, :] *= 0
        iso_mask[:, displacement_x:] *= 0

    else:
        atlas_mask[:displacement_y, :displacement_x] = atlas_mask[-margin_y:, -margin_x:]
        atlas_mask[displacement_y:, :] *= 0
        atlas_mask[:, displacement_x:] *= 0

        iso_mask[:displacement_y, :displacement_x] = iso_mask[-margin_y:, -margin_x:]
        iso_mask[displacement_y:, :] *= 0
        iso_mask[:, displacement_x:] *= 0

    return iso_mask, atlas_mask, bregma_coords


def get_colormap(cmap='hotcold'):
    hotcold = ['#aefdff', '#60fdfa', '#2adef6', '#2593ff', '#2d47f9', '#3810dc', '#3d019d',
               '#313131',
               '#97023d', '#d90d39', '#f8432d', '#ff8e25', '#f7da29', '#fafd5b', '#fffda9']

    cyanmagenta = ['#00FFFF', '#FFFFFF', '#FF00FF']

    if cmap == 'cyanmagenta':
        cmap = LinearSegmentedColormap.from_list("Custom", cyanmagenta)

    elif cmap == 'whitemagenta':
        cmap = LinearSegmentedColormap.from_list("Custom", ['#FFFFFF', '#FF00FF'])

    elif cmap == 'hotcold':
        cmap = LinearSegmentedColormap.from_list("Custom", hotcold)

    elif cmap == 'grays':
        cmap = get_cmap('Greys')

    elif cmap == 'viridis':
        cmap = get_cmap('viridis')

    elif cmap == 'blues':
        cmap = get_cmap('Blues')

    elif cmap == 'magma':
        cmap = get_cmap('magma')

    else:
        cmap = get_cmap(cmap)

    cmap.set_bad(color='k', alpha=0.1)

    return cmap
