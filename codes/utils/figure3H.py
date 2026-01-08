import matplotlib.pyplot as plt
from codes.utils.misc.fig_saving import save_fig


def figure3h_barplots(df, saving_path, name, formats):
    df = df.loc[df.Time > 0.02]

    # Significant level :
    wh_df_sig = df[df['Dprime'] > 2].groupby(['Spot'], as_index=False).first()

    colors = {'(-1.5, 0.5)': 'pink',
              '(-1.5, 3.5)': 'darkorange',
              '(-1.5, 4.5)': 'orange',
              '(0.5, 4.5)': 'red',
              '(1.5, 1.5)': 'blue',
              '(2.5, 2.5)': 'mediumorchid'}

    order_spot = wh_df_sig.sort_values('Time')['Spot'].values.tolist()
    order_colors = [colors.get(spot) for spot in order_spot]
    order_time = wh_df_sig.sort_values('Time')['Time'].values.tolist()

    fig, ax = plt.subplots(1, 1, figsize=(3, 4))
    ax.bar(x=order_spot, height=order_time, color=order_colors)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Sorted areas')
    fig.tight_layout()

    save_fig(fig, saving_path=saving_path, figure_name=name, formats=formats)
