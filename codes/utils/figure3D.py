from codes.utils.misc.plot_average_widefield_timecourse import plot_average_wf_timecourse


def figure3d(data, saving_path, formats=['png'], scale=(-0.035, 0.035), halfrange=0.015):
    trial_types = list(data[list(data.keys())[0]].keys())

    # Keep only correct:
    trial_types = [trial for trial in trial_types if 'hit' in trial]

    # Plot
    plot_average_wf_timecourse(data, trial_types, saving_path, formats=formats, scale=scale, diff_range=halfrange)

        