import os
os.environ.setdefault("MPLBACKEND", "Agg")

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


def create_layout(n_stages, plot_level):
    """Creates Axes to plot figures.

    Uses matplotlib.figure.Figure directly (not pyplot) so figure creation
    is thread-safe and never attempts to open a GUI window.

    Parameters
    ----------
    n_stages : int
        length of pipeline process
    plot_level : int
        0 : no plotting
        1 : regular plots
        2 : detailed plots

    Returns
    -------
    axes : list of Axes
    """
    if plot_level == 0:
        return None

    elif plot_level == 1:
        ncols = n_stages
        fig = Figure(figsize=(12, 5))
        axes = fig.subplots(nrows=1, ncols=ncols)
        ax_list = [axes] if n_stages == 1 else list(axes)
        return ax_list + [None] * (7 - n_stages)

    elif plot_level == 2:
        fig = Figure(figsize=(12, 10))
        gs = GridSpec(3, 3, figure=fig)
        ax_main      = fig.add_subplot(gs[0, 0])
        ax_structure = fig.add_subplot(gs[0, 1])
        ax_signal    = fig.add_subplot(gs[1, 0:2])
        ax_fourier   = fig.add_subplot(gs[2, 0:2])
        ax_tags      = fig.add_subplot(gs[0, 2])
        ax_bin       = fig.add_subplot(gs[1, 2])
        ax_poi       = fig.add_subplot(gs[2, 2])
        fig.tight_layout()
        if n_stages == 1:
            return [ax_main, None, None, ax_structure, ax_signal, ax_fourier, None]
        elif n_stages == 2:
            return [ax_main, ax_bin, None, ax_structure, ax_signal, ax_fourier, ax_tags]
        elif n_stages == 3:
            return [ax_main, ax_bin, ax_poi, ax_structure, ax_signal, ax_fourier, ax_tags]
