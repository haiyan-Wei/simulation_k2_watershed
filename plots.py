import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

#   setup_style         : set global rcParams; target='journal' (print) or 'screen'
#   save_figure         : save a figure as vector PDF (print) and/or PNG (screen)
#   plot_xy_1to1        : standalone scatter with 1:1 line + regression (makes its own figure)
#   plot_xy_1to1_ax     : same, but draws onto a provided ax (for multi-panel figures)
#   plot_x_y_ax         : general scatter onto a provided ax; keyword-only options


SINGLE_COL = 3.5    # ~89 mm  -> single journal column
DOUBLE_COL = 7.0    # ~178 mm -> full / double journal column
SCREEN_SQ  = 15.0   # large square for on-screen viewing on a big monitor

FIGURE_TARGET = 'screen'    # set by setup_style; read by save_figure

# what save_figure does per target, unless overridden per call
_SAVE_DEFAULTS = {
    'journal': dict(formats=('pdf', 'png'), dpi=300, transparent=True),
    'screen':  dict(formats=('png',),       dpi=150, transparent=False),
}

def setup_style(target='journal'):
    """Set global rcParams AND the save defaults for save_figure.
    Call once at the entry point (run.py), before any plotting.

    target='journal' -> compact fonts; saves vector PDF + 300dpi PNG,
                        transparent background for print layout.
    target='screen'  -> larger fonts, white bg; saves 150dpi PNG only.
    """
    global FIGURE_TARGET
    FIGURE_TARGET = target
    base = {
        'axes.labelsize': 12,    # X / Y axis labels
        'xtick.labelsize': 10,   # X tick labels
        'ytick.labelsize': 10,   # Y tick labels
        'legend.fontsize': 10,   # legend text
        'axes.titlesize': 13,    # subplot titles
        'font.family': 'sans-serif',
    }
    if target == 'screen':
        # Larger fonts to stay legible against a big (e.g. 15-inch) canvas,
        # and an opaque white background so figures don't vanish in dark IDEs.
        base.update({
            'axes.labelsize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'axes.titlesize': 20,
            'figure.facecolor': 'white',
            'figure.dpi': 110,
        })
    plt.rcParams.update(base)



def save_figure(fig, name, formats=None, dpi=None, transparent=None,
                **savefig_kwargs):
    """Save `fig` once per format. `name` is the path/stem WITHOUT
    extension. formats/dpi/transparent default from the current
    setup_style target; pass them explicitly to override per figure."""
    defaults = _SAVE_DEFAULTS[FIGURE_TARGET]
    formats = defaults['formats'] if formats is None else formats
    dpi = defaults['dpi'] if dpi is None else dpi
    transparent = defaults['transparent'] if transparent is None else transparent
    for ext in formats:
        fig.savefig(f'{name}.{ext}', dpi=dpi, bbox_inches='tight',
                    transparent=transparent, **savefig_kwargs)


def plot_xy_1to1_ax(ax, x, y, xlabel, ylabel, title):
    """Scatter with a 1:1 reference line and regression, drawn onto `ax`."""
    slope, intercept, r_value, p_value, error = linregress(x, y)
    r2 = r_value * r_value
    lim = max(max(x), max(y)) * 1.05
    regression_str = (f'y={slope:.2f}x + {intercept:.2f}\n'
                      f'$r^{2}$={r2:.2f}\nN={len(x)}')

    ax.scatter(x, y, s=15, c='b', alpha=0.5)
    ax.plot(x, intercept + slope * x, '-', color='b')
    ax.text(0.05 * lim, lim * 0.88, regression_str,
            fontweight='bold', fontsize=labelsize)
    ax.plot((0, lim), (0, lim), '--', color='grey', linewidth=0.7)

    ax.set_title(title, fontweight='bold', fontsize=labelsize)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=labelsize)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=labelsize)

    ax.set_xlim(-lim * 0.01, lim)
    ax.set_ylim(-lim * 0.01, lim)
    ax.set_aspect('equal')
    ax.tick_params(direction='in')
    ax.tick_params(labelsize=labelsize - 2)


def plot_xy_1to1(x, y, xlabel, ylabel, title, figname):
    """Standalone version of plot_xy_1to1_ax: builds the figure and saves it."""
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_xy_1to1_ax(ax, x, y, xlabel, ylabel, title)
    plt.tight_layout()
    plt.savefig(figname, transparent=True)
    plt.close(fig)

def plot_x_y_ax(ax, x, y, *,
                xlabel='', ylabel='', title='',
                label='', color='C0', marker='o', s=40,
                add_regression=False,
                annotation='', annotation_loc=(0.05, 0.95),
                lim=None, tick=None,
                equal_axes=False,
                show_xlabel=True, show_ylabel=True,
                show_xticklabels=True, show_yticklabels=True):
    """Scatter x vs y onto a provided Axes.

    Only ax, x, y are positional; everything else is keyword-only. Rows where
    x or y is NaN are dropped. Safe to call repeatedly on the same ax to
    overlay series. Font sizing comes from rcParams.
    """
    data = pd.DataFrame({'x': x, 'y': y}).dropna()
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    if len(x) == 0:
        return ax

    ax.scatter(x, y, s=s, color=color, marker=marker, alpha=0.8, label=label)
    if label:
        ax.legend(loc='upper left', frameon=True, prop={'weight': 'bold'})

    if add_regression and len(x) > 1:
        slope, intercept, *_ = linregress(x, y)
        xline = np.array([x.min(), x.max()])
        ax.plot(xline, slope * xline + intercept,
                color=color, linestyle='--', linewidth=0.5)

    if lim is None:
        lim = max(x.max(), y.max()) * 1.05

    if annotation:
        ax.text(*annotation_loc, annotation, transform=ax.transAxes,
                color=color, fontweight='bold', ha='left', va='top')

    if title:
        ax.set_title(title, fontweight='bold')

    if tick is not None:
        ax.set_xticks(np.arange(0, lim + tick, tick))
        ax.set_xticks(np.arange(0, lim + tick, tick / 2), minor=True)
        ax.set_yticks(np.arange(0, lim + tick, tick))
        ax.set_yticks(np.arange(0, lim + tick, tick / 2), minor=True)

    if equal_axes:
        ax.plot([0, lim], [0, lim], linestyle='--', linewidth=0.5, color='k')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-lim * 0.015, lim)
        ax.set_ylim(-lim * 0.015, lim)

    ax.tick_params(which='both', direction='in',
                   labelbottom=show_xticklabels, labelleft=show_yticklabels)
    ax.set_xlabel(xlabel if show_xlabel else '', fontweight='bold')
    ax.set_ylabel(ylabel if show_ylabel else '', fontweight='bold')
    return ax