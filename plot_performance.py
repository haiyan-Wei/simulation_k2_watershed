""" Model performance evaluation: obs vs sim scatter plots and indicator
tables for runoff, peak, and sediment.

Public:
- create_obs_sim_plots_log_indicators : scatter panel per variable,
  indicators logged and returned as a DataFrame
- log_summary_stats : N/min/max/mean of the standard obs & sim variables
"""

import math
import logging
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from evaluation_indicators import get_indicators
from plots import plot_x_y_ax, save_figure


def _nice_tick(lim, target_n=5):
    """Round tick interval giving ~target_n major ticks across 0..lim."""
    if lim <= 0:
        return 1
    raw = lim / target_n
    magnitude = 10 ** math.floor(math.log10(raw))
    for step in (1, 2, 5, 10):
        tick = step * magnitude
        if lim / tick <= target_n * 2:
            return tick
    return magnitude * 10


INDICATOR_COLUMNS = ['tag', 'label', 'data_type', 'N',
                     'PBIAS', 'RSR', 'R2', 'NS', 'RMSE', 'KGE', 'OF', 'OF_W']

_VARIABLE_CATALOG = {
    'runoff_mm':        {'label': 'Runoff (mm)'},
    'peak_mmhr':        {'label': 'Runoff Peak (mm/hr)'},
    'sedi_tonha':       {'label': 'Sediment Yield (ton/ha)'},
    'time_to_peak_min': {'label': 'Time to Peak (min)'},
}

_DEFAULT_EVAL_VARIABLES = ['runoff_mm', 'peak_mmhr', 'sedi_tonha']


def log_summary_stats(df):
    """Log N/min/max/mean for the standard observed & simulated variables."""
    logging.info("\n\n=== Model Performance ===\n\n")
    logging.info("Statistics (N/min/max/mean)")
    logging.info(
        f"{'Variable':<25} {'N':>8} {'Min':>12} {'Max':>12} {'Mean':>12}"
    )
    logging.info("-" * 70)

    for v in [
        "obs_rain_mm", "sim_rain_mm",
        "obs_runoff_mm", "sim_runoff_mm",
        "obs_sedi_kg", "sim_sedi_kg",
        "obs_peak_mmhr", "sim_peak_mmhr",
        "obs_time_to_peak_min", "sim_time_to_peak_min",
    ]:
        if v not in df.columns:
            logging.info(f"{v:<25} missing")
            continue
        s = df[v].dropna()
        if len(s) == 0:
            logging.info(f"{v:<25} {0:>8,d}  (all values missing)")
        else:
            logging.info(
                f"{v:<25} "
                f"{len(s):>8,d} "
                f"{s.min():>12.2f} "
                f"{s.max():>12.2f} "
                f"{s.mean():>12.2f}"
            )


def create_obs_sim_plots_log_indicators(wspace, df, watershed, tag,
                                        eval_variables=None):

    """ one scatter panel per variable (runoff, peak, sediment, time-to-peak),
        with indicators annotated, logged, and returned as a DataFrame.
        tag is used in the figure name and the indicator table.
        eval_variables: list of keys from _VARIABLE_CATALOG; defaults to
        ['runoff_mm', 'peak_mmhr', 'sedi_tonha']. """

    wspace = Path(wspace)

    if eval_variables is None:
        eval_variables = _DEFAULT_EVAL_VARIABLES
    unknown = [v for v in eval_variables if v not in _VARIABLE_CATALOG]
    if unknown:
        logging.warning(f'unknown eval_variables ignored: {unknown}')
    VARIABLES = [{'var': v, **_VARIABLE_CATALOG[v]}
                 for v in eval_variables if v in _VARIABLE_CATALOG]
    if not VARIABLES:
        logging.warning('no valid eval_variables — skipping performance plots')
        return pd.DataFrame(columns=INDICATOR_COLUMNS)

    print('Calculate Model Performance ...')
    log_summary_stats(df)

    print('Create Scatter Plot ...')

    shapes = {4: (2, 2, (12, 12)),
              3: (1, 3, (16, 6)),
              2: (1, 2, (11, 6)),
              1: (1, 1, (6,  6))}
    nrows, ncols, figure_size = shapes[len(VARIABLES)]

    fig, axes = plt.subplots(nrows, ncols, figsize=figure_size,
                             layout='constrained', squeeze=False)
    axes = list(axes.flat)

    indicator_rows = []
    no_data_labels = []
    for ax, cfg in zip(axes, VARIABLES):
        var = cfg['var']
        label = cfg['label']
        obs_var, sim_var = f'obs_{var}', f'sim_{var}'
        df_ = df[[obs_var, sim_var]].dropna()

        if df_.empty:
            no_data_labels.append(label)
            ax.set_visible(False)
            continue

        obs, sim = df_[obs_var], df_[sim_var]
        indicators = get_indicators(sim=sim, obs=obs)
        N, PBIAS, RSR, R2, NS, RMSE, KGE, OF, OF_W = indicators[1:]
        indicator_rows.append(
            [tag, label, 'ad', N, PBIAS, RSR, R2, NS, RMSE, KGE, OF, OF_W]
        )

        lim = max(obs.max(), sim.max()) * 1.05
        tick = _nice_tick(lim)
        plot_x_y_ax(
            ax, obs, sim,
            xlabel=f'Observed {label}', ylabel=f'Simulated {label}',
            s=40, add_regression=True,
            annotation=f'$r^2$={R2:.2f}\nNS={NS:.2f}\nKGE={KGE:.2f}\nN={N}',
            lim=lim, tick=tick, equal_axes=True,
        )

    header = (f"\n{'Variable':<25} {'N':>5} {'R2':>6} {'RMSE':>8} "
              f"{'NSE':>7} {'RSR':>7} {'PBIAS':>8} {'KGE':>7} {'OF':>8} {'OF_W':>8}")
    logging.info('\nPerformance Indicators')
    logging.info(header)
    logging.info('-' * 90)
    for row in indicator_rows:
        _, label, _, N, PBIAS, RSR, R2, NS, RMSE, KGE, OF, OF_W = row
        logging.info(
            f"{label:<25} {int(N):>5} {R2:>6.2f} {RMSE:>8.2f} "
            f"{NS:>7.2f} {RSR:>7.2f} {PBIAS:>8.2f} {KGE:>7.2f} {OF:>8.2f} {OF_W:>8.2f}"
        )
    for label in no_data_labels:
        logging.info(f"{label:<25} {'no data':>5}")

    fig.suptitle(f'Watershed {watershed}', fontweight='bold', fontsize=16)
    save_figure(fig, wspace / f'{tag}_results.png')
    plt.close(fig)

    df_indicator = pd.DataFrame(indicator_rows, columns=INDICATOR_COLUMNS)

    return df_indicator