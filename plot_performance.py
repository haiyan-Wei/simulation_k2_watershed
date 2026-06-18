""" Model performance evaluation: obs vs sim scatter plots and indicator
tables for runoff, peak, and sediment.

Public:
- create_obs_sim_plots_log_indicators : scatter panel per variable,
  indicators logged and returned as a DataFrame
- log_summary_stats : N/min/max/mean of the standard obs & sim variables
"""

import logging
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from evaluation_indicators import get_indicators
from plots import plot_x_y_ax, save_figure


INDICATOR_COLUMNS = ['tag', 'label', 'data_type', 'N',
                     'PBIAS', 'RSR', 'R2', 'NS', 'RMSE', 'KGE', 'OF', 'OF_W']


def log_summary_stats(df):
    """Log N/min/max/mean for the standard observed & simulated variables."""
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


def create_obs_sim_plots_log_indicators(wspace, df, watershed, tag):

    """ one scatter panel per variable (runoff, peak, sediment), with
        indicators annotated, logged, and returned as a DataFrame.
        tag is used in the figure name and the indicator table. """

    wspace = Path(wspace)

    print('Calculate Model Performance ...')
    log_summary_stats(df)

    print('Create Performance Plot ...')
    logging.info('\nIndicators')

    VARIABLES = [
        {
            'var': 'runoff_mm',
            'label': 'Runoff (mm)',
            'tick': 20,
        },
        {
            'var': 'peak_mmhr',
            'label': 'Runoff Peak (mm/hr)',
            'tick': 200,
        },
        {
            'var': 'sedi_tonha',
            'label': 'Sediment Yield (ton/ha)',
            'tick': 2,
        },
    ]

    shapes = {4: (2, 2, (8, 8)),
              3: (1, 3, (12, 5)),
              2: (1, 2, (8, 5)),
              1: (1, 1, (5, 5))}
    nrows, ncols, figure_size = shapes[len(VARIABLES)]

    fig, axes = plt.subplots(nrows, ncols, figsize=figure_size,
                             layout='constrained', squeeze=False)
    axes = list(axes.flat)

    indicator_rows = []
    for ax, cfg in zip(axes, VARIABLES):
        var = cfg['var']
        label = cfg['label']
        tick = cfg['tick']
        obs_var, sim_var = f'obs_{var}', f'sim_{var}'
        df_ = df[[obs_var, sim_var]].dropna()

        if df_.empty:
            logging.info(f'no data points for {label}')
            ax.set_visible(False)
            continue

        obs, sim = df_[obs_var], df_[sim_var]
        indicators = get_indicators(sim=sim, obs=obs)
        logging.info(f'{label}: {indicators[0]}')
        N, PBIAS, RSR, R2, NS, RMSE, KGE, OF, OF_W = indicators[1:]
        indicator_rows.append(
            [tag, label, 'ad', N, PBIAS, RSR, R2, NS, RMSE, KGE, OF, OF_W]
        )

        plot_x_y_ax(
            ax, obs, sim,
            xlabel=f'Observed {label}', ylabel=f'Simulated {label}',
            s=40, add_regression=True,
            annotation=f'$r^2$={R2:.2f}\nNS={NS:.2f}\nKGE={KGE:.2f}\nN={N}',
            tick=tick, equal_axes=True,
        )

    fig.suptitle(f'Flume {watershed}', fontweight='bold')
    save_figure(fig, wspace / f'{tag}_results.png')
    plt.close(fig)

    df_indicator = pd.DataFrame(indicator_rows, columns=INDICATOR_COLUMNS)

    return df_indicator