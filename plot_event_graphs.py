""" Per-event hydrograph figures.

Self-contained capability: given a finished simulation folder (saved
outlet_*.sim files) and the observed csv files, produce one obs/sim
comparison figure per event. Never invokes K2 -- safe to rerun for
replotting after a long batch.

Text styling comes from rcParams: call plots.setup_style() once at the
entry point (run.py) before plotting. Only line/marker sizes are set here.

Color/style convention (orthogonal encoding):
  color  -> variable : green = rainfall, blue = runoff
  style  -> source   : solid = observed, dashed = simulated
Multiple gauges are shades of green (all observed rainfall, light->dark).
Observed series carry markers (measured points); simulated series are
plain lines (continuous model output). Estimated-data flags are red stars.
Legend labels always start with 'observed'/'simulated' and put the event
total depth in parentheses.

Gauge rain rendering is dispatched automatically per event:
<= GAUGE_ENVELOPE_THRESHOLD gauges -> one curve per gauge (green shades);
 > GAUGE_ENVELOPE_THRESHOLD gauges -> green min-max envelope + mean curve.

Public:
- group_figures_by_flag    : move figures into one subfolder per flag value
- plot_single_event_obs_sim   : the per-event figure; panels adapt to options
    2 panels: hyetograph + hydrograph
    3 panels: + accumulated rain & runoff on top   (show_acc_rain=True)
    4 panels: + acc. runoff and sedigraph below    (df_obsSedi given)
    5 panels: both options together
"""

import os
import shutil
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colormaps

from plots import save_figure


MSIZE = 6                      # marker size for observed series, all panels
OBS_LW = 2.0                   # linewidth, observed series
SIM_LW = 2.2                   # linewidth, simulated series
GAUGE_ENVELOPE_THRESHOLD = 5   # >this many gauges -> envelope rendering
RAIN_COLOR = 'g'
RUNOFF_COLOR = 'b'


# ----------------------------------------------------------------------------
# small computation helpers
# ----------------------------------------------------------------------------

def _acc_depth_from_rate(t_min, rate_mmhr):
    """Accumulated depth (mm) by trapezoid integration of a rate series
    (mm/hr at minute timestamps)."""
    t_hr = np.asarray(t_min, dtype=float) / 60.
    rate = np.asarray(rate_mmhr, dtype=float)
    if len(rate) < 2:
        return np.zeros_like(rate)
    increments = np.diff(t_hr) * (rate[1:] + rate[:-1]) / 2.
    return np.concatenate([[0.], np.cumsum(increments)])


def _gauge_colors(n):
    """Distinguishable shades of green, light to dark, for n gauges."""
    if n == 1:
        return [RAIN_COLOR]
    return [colormaps['Greens'](x) for x in np.linspace(0.45, 0.95, n)]


def _gauges_summary_label(df_obsRain, prefix):
    """Collective legend entry: '<prefix>, N gauges (min-max mm)' or the
    single-gauge form '<prefix>, gauge X (total mm)'."""
    totals = df_obsRain.groupby('gauge').accDepth.max()
    if len(totals) == 1:
        return f'{prefix}, gauge {totals.index[0]} ({totals.iloc[0]:.2f}mm)'
    return (f'{prefix}, {len(totals)} gauges '
            f'({totals.min():.1f}-{totals.max():.1f}mm)')


def _gauge_rates_on_grid(df_obsRain, step_min=1.0):
    """Align all gauge rate series onto a common time grid (forward-fill,
    since rates are stepped). Returns (grid, DataFrame[time x gauge])."""
    t_max = df_obsRain.elapsedTime.max()
    grid = np.arange(0., t_max + step_min, step_min)
    aligned = {}
    for gauge, df in df_obsRain.groupby('gauge'):
        s = pd.Series(df.rainRate.values, index=df.elapsedTime.values)
        s = s[~s.index.duplicated(keep='first')].sort_index()
        aligned[gauge] = s.reindex(grid, method='ffill').fillna(0.).values
    return grid, pd.DataFrame(aligned, index=grid)


# ----------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------

def _legend(ax, handles=None, labels=None, loc='upper right'):
    """Legend with an opaque white background so curves can't obscure it."""
    if handles is not None:
        return ax.legend(handles, labels, loc=loc, frameon=True,
                         facecolor='white', framealpha=1.0, edgecolor='none')
    return ax.legend(loc=loc, frameon=True, facecolor='white',
                     framealpha=1.0, edgecolor='none')


def _plot_sim_runoff(df_outlet, simRunoff):
    """Simulated runoff hydrograph + time-to-peak marker (current axes)."""
    plt.plot(df_outlet.elapsedtime, df_outlet.runoffrate, '--',
             label=f'simulated runoff ({simRunoff:.2f}mm)',
             color=RUNOFF_COLOR, linewidth=SIM_LW)
    peak = df_outlet.runoffrate.max()
    tp = df_outlet[df_outlet.runoffrate == peak].elapsedtime.values[0]
    plt.plot([tp, tp], [0, peak], '--', color=RUNOFF_COLOR, linewidth=0.8)


def _plot_obs_runoff(df_obsRunoff, obsRunoff, data_type):
    """Observed runoff hydrograph + time-to-peak + estimated-point markers."""
    plt.plot(df_obsRunoff.elapsedTime, df_obsRunoff.runoffRate_DAP, '-o',
             label=f'observed runoff ({obsRunoff:.2f}mm)',
             color=RUNOFF_COLOR, linewidth=OBS_LW, markersize=MSIZE)
    peak = df_obsRunoff.runoffRate_DAP.max()
    tp = df_obsRunoff[df_obsRunoff.runoffRate_DAP == peak].elapsedTime.values[0]
    plt.plot([tp, tp], [0, peak], '-', color=RUNOFF_COLOR, linewidth=0.8)
    if data_type in ('a', 'd'):
        plt.plot(df_obsRunoff.loc[df_obsRunoff.runCode > 0, 'elapsedTime'],
                 df_obsRunoff.loc[df_obsRunoff.runCode > 0, 'runoffRate_DAP'],
                 '*', markersize=MSIZE + 3, label='estimated rate', color='r')


def _plot_gauge_rain_curves(ax, df_obsRain, data_type):
    """Hyetograph for FEW gauges: one curve per gauge in green shades,
    each with its own legend entry; estimated-rate stars labeled once."""
    totals = df_obsRain.groupby('gauge').accDepth.max()
    colors = _gauge_colors(len(totals))
    est_labeled = False
    for (gauge, df), color in zip(df_obsRain.groupby('gauge'), colors):
        ax.plot(df.elapsedTime, df.rainRate, '-o',
                linewidth=OBS_LW, markersize=MSIZE, color=color,
                label=f'observed rainfall, gauge {gauge} '
                      f'({totals[gauge]:.2f}mm)')
        if data_type == 'a':
            est = df.loc[df.intensityCode_a > 0]
            if len(est):
                ax.plot(est.elapsedTime, est.rainRate, '*',
                        markersize=MSIZE + 3, color='r',
                        label=None if est_labeled else 'estimated rate')
                est_labeled = True
        if data_type == 'd':
            est = df.loc[df.rainCode > 0]
            if len(est):
                ax.plot(est.elapsedTime, est.rainRate, '*',
                        markersize=MSIZE + 3, color='r',
                        label=None if est_labeled else 'estimated rain')
                est_labeled = True


def _plot_gauge_rain_envelope(ax, df_obsRain, data_type):
    """Hyetograph for MANY gauges: green min-max band across gauges plus
    the gauge-mean curve; one collective legend entry with totals range.
    Estimated-rate stars are omitted (per-gauge detail at this density)."""
    grid, rates = _gauge_rates_on_grid(df_obsRain)
    ax.fill_between(grid, rates.min(axis=1), rates.max(axis=1),
                    color=RAIN_COLOR, alpha=0.25, linewidth=0,
                    label=_gauges_summary_label(df_obsRain,
                                                'observed rainfall'))
    ax.plot(grid, rates.mean(axis=1), '-', linewidth=OBS_LW,
            color=RAIN_COLOR)


def _plot_gauge_rain(ax, df_obsRain, data_type):
    """Dispatch: per-gauge curves for few gauges, envelope for many."""
    if df_obsRain.gauge.nunique() > GAUGE_ENVELOPE_THRESHOLD:
        _plot_gauge_rain_envelope(ax, df_obsRain, data_type)
    else:
        _plot_gauge_rain_curves(ax, df_obsRain, data_type)


def _xlim_and_ticks(df_obsRunoff, df_outlet):
    """x-axis limit and hour-tick positions for a hydrograph panel.
    Spans the full simulated duration on purpose: a long empty tail tells
    the user the duration multiplier can be reduced."""
    if len(df_obsRunoff) == 0:
        xlim = df_outlet.elapsedtime.max() * 1.05
    else:
        xlim = max(df_obsRunoff.elapsedTime.max(),
                   df_outlet.elapsedtime.max()) * 1.05
    if xlim > 360:
        hrticks = np.arange(0, xlim, 60)
    elif xlim > 60:
        hrticks = np.arange(0, xlim, 30)
    else:
        hrticks = np.arange(0, xlim, 10)
    return hrticks, xlim


# ----------------------------------------------------------------------------
# main entry: loop over events
# ----------------------------------------------------------------------------

def plot_events_obs_sim(event_graph_dir, events, outletfiles, obs_dir,
                           pre_dir, df_event, df_sim, watershed, rain_type):

    """ create one obs/sim comparison figure per event from saved
        outlet_*.sim files and observed csv files. events without
        simulation results are skipped. pure post-processing: never
        invokes K2, safe to rerun for replotting. """

    n_ok, n_skipped, n_failed = 0, 0, 0

    for i, (event, outletfile) in enumerate(zip(events, outletfiles)):

        if event not in df_sim.index:
            logging.info(f'no hydrograph for {event}: no simulation results')
            n_skipped += 1
            continue

        try:
            print(f'plot hydrograph {event} {i}/{len(events)}')

            obsRunoff = df_event.loc[event, 'obs_runoff_mm']
            simRunoff = df_sim.loc[event, 'sim_runoff_mm']
            simRain = df_sim.loc[event, 'sim_rain_mm']

            df_outlet = pd.read_csv(outletfile, skiprows=2,
                                    header=None, usecols=[0, 1, 2, 4])
            df_outlet.columns = ['elapsedtime', 'rainfallrate',
                                 'runoffrate', 'sediment']

            df_obsRunoff = pd.read_csv(f'{obs_dir}/{event}_runoff.csv')
            df_obsRunoff = df_obsRunoff[
                ['elapsedTime', 'runoffRate_DAP', 'runCode', 'accDepth']]

            if rain_type == 'gauge':
                df_obsRain = pd.read_csv(f'{obs_dir}/{event}_rainfall.csv')
                df_obsRain = df_obsRain[
                    ['gauge', 'elapsedTime', 'intensityCode_a', 'rainCode',
                     'rainRate', 'accDepth']]
            else:
                raise ValueError(f'unknown rain_type "{rain_type}"')

            plot_single_event_obs_sim(
                event_graph_dir, event, watershed, df_obsRunoff, df_obsRain,
                df_outlet, simRunoff, simRain, obsRunoff,
                df_obsSedi=None, show_acc_rain=True)

            n_ok += 1

        except Exception:
            logging.exception(f'plotting hydrograph failed for {event}')
            n_failed += 1

    msg = (f'hydrographs: {n_ok} plotted, {n_skipped} skipped '
           f'(no sim results), {n_failed} failed')
    print(msg)
    logging.info(msg)


def group_figures_by_flag(event_graph_dir, df_event, flag_col):

    """ move event figures into subfolders, one per unique flag value.
        events with no flag value go to 'flag_unflagged'. existing flag_*
        subfolders are cleared first so regrouping after flag changes
        leaves no stale copies. does nothing if flag_col is None/missing. """

    if not flag_col:
        return
    if flag_col not in df_event.columns:
        logging.info(f'flag column "{flag_col}" not in event table, '
                     f'figures not grouped')
        return

    # clear previous grouping so a figure can't linger under an old flag
    for d in os.listdir(event_graph_dir):
        old = os.path.join(event_graph_dir, d)
        if os.path.isdir(old) and d.startswith('flag_'):
            for f in os.listdir(old):
                shutil.move(os.path.join(old, f),
                            os.path.join(event_graph_dir, f))
            os.rmdir(old)

    for event, flag in df_event[flag_col].items():
        src = f'{event_graph_dir}/{event}.png'
        if not os.path.exists(src):
            continue
        if pd.isna(flag):
            label = 'unflagged'
        elif isinstance(flag, float) and flag.is_integer():
            label = str(int(flag))      # avoid 'flag_1.0'
        else:
            label = str(flag)
        subdir = f'{event_graph_dir}/flag_{label}'
        os.makedirs(subdir, exist_ok=True)
        shutil.move(src, f'{subdir}/{event}.png')
        print(f'group graphs  {event} -> flag_{label}')


# ----------------------------------------------------------------------------
# the per-event figure
# ----------------------------------------------------------------------------

def plot_single_event_obs_sim  (fig_dir, event, watershed, df_obsRunoff,
                               df_obsRain, df_outlet, simRunoff, simRain,
                               obsRunoff, df_obsSedi=None,
                               show_acc_rain=False):

    """ obs/sim comparison figure for one event. always includes a
        hyetograph and a hydrograph panel; options add more:

        show_acc_rain : add a top panel with accumulated rainfall
                        (observed green solid, simulated areal green
                        dashed) and accumulated runoff on a right axis
                        (observed blue solid, simulated blue dashed)
        df_obsSedi    : if given (and non-empty), add accumulated-runoff
                        and sedigraph panels below; df_obsRunoff must then
                        include an 'accDepth' column
    """

    hrticks, xlim = _xlim_and_ticks(df_obsRunoff, df_outlet)
    data_type = event[0]
    has_sedi = df_obsSedi is not None and len(df_obsSedi) > 0

    n_panels = 2 + show_acc_rain + 2 * has_sedi
    fig_height = {2: 8, 3: 10, 4: 12, 5: 14}[n_panels]

    fig, axes = plt.subplots(n_panels, 1, figsize=(15, fig_height),
                             constrained_layout=True)

    panel = 0

    # --- accumulated rainfall (left) + runoff (right), optional
    if show_acc_rain:
        ax = axes[panel]
        totals = df_obsRain.groupby('gauge').accDepth.max()
        colors = _gauge_colors(len(totals))
        first = True
        for (gauge, df_gauge), color in zip(df_obsRain.groupby('gauge'),
                                            colors):
            ax.plot(df_gauge.elapsedTime, df_gauge.accDepth, '-',
                    linewidth=OBS_LW * 0.75, color=color,
                    label=(_gauges_summary_label(df_obsRain,
                                                 'observed acc. rainfall')
                           if first else None))
            first = False
        ax.plot(df_outlet.elapsedtime,
                _acc_depth_from_rate(df_outlet.elapsedtime,
                                     df_outlet.rainfallrate),
                '--', linewidth=SIM_LW, color=RAIN_COLOR,
                label=f'simulated acc. rainfall, areal ({simRain:.2f}mm)')
        ax.set_ylabel('Acc. Rainfall\n(mm)')

        ax_run = ax.twinx()
        if len(df_obsRunoff) > 0 and 'accDepth' in df_obsRunoff.columns:
            ax_run.plot(df_obsRunoff.elapsedTime, df_obsRunoff.accDepth,
                        '-', linewidth=OBS_LW, color=RUNOFF_COLOR,
                        label=f'observed acc. runoff ({obsRunoff:.2f}mm)')
        ax_run.plot(df_outlet.elapsedtime,
                    _acc_depth_from_rate(df_outlet.elapsedtime,
                                         df_outlet.runoffrate),
                    '--', linewidth=SIM_LW, color=RUNOFF_COLOR,
                    label=f'simulated acc. runoff ({simRunoff:.2f}mm)')
        ax_run.set_ylabel('Acc. Runoff (mm)', color=RUNOFF_COLOR)
        ax_run.tick_params(axis='y', colors=RUNOFF_COLOR)
        ax_run.set_ylim(bottom=0)

        # legend goes on the twin axis: twins render on top of the host,
        # so a legend on `ax` would be painted over by ax_run's curves
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_run.get_legend_handles_labels()
        _legend(ax_run, h1 + h2, l1 + l2)
        panel += 1

    # --- hyetograph: gauges (curves or envelope) + simulated areal
    ax = axes[panel]
    _plot_gauge_rain(ax, df_obsRain, data_type)
    ax.plot(df_outlet.elapsedtime, df_outlet.rainfallrate, '--',
            color=RAIN_COLOR, linewidth=SIM_LW,
            label=f'simulated rainfall, areal ({simRain:.2f}mm)')
    ax.set_ylabel('Rainfall Rate\n(mm/hr)')
    _legend(ax)
    panel += 1

    # --- hydrograph
    ax = axes[panel]
    plt.sca(ax)
    _plot_sim_runoff(df_outlet, simRunoff)
    if len(df_obsRunoff) > 0:
        _plot_obs_runoff(df_obsRunoff, obsRunoff, data_type)
    ax.set_ylabel('Runoff Rate\n(mm/hr)')
    _legend(ax)
    panel += 1

    # --- sediment panels (optional)
    if has_sedi:
        ax = axes[panel]
        ax.plot(df_obsRunoff.elapsedTime, df_obsRunoff.accDepth, '-o',
                color=RUNOFF_COLOR, linewidth=OBS_LW, markersize=MSIZE,
                label=f'observed acc. runoff ({obsRunoff:.2f}mm)')
        for c in df_obsSedi.elapsedTime.values:
            ax.axvline(c, color='k')
        for c in (obsRunoff * 0.25, obsRunoff * 0.50, obsRunoff * 0.75):
            ax.axhline(c, color='grey', linestyle='--')
        ax.set_ylabel('Acc. Runoff\n(mm)')
        _legend(ax, loc='upper left')
        panel += 1

        ax = axes[panel]
        ax.plot(df_outlet.elapsedtime, df_outlet.sediment, '--',
                color='saddlebrown', linewidth=SIM_LW,
                label='simulated sedigraph')
        ax.set_ylabel('Sediment\n(kg/s)')
        _legend(ax, loc='upper left')

        ax2 = ax.twinx()
        ax2.plot(df_obsSedi.elapsedTime, df_obsSedi.concentr_weight, '-o',
                 color='saddlebrown', markersize=MSIZE,
                 label='observed sediment concentration')
        ax2.set_ylabel('Sediment Concentration (weight)')
        _legend(ax2)

    # --- common formatting
    axes[0].set_title(f'Runoff Event {event} at Flume {watershed}',
                      fontweight='bold')
    for ax in axes:
        ax.set_xticks(hrticks)
        ax.set_xlim(-10, xlim)
        ax.tick_params(direction='in')
    axes[-1].set_xlabel('Time (min)', fontweight='bold')

    save_figure(fig, f'{fig_dir}/{event}', formats=('png',),
                transparent=False)
    plt.close(fig)