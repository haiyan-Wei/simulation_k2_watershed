import os
import re
import shutil
import logging
import pandas as pd
from pathlib import Path

from k2_runner import run_k2_batch 
from plot_event_graphs import plot_events_obs_sim, group_figures_by_flag
from plot_performance import create_obs_sim_plots_log_indicators


def _check_parfile_outlet_print(parfile_path, plot_outlet_graph):

    """ verify the LAST element in the parfile (the outlet, a channel) has
        PRINT = 3, so K2 writes the outlet time series to OUTLET.SIM.
        raises ValueError to stop before any K2 run if it doesn't. """

    with open(parfile_path) as f:
        lines = f.readlines()

    elements = []        # (element_type, print_value) in file order
    current = None
    for line in lines:
        m = re.match(r'\s*BEGIN\s+(PLANE|CHANNEL)', line, flags=re.IGNORECASE)
        if m:
            current = [m.group(1).upper(), None]
            elements.append(current)
            continue
        if re.match(r'\s*END\s+(PLANE|CHANNEL)', line, flags=re.IGNORECASE):
            current = None
            continue
        if current is not None and current[1] is None:
            m = re.search(r'PRINT\s*=\s*(\d+)', line, flags=re.IGNORECASE)
            if m:
                current[1] = int(m.group(1))

    if not elements:
        raise ValueError(f'no PLANE/CHANNEL elements found in {parfile_path}')

    
    n_planes = sum(1 for etype, _ in elements if etype == 'PLANE')
    n_channels = sum(1 for etype, _ in elements if etype == 'CHANNEL')
    msg = (f'parfile {os.path.split(parfile_path)[1]}: '
           f'{n_planes} plane and {n_channels} channel elements')
    logging.info(msg)

    if plot_outlet_graph:
        last_type, last_print = elements[-1]

        if last_type != 'CHANNEL':
            logging.info(f'note: last element in {parfile_path} is a {last_type}, '
                        f'expected a CHANNEL at the outlet')

        if last_print != 3:
            raise ValueError(
                f'plot_outlet_graph is on, but the last element has PRINT = {last_print}. '
                f'It must be PRINT = 3 for K2 to write the outlet time series '
                f'Edit the parfile or set plot_outlet_graph = False.')


def _load_events(wspace, event_table, sheet_name):

    """ read the event table, back it up to the workspace,
        and rename observed variables to standard names """

    shutil.copyfile(event_table, f'{wspace}/{os.path.split(event_table)[-1]}')
    logging.info('A copy of the event table is saved in the output folder')

    if event_table.endswith('.csv'):
        df = pd.read_csv(event_table).set_index('event')
    else:
        df = pd.read_excel(
            event_table, sheet_name=sheet_name).set_index('event')
    df = df.dropna(how='all')

    OBSERVED_COLUMN_MAP = {"obsAvgRain": "obs_rain_mm",
                            "obsRunoff": "obs_runoff_mm",
                            "obsPeak": "obs_peak_mmhr",
                            "obsSediKg": "obs_sedi_kg",
                            "obsTp": "obs_time_to_peak_min",
    }
    df = df.rename(columns=OBSERVED_COLUMN_MAP)

    return df


def simulate_watershed(wspace, watershed, parfile_path, event_table, sheet_name,
                       pre_dir, obs_dir, k2_exe, dur_mp, dur_var, timestep,
                       rain_type='gauge', flag_col=None,
                       plot_outlet_graph=True,
                       group_graphs_by_flag=True,
                       indicators_by_flag=False,
                       overwrite=False, figure_style='screen'):

    """ full workflow for one watershed: load the event table, run k2 for
        every event, (optionally) plot per-event hydrographs and group them
        by flag, merge results with observations, and create evaluation
        plots and indicators. set indicators_by_flag=True to also compute
        indicators separately for each unique flag value. """

    # validate parfile before doing any work
    _check_parfile_outlet_print(parfile_path, plot_outlet_graph)

    # --- set up directories
    wspace = Path(wspace)
    k2pool_dir = wspace / 'k2pool'
    os.makedirs(k2pool_dir, exist_ok=True)

    # --- load events, stage inputs
    df_input_events = _load_events(wspace, event_table, sheet_name)

    parfile = os.path.split(parfile_path)[1]
    shutil.copy(parfile_path, f'{k2pool_dir}/{parfile}')
    shutil.copytree(pre_dir, k2pool_dir, dirs_exist_ok=True)

    # --- simulation file names and durations
    events = df_input_events.index
    prefiles = [f'{e}.pre' for e in events]
    outfiles = [f'{e}.out' for e in events]
    kinfiles = [f'kin_{e}.fil' for e in events]

    MIN_EVENT_DURATION = 200
    DEFAULT_SHORT_DURATION = 300
    df_input_events.loc[df_input_events[dur_var] <= MIN_EVENT_DURATION, dur_var] = DEFAULT_SHORT_DURATION
    simulation_durations = df_input_events[dur_var].values * dur_mp

    outletfiles = None
    if plot_outlet_graph:
        outletfiles = [f'{k2pool_dir}/outlet_{e}.sim' for e in events]
        event_graph_dir = f'{wspace}/event_graphs '
        os.makedirs(event_graph_dir, exist_ok=True)


    # --- run all events
    df_sim_results, failed = run_k2_batch(
        f'wshd_{watershed}', k2pool_dir, events, prefiles,
        parfile, outfiles, kinfiles, simulation_durations, k2_exe, timestep,
        outlet_sim_files=outletfiles, overwrite=overwrite)        

    if df_sim_results.empty:
        raise RuntimeError(
            f'no events produced results ({len(failed)} failed) -- '
            f'check the log and that prefiles exist in {pre_dir}')

    print('Simulations Done')
    
    # --- per-event figures (post-processing, reads files only)
    if plot_outlet_graph:
        plot_events_obs_sim(event_graph_dir, events, outletfiles, obs_dir,
                            pre_dir, df_input_events, df_sim_results, watershed, rain_type)
        if group_graphs_by_flag:
            group_figures_by_flag(event_graph_dir, df_input_events, flag_col)

    # --- merge simulation results with the event table
    df_sim_results = df_sim_results.apply(pd.to_numeric, errors='coerce')
    df_results = pd.concat([df_input_events, df_sim_results], axis=1)
    
    area_ha = df_sim_results['sim_area_ha'].iloc[0]
    df_results['obs_sedi_kgha'] = df_results['obs_sedi_kg'] / area_ha
    df_results['obs_sedi_tonha'] = df_results['obs_sedi_kg']/ area_ha / 1000.

    df_results.to_csv(wspace / 'simulation_results.csv')

    # --- evaluation plots and indicators
    df_indicator = create_obs_sim_plots_log_indicators(wspace, df_results, watershed, 'all')

    if indicators_by_flag and flag_col and flag_col in df_results.columns:
        for flag, df_flag in df_results.groupby(flag_col, dropna=False):
            tag = 'unflagged' if pd.isna(flag) else f'flag_{flag}'
            if len(df_flag) > 2:
                logging.info(f'\nresults for {tag} events')
                df_indicator_ = create_obs_sim_plots_log_indicators(
                    wspace, df, watershed, tag)
                df_indicator = pd.concat([df_indicator, df_indicator_])

    df_indicator.to_excel(f'{wspace}/df_indicator.xlsx')

    return df_results

