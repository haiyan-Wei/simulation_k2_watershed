""" simplified for single watershed"""

import os
import time
import logging
from k2_simulation_workflow import simulate_watershed
from utils import log_variables, create_loginfo_file
from plots import setup_style


# this code was used for LH Paper

watershed = 104
out = f'{watershed}_v6'
rain_type='gauge'  # default 


# event table
edc_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 
                              f'testrun_{watershed}_gauge384_rain12_rainAgg8_runAgg8')
event_table = os.path.join(edc_output_dir, 'df_event_w_flag.xlsx')
sheet_name = 'Sheet1'  # None if event table is in csv format
flag_col = 'final_flag'
simuation_duration_variable = 'eventDurMin'
simuation_duration_multiplier = 1.25
simulation_timestep = 0.5

# options
plot_outlet_graph = True
group_graphs_by_flag = False
overwrite_existing_runs = False
parfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs', 
                        'parfiles', f'{watershed}_head_channel_outleton.par')

figure_style = 'screen'    # 'journal': paper styling, saves PDF+PNG; 'screen': preview

def main():
    t0 = time.perf_counter()
    setup_style(figure_style)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.join(base_dir, 'outputs', out)
    os.makedirs(workspace, exist_ok=True)
    script_goal = 'Run K2.exe with EDC results'
    create_loginfo_file(os.path.join(workspace, f'{watershed}_k2_simulation.log'), script_goal, t0)
    
    k2_exe = os.path.join(base_dir, 'inputs','k2_x64.exe')
    pre_dir = os.path.join(edc_output_dir, 'prefiles')
    obs_dir = os.path.join(edc_output_dir, 'hyeto_hydro_graphs')

    log_variables('INPUTS: ', {'edc_output_dir': edc_output_dir,
                                'k2_exe': k2_exe,
                                'workspace': workspace,
                                'event_table': event_table,
                                'pre_dir': pre_dir,
                                'obs_dir': obs_dir,
                                'parfile': parfile,
                                'watershed': watershed,
                                'out': out,
                                'flag_col': flag_col,
                                'plot_outlet_graph': plot_outlet_graph,
                                'group_graphs_by_flag': group_graphs_by_flag,
                                'simulation_timestep': simulation_timestep,
                                'simuation_duration_variable': simuation_duration_variable,
                                'simuation_duration_multiplier': simuation_duration_multiplier,
                                'rain_type': rain_type,
                                'overwrite_existing_runs':overwrite_existing_runs,
                                'figure_style': figure_style,
    })

    simulate_watershed(workspace, watershed, parfile, event_table, sheet_name,
                        pre_dir, obs_dir, k2_exe,
                        simuation_duration_multiplier, simuation_duration_variable,
                        simulation_timestep,
                        rain_type=rain_type, flag_col=flag_col,
                        plot_outlet_graph=plot_outlet_graph,
                        group_graphs_by_flag=group_graphs_by_flag,
                        overwrite=overwrite_existing_runs)

    print('Done.')
    logging.info(f'\n\nDone. Total runtime: {((time.perf_counter() - t0)/60.):.2f} minutes')


if __name__ == '__main__':
    main()
