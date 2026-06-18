import os
import logging
import subprocess
import pandas as pd
from k2_output_parser import get_k2_results_outlet

def run_k2(k2_exe, run_dir, parfile, prefile, outfile, kinfile, dur,
           timestep, sediment=True):

    """ run one k2 simulation inside run_dir. run_dir must already contain
        the parfile and prefile. all k2 outputs land in run_dir, so
        concurrent runs in separate dirs cannot collide. """

    sedi_flag = 'Y' if sediment else 'N'
    kin_str = (f'{parfile},{prefile},{outfile}, " ", '
               f'{dur},{timestep},N,{sedi_flag},N,N\n')
    with open(os.path.join(run_dir, kinfile), 'w') as f:
        f.write(kin_str)
    subprocess.run([k2_exe, '-s', kinfile], cwd=run_dir)

def run_k2_batch(tag, k2pool_dir, events, prefiles, parfile,
                 outfiles, kinfiles, durs, k2_exe, timestep,
                 sediment=True, outlet_sim_files=None, overwrite=False):

    """ run a group of events and read outlet totals into a df.
        existing results are skipped (resume behavior) unless overwrite=True,
        in which case every event is rerun and old outputs are replaced.
        returns (df, failed) where failed is a list of (event, reason);
        failures are also written to the log. """

    save_outlet = outlet_sim_files is not None
    if not save_outlet:
        outlet_sim_files = [None] * len(events)

    df = pd.DataFrame()
    failed = []                     # (event, reason)

    for i, (event, prefile, outfile, outletfile, kinfile, dur) in enumerate(
            zip(events, prefiles, outfiles, outlet_sim_files, kinfiles, durs)):

        out_path = os.path.join(k2pool_dir, outfile)

        # an event is done if its result file exists
        done = os.path.exists(out_path)
        if save_outlet:
            done = os.path.exists(outletfile) and os.path.exists(out_path)

        if overwrite or not done:
            print(f'run {tag}, {event}, {i}/{len(events)}')
            # remove old outputs first so a failed rerun can't leave
            # stale files that look like fresh results
            for f in [out_path, outletfile]:
                if f and os.path.exists(f):
                    os.remove(f)
            try:
                run_k2(k2_exe, k2pool_dir, parfile, prefile, outfile, kinfile,
                       dur, timestep, sediment=sediment)
                if save_outlet:
                    os.rename(os.path.join(k2pool_dir, 'OUTLET.SIM'),
                              outletfile)
            except Exception:   # k2 did not run, e.g. missing prefile
                pass
        else:
            print(f'{tag}, {event}, {i}/{len(events)} skipped, outfiles exist')

        if os.path.exists(out_path):
            try:
                df_ = get_k2_results_outlet(out_path)
                df_.insert(0, 'event', event)
                df = pd.concat([df, df_], axis=0)
            except Exception:
                failed.append((event, 'output file could not be parsed'))
        else:
            print(f'{tag}, {event}, {i}/{len(events)} failed')
            failed.append((event, 'no output file (k2 did not run)'))

    if failed:
        msg = f'{len(failed)} of {len(events)} events failed'
        print(msg)
        logging.info(msg)
        for event, reason in failed:
            # print(f'  {event} -- {reason}')
            logging.info(f'  failed: {event} -- {reason}')

    if not df.empty:
        df = df.set_index('event')
    return df, failed