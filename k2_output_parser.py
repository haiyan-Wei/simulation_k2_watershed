""" Parse K2 .out output files into DataFrames.

Pure reading: nothing in this module runs K2 or writes files.

Public:
- get_k2_results_outlet     : event totals at the outlet (one-row df)
- get_k2_results_by_element : rainfall/runoff/sediment for every plane
                              and channel element (one row per element)
"""

import re
import numpy as np
import pandas as pd


def get_float_from_string(string, loc):
    """ return the loc-th number in the string as float
        (tries decimal numbers first, then integers) """
    try:
        value = float(re.findall(r'\d+\.\d+', string)[loc])
    except IndexError:
        value = float(re.findall(r'\d+', string)[loc])
    return value


def get_int_from_string(string, loc):
    """ return the loc-th integer in the string """
    return int(re.findall(r'\d+', string)[loc])


def get_k2_results_outlet(outfile):

    """ get event totals at the outlet from a k2 .out file.
        returns a one-row df. values not found in the file are NaN,
        except runoff/infiltration/sediment-yield which default to 0
        (k2 omits them from the summary when they are zero). """

    with open(outfile) as f:
        lines = f.readlines()

    # not reported by k2 when zero -> default 0
    runoff_mm, infil_chan_mm, infil_plane_mm, sedi_tonha = 0., 0., 0., 0.
    # always reported -> NaN means the file is incomplete
    area_ha, rain_mm, peak_mmhr, sedi_kg = [np.nan] * 4

    for line in lines[-50:]:
        if 'Peak flow =' in line:
            peak_mmhr = get_float_from_string(line, 1)
        if 'Rainfall' in line and 'mm' in line:   # includes interception
            rain_mm = get_float_from_string(line, 0)
        if 'Outflow' in line:
            runoff_mm = get_float_from_string(line, 0)
        if 'Out' in line and 'kg' in line:
            # total kg, not kg/ha or t/ha
            sedi_kg = get_float_from_string(line.split('Out:')[1], 0)
        if 'Plane infiltration' in line:
            infil_plane_mm = get_float_from_string(line, 0)
        if 'Channel infiltration' in line:
            infil_chan_mm = get_float_from_string(line, 0)
        if 'Total watershed area' in line:
            area_ha = get_float_from_string(line, 0)
        if 'Sediment yield =' in line:
            sedi_tonha = get_float_from_string(line, 0)
            if 'kg/ha' in line:
                sedi_tonha = sedi_tonha / 1000.

    return pd.DataFrame([{'sim_area_ha': area_ha,
                          'sim_rain_mm': rain_mm,
                          'sim_runoff_mm': runoff_mm,
                          'sim_peak_mmhr': peak_mmhr,
                          'sim_sedi_kg': sedi_kg,
                          'sim_infil_plane_mm': infil_plane_mm,
                          'sim_infil_chan_mm': infil_chan_mm,
                          'sim_sedi_tonha': sedi_tonha,
                          }])


def get_k2_results_by_element(outfile):

    """ get rainfall, runoff and sediment at all elements from a .out file.
    returns one df containing planes and channels, in routing sequence.
    variables: element type and ID, contributing area (ha), peak (mm/hr),
    rainfall (mm), runoff (mm), and sediment terms (kg and kg/ha).

    NOTE: parses fixed line offsets within each element block -- verify
    against the .out file if the k2 version (and output format) changes. """

    def get_sim_for_element(ele_type, outblock):

        eleid = get_int_from_string(outblock[0], 0)
        area = get_float_from_string(outblock[2], 0)
        peak = get_float_from_string(outblock[10], 1)

        if ele_type == 'plane':
            rain = get_float_from_string(outblock[16], 1)
            sedi_in = get_float_from_string(outblock[16], 2)
            deposit = get_float_from_string(outblock[17], 2)
            sloss = get_float_from_string(outblock[18], 2)
            sedi_out = get_float_from_string(outblock[19], 2)
            runoff = get_float_from_string(outblock[20], 1)

        elif ele_type == 'channel':
            # rainfall reported as a volume (cu m) -> convert to mm
            rain = get_float_from_string(outblock[16], 0) / area / 10000 * 1000
            sedi_in = get_float_from_string(outblock[16], 1)
            deposit = get_float_from_string(outblock[17], 1)
            sloss = get_float_from_string(outblock[18], 1)
            sedi_out = get_float_from_string(outblock[19], 1)
            runoff = get_float_from_string(outblock[20], 0)

        return pd.DataFrame(
            data=[[ele_type, eleid, area, peak,
                   rain, deposit, sloss,
                   sedi_in, sedi_out, runoff]],
            columns=['eleType', 'eleID', 'area_ha', 'peak_mmhr',
                     'rain_mm', 'deposit_kg', 'sloss_kg',
                     'sedi_in_kg', 'sedi_out_kg', 'runoff_mm'])

    with open(outfile) as f:
        lines = f.readlines()

    df = pd.DataFrame()

    for i, line in enumerate(lines):

        if 'Plane Element   ' in line:
            outblock = lines[i:i + 21]
            df_ = get_sim_for_element('plane', outblock)
            df = pd.concat([df, df_], axis=0)

        if 'Channel Elem.   ' in line and 'rating exceeded' not in line:
            outblock = lines[i:i + 21]
            df_ = get_sim_for_element('channel', outblock)
            df = pd.concat([df, df_], axis=0)

    df = df.reset_index(drop=True)
    df = df.assign(sedi_out_kgha=df.sedi_out_kg / df.area_ha,
                   deposit_kgha=df.deposit_kg / df.area_ha,
                   sloss_kgha=df.sloss_kg / df.area_ha)

    return df