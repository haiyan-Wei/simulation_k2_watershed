import os
import sys
import inspect
import logging
import pandas as pd
from datetime import datetime


def log_variables(description, vars):
    """Write a description and a dict of {name: value} pairs to the log."""
    logging.info(description)
    for name, val in vars.items():
        logging.info(f'{name}: {val}')
    logging.info('\n\n')



def create_loginfo_file(logname, script_goal, t0):

    """ create a log file (level=INFO); print script location and goals """

    logging.basicConfig(filename=logname, level=logging.INFO,
        format='%(message)s', filemode='w')

    script_loc = sys.argv[0]
    msg = (
        'Please turn off the option of "wrap text" to view.\n'
        'This is a log file generated with the following script:\n'
        f'{script_loc}\n\nObjectives:\n{script_goal}\n\n'
        f'Started at: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n')
    logging.info(msg)





