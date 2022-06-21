from datetime import timedelta

import numpy as np

from data_structure import Adn
from os.path import dirname, exists, join
from os import mkdir
from front_end_utilities import down_sample_pdata
from post_process.performance_analyzer import nvv_updown
from mpc import Mpc
import sys
from time import time


from global_definitions import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

original_stdout = sys.stdout
THIS_DIR = dirname(__file__)


def add_gaussian_noise(df_data):
    pass


""" -------------------------------------------------------------------------------------------- """
""" ################################## Input and config ######################################## """
""" -------------------------------------------------------------------------------------------- """
FN_DEFAULT_CONFIG = join(THIS_DIR, '../../experiments/final/default_general_config.csv')
IFOL = '/home/jp/Data/Dropbox/tesis/tests/case_generation'
CASE_NAME_OUT = 'ieee34bus_ev0.1_pv1.0'
CASE_FOLDER = join(IFOL, CASE_NAME_OUT)
CASE_NAME = 'ieee34bus'
OFOL = '/home/jp/Data/Dropbox/tesis/tests/case_generation_scaled'

if not exists(OFOL):
    mkdir(OFOL)

B_PROP_DISTANCE = True
#AR_SCALE_FACTORS = np.linspace(start=0.4, end=0.6, num=4)

L_CASES_NAMES_OUT = [
    #'ieee34bus_ev0.1_pv0.5',
    'ieee34bus_ev0.1_pv1.0',
    #'ieee34bus_ev0.1_pv1.5',
    #'ieee34bus_ev0.2_pv0.5',
    'ieee34bus_ev0.2_pv1.0',
    #'ieee34bus_ev0.2_pv1.5',
    #'ieee34bus_ev0.3_pv0.5',
    'ieee34bus_ev0.3_pv1.0',
    #'ieee34bus_ev0.3_pv1.5'
]

L_CASES_FOLDERS = [join(IFOL, i) for i in L_CASES_NAMES_OUT]

print('Scaling the following cases:')
for i in L_CASES_NAMES_OUT:
    print(i)
print()
for case_folder, case_name_out in zip(L_CASES_FOLDERS, L_CASES_NAMES_OUT):
    ################################################################################################
    # Load pdata, config and noisy downsample                                                      #
    ################################################################################################
    # Read and set default config
    pdata = Adn()
    pdata.read(case_folder, CASE_NAME)
    pdata.read_config(FN_DEFAULT_CONFIG, append=False)

    l_cols_v = [SNAM_V + str(i) for i in pdata.l_buses]

    # Down sample data
    idx_tini = 60 * 2 * 24 * 0
    idx_tiniout = 60 * 2 * 24 * 2
    idx_tend = pdata.df_data.shape[0] - 1

    tini = pdata.df_data.index[idx_tini]
    tend = pdata.df_data.index[idx_tend]
    dt_original = pdata.df_data.index[1] - pdata.df_data.index[0]

    dt_hour = timedelta(hours=1)
    assert (tend - tini + dt_original) % dt_hour == timedelta(0)
    n_hours = (tend - tini + dt_original) // dt_hour
    dt = timedelta(seconds=6)
    n_rh = 30 * (dt_original // dt)
    down_sample_pdata(pdata, n_hours, n_rh, Dt_down=dt, idx_ini=idx_tini)
    pdata.time_config['tiniout'] = tini + timedelta(days=2)

    # Add Gaussian noise
    add_gaussian_noise(pdata.df_data)

    ################################################################################################
    # Achieve target performance metrics                                                           #
    ################################################################################################
    target_up_nvv = 20
    target_down_nvv = 20
    eps_nvv = 5

    factor_df_data = 0.45
    factor_delta = 0.05

    df_data_original = pdata.df_data.copy()
    pdata.df_data = factor_df_data * df_data_original
    keep_iterating = True
    pdata.time_config['tend'] = (pdata.time_config['tiniout'] + timedelta(hours=24)
                                 - pdata.time_config['dt'])

    print('Bucle started')
    t00 = time()
    count = 0
    while keep_iterating:
        # Run IEEE 1547
        mpc = Mpc(pdata)
        sys.stdout = None
        df_sim = mpc.run_ieee_1547()
        sys.stdout = original_stdout

        # Evaluate performance
        nvv_up, nvv_down = nvv_updown(df_sim, Dt_interval=timedelta(minutes=5))
        vmin = df_sim.loc[:, l_cols_v].min().min()
        vmax = df_sim.loc[:, l_cols_v].max().max()

        count += 1
        t11 = time()
        print('bucle\t{0:2}\ttime\t{1:3.3f} minutes'.format(count, (t11 - t00)/60))
        if nvv_up > target_up_nvv + eps_nvv or nvv_down > target_down_nvv + eps_nvv:
            print('nvv_up\t{0:{width}}\tnvv_down\t{1:{width}}\tapplied_factor'
                  '\t{2:1.2f}\tvmax\t{3:1.2f}\tvmin\t{4:1.2f}'.format(
                nvv_up, nvv_down, factor_df_data, vmax, vmin, width=3))
            break
        else:
            factor_df_data += factor_delta
            pdata.df_data = factor_df_data * df_data_original
            print('nvv_up\t{0:{width}}\tnvv_down\t{1:{width}}\tapplied_factor'
                  '\t{2:1.2f}\tvmax\t{3:1.2f}\tvmin\t{4:1.2f}'.format(
                nvv_up, nvv_down, factor_df_data, vmax, vmin, width=3))

    folder_name = join(OFOL, case_name_out + '_scaled')
    if not exists(folder_name):
        mkdir(folder_name)
    pdata.write(folder_name, CASE_NAME)
