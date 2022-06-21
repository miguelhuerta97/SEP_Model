from datetime import timedelta
import pandas as pd
from data_structure import Adn
from front_end_utilities import down_sample_pdata
import numpy as np
from global_definitions import *
import re
from os.path import exists
from os import mkdir
from case_generation.pdata_scaling import scale_data_ratios, scale_data_target_avgvv

CASE_NAME = 'ieee123bus'


# ------------------------------------------------------------------------------------------------ #
# Load pdata and preliminaries
# ------------------------------------------------------------------------------------------------ #
def create_ieee123bus(case_folder, case_folder_seed):
    if not exists(case_folder):
        mkdir(case_folder)

    pdata = Adn()
    pdata.read(case_folder_seed, CASE_NAME)
    df_data = pdata.df_data

    # no b
    pdata.branches.loc[:, 'b'] = 0.
    # No current limit
    pdata.branches.loc[:, 'imax'] = 999.

    # -------------------------------------------------------------------------------------------- #
    # DG location
    # -------------------------------------------------------------------------------------------- #
    N_DGS = 70
    EPS_LOAD = 0.0001
    set_buses_loads = set(pdata.buses[pdata.buses['loadp_mw'].abs() > EPS_LOAD].index.to_list())
    se_n = pdata.se_distance_to_root()
    set_buses_near_leaves = set(se_n[se_n > 6].index.to_list())
    set_buses_dgs = set_buses_loads & set_buses_near_leaves
    ar_buses_dgs = np.asarray(list(set_buses_dgs))
    np.random.seed(0)
    ar_buses_dgs = np.random.choice(ar_buses_dgs, N_DGS, replace=False)
    ar_buses_dgs = np.sort(ar_buses_dgs)

    l_dgs = list(range(N_DGS))
    df_dgs = pd.DataFrame(index=l_dgs, columns=pdata.dgs.columns, dtype='float64')
    df_dgs.loc[:, 'snom'] = 1.
    df_dgs.loc[:, 'bus'] = ar_buses_dgs

    pdata.grid_tables['dgs'] = df_dgs
    # Setting dgpmax columns in df_data
    l_cols_loaddgp_dgs = [SNAM_LOADDGP + str(i) for i in ar_buses_dgs]
    l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in l_dgs]

    df_dgpmax = df_data.loc[:, l_cols_loaddgp_dgs].copy()
    df_dgpmax.columns = l_cols_dgpmax
    df_dgpmax.clip(lower=0., inplace=True)

    l_cols_loadp = [i for i in df_data.columns if i.startswith(SNAM_LOADP)]
    l_cols_loadq = [i for i in df_data.columns if i.startswith(SNAM_LOADQ)]
    l_cols_loadevp = [i for i in df_data.columns if i.startswith(SNAM_LOADEVP)]
    l_cols_loaddgp = [i for i in df_data.columns if i.startswith(SNAM_LOADDGP)]

    l_cols_loads = l_cols_loadp + l_cols_loadq + l_cols_loadevp +  l_cols_loaddgp

    df_data_loads = df_data.loc[:, l_cols_loads]

    df_data_new = pd.concat([df_data_loads, df_dgpmax], axis=1)

    pdata.df_data = df_data_new

    # -------------------------------------------------------------------------------------------- #
    # Down sample data
    # -------------------------------------------------------------------------------------------- #
    dt_data = pdata.df_data.index[1] - pdata.df_data.index[0]
    dt_total = pdata.df_data.index[-1] - pdata.df_data.index[0] + dt_data
    dt_hour = timedelta(hours=1)
    n_hours = dt_total // dt_hour
    dt_down = timedelta(seconds=6)

    down_sample_pdata(pdata, n_hours, n_rh=150, Dt_down=dt_down, idx_ini=0)

    # -------------------------------------------------------------------------------------------- #
    # Validation
    # -------------------------------------------------------------------------------------------- #
    # Order
    assert pdata.dgs.index[0] == 0
    assert np.all(pdata.dgs.index[1:] - pdata.dgs.index[:-1] == 1)

    l_buses_loads = [int(re.search('[a-z]+([0-9]+)', i).group(1)) for i in df_data.columns if i.startswith(SNAM_LOADP)]
    ar_buses_loads = np.asarray(l_buses_loads)
    assert np.all((ar_buses_loads[1:] - ar_buses_loads[:-1]) >= 1)
    assert np.all(pdata.df_data.loc[:, l_cols_dgpmax] >= 0.)

    # -------------------------------------------------------------------------------------------- #
    # Write data
    # -------------------------------------------------------------------------------------------- #
    pdata.write(case_folder, CASE_NAME)


def scale_ieee123bus(case_folder, case_folder_seed, time_config=None):
    if not exists(case_folder):
        mkdir(case_folder)

    ratio_snom_pmax = 1.2

    pdata = Adn()
    pdata.read(case_folder_seed, CASE_NAME)

    se_buses_dgs = pdata.dgs['bus']
    l_cols_loadp = [i for i in pdata.df_data.columns if i.startswith(SNAM_LOADP)]
    ar_loadp_nom = pdata.df_data.loc[:, l_cols_loadp].max().values
    scale_data_ratios(pdata.df_data, ar_loadp_nom * 2., se_buses_dgs, ratio_loadevp=0.1,
                      ratio_loaddgp=0.5, ratio_dgpmax=0.5)

    l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in pdata.l_dgs]
    ar_dgpmax = pdata.df_data.loc[:, l_cols_dgpmax].max().values
    pdata.dgs.loc[:, 'snom'] = ratio_snom_pmax * ar_dgpmax


    if time_config is not None:
        pdata.time_config = time_config
        scale_data_target_avgvv(pdata, target_avgvv=1e-5, avgvv_eps=1e-7, m=1., b=0.05)
    pdata.write(case_folder, CASE_NAME)
