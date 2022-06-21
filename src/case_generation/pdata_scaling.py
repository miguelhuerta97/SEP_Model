import re
import sys
from datetime import timedelta
from time import time

import numpy as np
import pandas as pd

import data_structure as ds
from errors import InputError

from front_end_utilities import FLOAT_EPS
from global_definitions import SNAM_LOADP, SNAM_LOADDGP, SNAM_LOADEVP, SNAM_LOADQ, SNAM_DGPMAX, \
    SNAM_V
from mpc import Mpc
from post_process import nvv_updown
from post_process.performance_analyzer import calculate_avg_vv_pu


def scale_data_ratios(df_data: pd.DataFrame, ar_fact_loadp: np.ndarray, se_buses_dgs: pd.Series,
                      ratio_loadevp: float, ratio_loaddgp: float, ratio_dgpmax: float,
                      power_factor=0.9, float_eps=1e-7):
    """
    Rescale de data. All ratios are with respect to the accum_loadp. Preserving accum_loadp
    :param se_buses_dgs: {numpy.ndarray} Series of dg buses indices
    :param df_data: {pandas.DataFrame}
    :param ar_fact_loadp: {numpy.ndarray}
    :param ratio_loadevp: {float}
    :param ratio_loaddgp: {float}
    :param ratio_dgpmax: {float}
    :param power_factor: {float}
    :param float_eps: {float}
    """
    if not isinstance(ar_fact_loadp, np.ndarray):
        raise InputError('ar_fact_loadp must be np.ndarray!')
    ################################################################################################
    # Reduce the space to non-zero elems
    ################################################################################################
    #ar_buses = np.asarray([int(re.search('{}([0-9]+)'.format(SNAM_LOADP), i).group(1))
    #                      for i in df_data.columns if i.startswith(SNAM_LOADP)])

    #set_buses = set(ar_buses)
    #ar_buses_load_notnull = ar_buses[abs(ar_fact_loadp) > FLOAT_EPS]

    # Eliminate load null columns
    #set_buses_load_null = set_buses.difference(set(ar_buses_load_notnull))
    #set_cols_loadp_null = {SNAM_LOADP + str(i) for i in set_buses_load_null}
    #set_cols_loaddgp_null = {SNAM_LOADDGP + str(i) for i in set_buses_load_null}
    #set_cols_loadevp_null = {SNAM_LOADEVP + str(i) for i in set_buses_load_null}
    #set_cols_loadq_null = {SNAM_LOADQ + str(i) for i in set_buses_load_null}
    #set_cols_load_null = set.union(set_cols_loadp_null, set_cols_loadevp_null,
    #                               set_cols_loaddgp_null, set_cols_loadq_null)
    #df_data.drop(set_cols_load_null, axis=1, inplace=True)

    # Eliminate dgpmax columns corresponding to no-dg buses
    #l_buses_dgs = se_buses_dgs.to_list()
    #l_buses_no_dgs = list(set_buses.difference(set(l_buses_dgs)))
    #l_cols_dgpmax_nodg = [SNAM_DGPMAX + str(i) for i in l_buses_no_dgs]
    #df_data.drop(l_cols_dgpmax_nodg, axis=1, inplace=True, errors='ignore')

    #   Mapping bus based col names to l_dgs based col names (this is not necessary for load cols)
    #l_dgs = se_buses_dgs.index.to_list()
    #l_cols_dgpmax_prev = [SNAM_DGPMAX + str(i) for i in l_buses_dgs]
    #l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in l_dgs]
    #map_cols_bus2dg = dict(zip(l_cols_dgpmax_prev, l_cols_dgpmax))
    #df_data.rename(columns=map_cols_bus2dg, inplace=True)

    ################################################################################################
    # Calculate and apply scaling factors
    ################################################################################################
    l_cols_loadp = [i for i in df_data.columns if i.startswith(SNAM_LOADP)]
    l_cols_loadq = [i for i in df_data.columns if i.startswith(SNAM_LOADQ)]
    l_cols_dgpmax = [i for i in df_data.columns if i.startswith(SNAM_DGPMAX)]
    l_cols_loadevp = [i for i in df_data.columns if i.startswith(SNAM_LOADEVP)]
    l_cols_loaddgp = [i for i in df_data.columns if i.startswith(SNAM_LOADDGP)]

    # Creating copies of the df_data components
    ar_loadp = df_data.loc[:, l_cols_loadp].copy().values
    ar_loadevp = df_data.loc[:, l_cols_loadevp].copy().values
    ar_loaddgp = df_data.loc[:, l_cols_loaddgp].copy().values
    ar_dgpmax = df_data.loc[:, l_cols_dgpmax].copy().values

    # Apply formula to obtain loadp_base component
    ar_loadp_base = ar_loadp - ar_loadevp + ar_loaddgp

    # Apply columnwise factors
    ar_fact_loadp_after = ar_fact_loadp[ar_fact_loadp > float_eps]
    ar_loadp = ar_loadp * ar_fact_loadp_after
    ar_loadevp = ar_loadevp * ar_fact_loadp_after
    ar_loaddgp = ar_loaddgp * ar_fact_loadp_after
    ar_total_demand = ar_loadp + ar_loaddgp

    # Integrate to obtain the total energy of each component
    accum_loadp = ar_loadp.sum()
    accum_loadp_base = ar_loadp_base.sum()
    accum_loadevp = ar_loadevp.sum()
    accum_loaddgp = ar_loaddgp.sum()
    accum_dgpmax = ar_dgpmax.sum()
    accum_total_demand = ar_total_demand.sum()

    # Define target integral value
    target_accum_total_demand = accum_total_demand
    target_accum_loadevp = ratio_loadevp * accum_total_demand
    target_accum_loaddgp = ratio_loaddgp * accum_total_demand
    target_accum_dgpmax = ratio_dgpmax * accum_total_demand
    target_accum_loadp_base = target_accum_total_demand - target_accum_loadevp

    # Base on the integrals, calculate scaling factors
    factor_loadevp = target_accum_loadevp / accum_loadevp
    factor_loaddgp = target_accum_loaddgp / accum_loaddgp
    factor_dgpmax = target_accum_dgpmax / accum_dgpmax

    # Log factors
    l_factors = [factor_loadevp, factor_loaddgp, factor_dgpmax]
    l_names = ['factor_loadevp', 'factor_loaddgp', 'factor_dgpmax']
    for i, j in zip(l_names, l_factors):
        print('{}: {}'.format(i, j))

    # Apply factors
    ar_loadevp = factor_loadevp * ar_loadevp
    ar_loaddgp = factor_loaddgp * ar_loaddgp
    ar_dgpmax = factor_dgpmax * ar_dgpmax

    # Apply power factor to obtain loadq
    phi = np.arccos(power_factor)
    ar_loadq = ar_loadp * np.tan(phi)

    # Apply formula to obtain new loadp
    ar_loadp = ar_total_demand - ar_loaddgp

    df_data.loc[:, l_cols_loadp] = ar_loadp
    df_data.loc[:, l_cols_loadevp] = ar_loadevp
    df_data.loc[:, l_cols_loaddgp] = ar_loaddgp
    df_data.loc[:, l_cols_dgpmax] = ar_dgpmax
    df_data.loc[:, l_cols_loadq] = ar_loadq


def scale_data_target(pdata: ds.Adn, target_nvv: float,
                      nvv_Dt_interval: timedelta = timedelta(minutes=5), nvv_eps: int = 10,
                      m: float = 1., b: float = 0.05, max_iterations: int = 10,
                      ratio_snom_pmax: float = 1.2):

    # Check if time config integrity
    if pdata.time_config is None:
        raise InputError('Time config is not defined')
    for v in pdata.time_config.values():
        if v is None:
            raise InputError('Time config is not defined')

    # Definitions
    original_stdout = sys.stdout
    l_cols_v = [SNAM_V + str(i) for i in pdata.l_buses]
    factor_df_data = 1.
    df_data_original = pdata.df_data.copy()

    # Main loop
    print('Bucle started')
    t00 = time()
    for count in range(max_iterations):
        # Update and apply factor
        factor_df_data = factor_df_data * m + b
        pdata.df_data = factor_df_data * df_data_original

        # Run IEEE 1547
        mpc = Mpc(pdata)
        sys.stdout = None
        df_sim = mpc.run_ieee_1547()
        sys.stdout = original_stdout

        # Evaluate performance
        nvv = sum(nvv_updown(df_sim, Dt_interval=nvv_Dt_interval))
        vmin = df_sim.loc[:, l_cols_v].min().min()
        vmax = df_sim.loc[:, l_cols_v].max().max()

        # Report performance
        print('nvv\t{0:{width}}\tapplied_factor'
              '\t{1:1.2f}\tvmax\t{2:1.2f}\tvmin\t{3:1.2f}'.format(
            nvv, factor_df_data, vmax, vmin, width=3))

        # Report loop end
        t11 = time()
        print('bucle\t{0:2}\ttime\t{1:3.3f} minutes'.format(count, (t11 - t00) / 60))

        # Evaluate terminating condition
        if (nvv >= target_nvv - nvv_eps) or (nvv <= target_nvv + nvv_eps):
            print('Scaling terminated: target condition')
            return

    print('Scaling terminated: max_iterations condition')

    l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in pdata.l_dgs]
    ar_dgpmax = pdata.df_data.loc[:, l_cols_dgpmax].max().values
    pdata.dgs.loc[:, 'snom'] = ratio_snom_pmax * ar_dgpmax
    print('snom values for dgs have been scaled by a factor of: {}'.format(ratio_snom_pmax))


def scale_data_target_avgvv(pdata: ds.Adn, target_avgvv: float, avgvv_eps: float = 1e-5,
                      m: float = 1., b: float = 0.05, max_iterations: int = 10,
                      ratio_snom_pmax: float = 1.2, vlim_r: float = 0.05):
    # Check if time config integrity
    if pdata.time_config is None:
        raise InputError('Time config is not defined')
    for v in pdata.time_config.values():
        if v is None:
            raise InputError('Time config is not defined')

    # Definitions
    original_stdout = sys.stdout
    l_cols_v = [SNAM_V + str(i) for i in pdata.l_buses]
    factor_df_data = 1.
    df_data_original = pdata.df_data.copy()

    # Main loop
    print('Bucle started')
    t00 = time()
    for count in range(max_iterations):
        # Update and apply factor
        factor_df_data = factor_df_data * m + b
        pdata.df_data = factor_df_data * df_data_original

        # Run IEEE 1547
        mpc = Mpc(pdata)
        sys.stdout = None
        df_sim = mpc.run_ieee_1547()
        sys.stdout = original_stdout

        # Evaluate performance
        avg_vv = calculate_avg_vv_pu(df_sim, vlim_r=vlim_r)
        vmin = df_sim.loc[:, l_cols_v].min().min()
        vmax = df_sim.loc[:, l_cols_v].max().max()

        # Report performance
        print('avg_vv\t{0:{width}}\tapplied_factor'
              '\t{1:1.2f}\tvmax\t{2:1.2f}\tvmin\t{3:1.2f}'.format(
            avg_vv, factor_df_data, vmax, vmin, width=3))

        # Report loop end
        t11 = time()
        print('bucle\t{0:2}\ttime\t{1:3.3f} minutes'.format(count, (t11 - t00) / 60))

        # Evaluate terminating condition
        if (avg_vv <= target_avgvv + avgvv_eps):
            print('Scaling terminated: target condition')
            return

    print('Scaling terminated: max_iterations condition')

    l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in pdata.l_dgs]
    ar_dgpmax = pdata.df_data.loc[:, l_cols_dgpmax].max().values
    pdata.dgs.loc[:, 'snom'] = ratio_snom_pmax * ar_dgpmax
    print('snom values for dgs have been scaled by a factor of: {}'.format(ratio_snom_pmax))