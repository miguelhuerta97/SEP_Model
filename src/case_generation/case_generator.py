import math
import gc
import numpy as np
import pandas as pd

from os import mkdir
from os.path import join, exists

from datetime import timedelta
from case_generation.ts_electric_vehicle_demand import ev_demand
import data_structure as ds
from global_definitions import *
from mpc import Mpc
from front_end_utilities import write_df_result
from post_process import write_report
import subprocess

# ISSUE:
#    - Dg location is arbitrary
#    - ts_dgpmax and ts_loaddgp are incoherent. one comes from NREL and the other from pecan
#    - ts_evload is very similar between generated samples

# Input
CASE_FOLDER = '/home/jp/tesis/cases_seed/src_v2_cases'
CASE_NAME = 'case33bw'
# CASE_NAME = 'case4_dist'

FOLDER_SEED_DATA = '/home/jp/Data/Dropbox/tesis/data_seed/pecan_case_specific_chunks/bw33'
# DATA_NAME = '4bus_dist_0_data.csv'
DATA_NAME = 'bw33_0_data.csv'

OFOL = '/home/jp/tesis/experiments/bw33_chunk_0'
if not exists(OFOL):
    mkdir(OFOL)

# System grid
pdata = ds.Adn()
pdata.read(CASE_FOLDER, CASE_NAME)

pdata.grid_tables['data'] = None
gc.collect()

pdata.set('cost_nse', 10000.)
pdata.set('cost_putility', 1.)
pdata.set('cost_losses', 0.01)
pdata.set('cost_vlim', 10000.)
pdata.set('ctrl_active', True)
pdata.set('ctrl_project', True)

pdata.relabel_branches()
pdata.grid_tables['b2bs'] = pd.DataFrame()
pdata.branches.loc[:, 'imax'] = 1.

# Preliminary definitions
n_buses = len(pdata.l_buses)
n_loads = min((pdata.buses['loadp_mw'] > 0.).sum(), n_buses)

l_buses = pdata.l_buses
l_load_buses = (pdata.buses['loadp_mw'][pdata.buses['loadp_mw'] > 0.]).index.to_list()
l_cols_loadp = [SNAM_LOADP + str(i) for i in l_load_buses]
l_cols_loadq = [SNAM_LOADQ + str(i) for i in l_load_buses]
l_cols_load = l_cols_loadp + l_cols_loadq
l_cols_loaddgp = [SNAM_LOADDGP + str(i) for i in l_load_buses]
l_cols_loadevp = [SNAM_LOADEVP + str(i) for i in l_load_buses]
l_cols_load_related = l_cols_load + l_cols_loadevp + l_cols_loaddgp
l_cols_v = [SNAM_V + str(i) for i in l_buses]

# Demands
factors0_loadp: np.ndarray = pdata.buses['loadp_mw'][l_load_buses].values / pdata.sbase_mva
factors0_loadq = pdata.buses['loadq_mvar'][l_load_buses].values / pdata.sbase_mva

# Load data
df_data_seed = pd.read_csv(join(FOLDER_SEED_DATA, DATA_NAME), sep='\t', index_col='time')
df_data_seed.index = pd.to_datetime(df_data_seed.index)


""" ############################################################################################ """
""" ####################################### Time config ########################################"""
""" ############################################################################################ """
n_hist_days = 1
tini = df_data_seed.index[0] + timedelta(days=n_hist_days)
# idx_tend = min(1440 * 3 - 1, df_data_seed.shape[0] - 1)
n_days = (df_data_seed.index[-1] - df_data_seed.index[0]) // timedelta(days=1)
#n_days = 2
idx_tend = n_days * 1440 - 1
tend = df_data_seed.index[idx_tend]
dt = df_data_seed.index[1] - df_data_seed.index[0]

pdata.time_config = {
    'tini': tini,
    'tend': tend,
    'dt': dt,
    'n_rh': 60
}

trange = pdata.time_map(2)
tini_all = pdata.time_config['tini'] - timedelta(days=n_hist_days)
trange_all = pd.date_range(start=tini_all, end=pdata.time_config['tend'],
                           freq=pdata.time_config['dt'])
assert ((trange_all[-1] + pdata.time_config['dt'] - trange_all[0]) % timedelta(days=1)
        == timedelta(0))


""" ############################################################################################ """
""" ############################ DGs location and solar power series ########################### """
""" ############################################################################################ """
# Dg location
se_n = pdata.se_distance_to_root()
l_dg_buses = se_n[se_n > 5].index.to_list()
# l_dg_buses = l_buses
n_dgs = len(l_dg_buses)
df_dgs = pd.DataFrame(
    {'bus': l_dg_buses, 'snom': [10. for i in l_dg_buses]}, columns=['bus', 'snom'])
df_dgs.loc[:, 'bus'] = df_dgs.loc[:, 'bus'].astype('int64')
df_dgs.loc[:, 'snom'] = df_dgs.loc[:, 'snom'].astype('float64')
pdata.grid_tables['dgs'] = df_dgs

l_dgs = pdata.l_dgs
l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in l_dgs]

df_dgpmax = df_data_seed.loc[:, l_cols_dgpmax]
factors_dgpmax = 0.02 / df_data_seed.loc[:, l_cols_dgpmax].max()
assert factors_dgpmax.shape[0] == len(l_cols_dgpmax)

df_dgpmax = factors_dgpmax * df_dgpmax
df_dgpmax[df_dgpmax < 0.] = 0.


""" ############################################################################################ """
""" ################################ Load components manipulation ############################## """
""" ############################################################################################ """
# Target ratios
target_loadevp = 0.3
target_loaddgp = 0.3
target_base_load = 1 - target_loadevp - target_loaddgp
assert target_base_load > 0.

# Obtain load data components
df_loaddgp = df_data_seed.loc[trange_all, l_cols_loaddgp]
df_base_loadp = df_data_seed.loc[trange_all, l_cols_loadp] - df_loaddgp.values
df_base_loadq = df_data_seed.loc[trange_all, l_cols_loadq]
df_data_seed = None
gc.collect()

df_evload_seed = ev_demand(n_days, trange_all[0], n_samples=1)
assert df_evload_seed.index.equals(trange_all)
df_loadevp = pd.DataFrame(index=df_evload_seed.index, columns=l_cols_loadevp, dtype='float64')
df_loadevp.loc[:, l_cols_loadevp] = df_evload_seed.values

# Apply target factors
total_loadevp = df_loadevp.sum().sum()
total_loaddgp = df_loaddgp.sum().sum()
total_base_loadp = df_base_loadp.sum().sum()
total_load = total_base_loadp + total_loadevp + total_loaddgp

factor_loadevp = target_loadevp * total_load / total_loadevp
factor_loaddgp = target_loaddgp * total_load / abs(total_loaddgp)
factor_base_loadp = target_base_load * total_load / total_base_loadp

df_loadevp = df_loadevp * factor_loadevp
df_loaddgp = df_loaddgp * factor_loaddgp
df_base_loadp = df_base_loadp * factor_base_loadp

df_loadp = df_base_loadp + df_loadevp.values - df_loaddgp.values
df_loadq = df_base_loadq

# Apply pdata.buses factors (this also defines the loads power factors)
factors_loadp = (factors0_loadp / df_loadp.max()).values
factors_loadq = (factors0_loadq / df_loadq.max()).values
df_loaddgp = df_loaddgp * factors_loadp
df_loadevp = df_loadevp * factors_loadp
df_loadp = df_loadp * factors_loadp
df_loadq = df_loadq * factors_loadq
df_data_loads = pd.concat([df_loadp, df_loadq, -df_loaddgp, df_loadevp], axis=1)


""" ############################################################################################ """
""" ######################################### Main loop ######################################## """
""" ############################################################################################ """
assert set(l_cols_load_related) == set(df_data_loads.columns.to_list())
# Get together pdata.df_data
l_cols_data = l_cols_loadp + l_cols_loadq + l_cols_dgpmax + l_cols_loaddgp + l_cols_loadevp
pdata.df_data = pd.DataFrame(0., index=trange_all, columns=l_cols_data, dtype='float64')

# Set initial factors
load_factor = 1.15
solar_factor = 0.09

pdata.df_data.loc[:, l_cols_load_related] = df_data_loads * load_factor
pdata.df_data.loc[:, l_cols_dgpmax] = df_dgpmax * solar_factor

# Sizing the dgs
pdata.dgs.loc[:, 'snom'] = pdata.df_data.loc[:, l_cols_dgpmax].max().max() * 3

load_step = 0.05
solar_step = 0.005

v_lb_ref = 0.94
v_ub_ref = 1.06

eps = 0.002
n_max_iter = 15
n_iter = 0

try:
    while True:
        print('LOG:load_factor\t{}\tsolar_factor\t{}'.format(load_factor, solar_factor))

        mpc = Mpc(pdata)
        # df_sim, _ = mpc.run_noctrl()
        df_sim = mpc.run_ieee_1547()

        v_ub = df_sim.loc[:, l_cols_v].max().max()
        v_lb = df_sim.loc[:, l_cols_v].min().min()

        F_load = v_lb - v_lb_ref
        F_solar = v_ub - v_ub_ref

        load_achieved = abs(F_load) < eps
        solar_achieved = abs(F_solar) < eps

        with open(join(OFOL, 'summary_{}.txt'.format(n_iter)), 'w') as hfile:
            str_report = write_report(pdata, df_sim, mpc)
            hfile.write(str_report)

        write_df_result(df_sim, join(OFOL, 'df_sim_{}.csv'.format(n_iter)))
        fol_case = join(OFOL, '{}_{}'.format(CASE_NAME, n_iter))
        if not exists(fol_case):
            mkdir(fol_case)
        pdata.write(fol_case, '{}_{}'.format(CASE_NAME, n_iter))

        if load_achieved and solar_achieved:
            break

        if not load_achieved:
            load_factor += math.copysign(load_step, F_load)

        if not solar_achieved:
            solar_factor += math.copysign(solar_step, - F_solar)
            # loaddgp_factor += math.copysign(solar_step, - F_solar)

        n_iter += 1
        if n_iter >= n_max_iter:
            break

        # Apply factors
        pdata.df_data.loc[:, l_cols_load_related] = df_data_loads * load_factor
        pdata.df_data.loc[:, l_cols_dgpmax] = df_dgpmax * solar_factor
        # Resizing the dgs
        pdata.dgs.loc[:, 'snom'] = pdata.df_data.loc[:, l_cols_dgpmax].max().max() * 3

finally:
    ...
    # subprocess.call(['shutdown', '+0'])