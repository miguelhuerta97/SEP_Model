from global_definitions import *
import pandas as pd
import numpy as np
from numpy.random import choice
from os.path import join, exists
from os import makedirs
from datetime import datetime, timedelta
from data_structure import Adn
from mpc import Mpc


CASE_NAME = 'ieee34bus'
CASE_FOLDER = '/home/jp/tesis/experiments/cases_final/ieee34bus'
OFOL = '/home/jp/tesis/experiments/computational_tractability'

if not exists(OFOL):
    makedirs(OFOL)

FN_SOLVE_LOG = join(OFOL, 'times')

TIME_CONFIG = {
    'tini': datetime(2018, 7, 27, 0, 0, 0),
    'tiniout': datetime(2018, 8, 11, 12, 0, 0),
    'tend': datetime(2018, 8, 11, 12, 14, 54),
    'dt': timedelta(seconds=6),
    'n_rh': 150
}

pdata = Adn()
pdata.read(CASE_FOLDER, CASE_NAME)

n_sce = 500
n_win_ahead = 0
n_win_before = 0
n_days = 5
n_days_delay = 0

cost_lasso_x = 0.0
fn_adalasso_weights = 0.

# ---------------------------------------------------------------------------------------- #
# General configuration
# ---------------------------------------------------------------------------------------- #
pdata.time_config = TIME_CONFIG

pdata.set('log_folder', None)

pdata.set('cost_vlim', 1e5)
pdata.set('cost_putility', 1.)
pdata.set('cost_lasso_v', 1e-11)

pdata.set('ctrl_robust', True)
pdata.set('risk_measure', 'worst_case')
pdata.set('scegen_type', 'ddus_kmeans')
pdata.set('ctrl_robust_n_sce', n_sce)
pdata.set('scegen_n_win_ahead', n_win_ahead)
pdata.set('scegen_n_win_before', n_win_before)
pdata.set('scegen_n_days', n_days)
pdata.set('scegen_n_days_delay', n_days_delay)

pdata.set('cost_lasso_x', cost_lasso_x)
pdata.set('fn_adalasso_weights', None)


l_buses = pdata.l_buses
pdata.set('bilinear_approx', 'mccormick')
pdata.buses.loc[l_buses, 'mc_dv_ub'] = 0.09
pdata.buses.loc[l_buses, 'mc_dv_lb'] = -0.09
pdata.buses.loc[l_buses, 'mc_gp_lb'] = -0.3
pdata.buses.loc[l_buses, 'mc_gq_lb'] = -0.3

pdata.set('cost_vlim', 1.e5)
pdata.set('ctrl_type', 'droop_polypol')
pdata.set('polypol_deg', 3)


# ---------------------------------------------------------------------------------------- #
# Run experiment
# ---------------------------------------------------------------------------------------- #
mean_snom = pdata.dgs.loc[:, 'snom'].mean()

pdata.buses.loc[l_buses, 'vmin'] = 0.9
pdata.buses.loc[l_buses, 'vmax'] = 1.1

n_buses = len(l_buses)


ar_buses = np.asarray(l_buses)

df_data = pdata.df_data

l_cols_dgpmax = [i for i in df_data.columns if i.startswith(SNAM_DGPMAX)]

ar_dgpmax = df_data.loc[:, l_cols_dgpmax].mean(axis=1).values.reshape((df_data.shape[0], 1))

df_data.drop(columns=l_cols_dgpmax, inplace=True)

l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in range(n_buses)]
df_dgpmax = pd.DataFrame(index=df_data.index, columns=l_cols_dgpmax, dtype='float64')
df_dgpmax.loc[:, :] = ar_dgpmax

df_data = pd.concat([df_data, df_dgpmax], axis=1)
pdata.df_data = df_data

for n in range(1, n_buses):
    dg_buses = choice(ar_buses, n, replace=False)
    l_dgs = range(n)
    df_dgs = pd.DataFrame(index=l_dgs, columns=['bus', 'snom'])
    df_dgs.loc[:, 'bus'] = dg_buses
    df_dgs.loc[:, 'bus'] = df_dgs.loc[:, 'bus'].astype('int64')

    df_dgs.loc[:, 'snom'] = mean_snom

    pdata.grid_tables['dgs'] = df_dgs

    fn_log_file = join(OFOL, 'log_{}'.format(n))
    pdata.set('opt_gurobi', 'logfile={}'.format(fn_log_file))

    mpc = Mpc(pdata)
    mpc.run()

    print('caca')



print('caca')




