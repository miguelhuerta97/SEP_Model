import re
import numpy as np
import pandas as pd
from os.path import join, exists
from os import mkdir
from global_definitions import *
from case_generation.ts_electric_vehicle_demand import ev_demand
from data_structure import Adn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from case_generation.pdata_scaling import scale_data_ratios
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

FLOAT_EPS = 1e-7


def mapper(x):
    return x - 1


""" ############################################################################################ """
"""                                            Input                                             """
""" ############################################################################################ """
FOL_CHUNKS = '/home/jp/tesis/data_seed/pecan_case_specific_chunks'
FOL_GRID = '/home/jp/tesis/cases_seed'
CASE_NAME = 'ieee123bus'
OFOL = '/home/jp/Data/Dropbox/tesis/tests/case_generation'
DG_LOC_MINORDER = 10
DG_SIZE_PU = 1.
VISUALIZE = False
i_chunk = 0

####################################################################################################
# Preamble                                                                                         #
####################################################################################################
if not exists(OFOL):
    mkdir(OFOL)

####################################################################################################
# Grid data manipulation                                                                           #
####################################################################################################

# Load grid data
fn_buses = join(FOL_GRID, join(CASE_NAME, CASE_NAME + '_buses.csv'))
fn_lines = join(FOL_GRID, join(CASE_NAME, CASE_NAME + '_lines.csv'))
fn_general = join(FOL_GRID, join(CASE_NAME, CASE_NAME + '_general.csv'))

df_buses = pd.read_csv(fn_buses, index_col='index', sep='\t')
df_lines = pd.read_csv(fn_lines, index_col='index', sep='\t')
df_general = pd.read_csv(fn_general, index_col='index', sep='\t')

# TODO: Avoid hardcode!
#   Hardcode bus map to 0...n where 0 is the slack_bus
df_general.loc['slack_bus', 'value'] = df_general.loc['slack_bus', 'value'].astype('int64')
df_general.loc['slack_bus', 'value'] = mapper(df_general.loc['slack_bus', 'value'])
df_buses.index = df_buses.index.map(mapper)
df_lines.loc[:, 'busf'] = df_lines.loc[:, 'busf'].apply(mapper)
df_lines.loc[:, 'bust'] = df_lines.loc[:, 'bust'].apply(mapper)
df_lines.index = df_lines.index.map(mapper)

# Construct Adn without time series data
pdata = Adn()
pdata.grid_tables['general'] = df_general
pdata.grid_tables['buses'] = df_buses
pdata.grid_tables['branches'] = df_lines

# Check if its connected

# Relabel branches
pdata.relabel_branches()

# General definitions
l_buses0 = pdata.l_buses0
l_buses = pdata.l_buses

# DG localization
set_buses_load_notnull = set(
    pdata.buses.index[abs(pdata.buses.loc[:, SNAM_LOADP_MW].values) > FLOAT_EPS]
)

se_buses2root = pdata.se_distance_to_root()
set_buses_dgs = set(se_buses2root[se_buses2root >= DG_LOC_MINORDER].index)
set_buses_dgs = set_buses_dgs.intersection(set_buses_load_notnull)
ar_buses_dgs = np.asarray(list(set_buses_dgs))

n_dgs = len(set_buses_dgs)
l_dgs = range(1, n_dgs + 1)
df_dgs = pd.DataFrame({'bus': ar_buses_dgs, 'snom': DG_SIZE_PU}, index=l_dgs)
se_buses_dgs = df_dgs['bus']

####################################################################################################
# Time series data manipulation:                                                                   #
#     Here the default scaling is applied. It consists of a set of scenarios with different        #
#     df_data ratios {}                                                                            #
####################################################################################################

# Load time series data
fn_data = join(FOL_CHUNKS, join(CASE_NAME, CASE_NAME + '_{}_data.csv'.format(i_chunk)))
df_data = pd.read_csv(fn_data, sep='\t')
df_data.loc[:, 'time'] = pd.to_datetime(df_data.loc[:, 'time'])
df_data.set_index('time', drop=True, inplace=True)
trange_all = df_data.index

l_cols_loadp = [i for i in df_data.columns if i.startswith(SNAM_LOADP)]
l_cols_loadq = [i for i in df_data.columns if i.startswith(SNAM_LOADQ)]
l_cols_loaddgp = [i for i in df_data.columns if i.startswith(SNAM_LOADDGP)]
l_cols_dgpmax = [i for i in df_data.columns if i.startswith(SNAM_DGPMAX)]
df_data.loc[:, l_cols_dgpmax] = df_data.loc[:, l_cols_dgpmax].clip(lower=0.)

# Create ev load
l_cols_loadevp = [SNAM_LOADEVP + str(i) for i in l_buses]
n_days = df_data.shape[0] // 1440
df_evload_seed = ev_demand(n_days, trange_all[0], n_samples=1)
assert df_evload_seed.index.equals(trange_all)
df_loadevp = pd.DataFrame(index=df_evload_seed.index, columns=l_cols_loadevp, dtype='float64')
df_loadevp.loc[:, l_cols_loadevp] = df_evload_seed.values

# Incorporate ev load
df_data.loc[:, l_cols_loadp] = df_data.loc[:, l_cols_loadp].values + df_loadevp.values
df_data = pd.concat([df_data, df_loadevp], axis=1)

if VISUALIZE:
    dt = df_data.index[1] - df_data.index[0]
    xformatter = '%H'
    fig, axs = plt.subplots(3, 1)

    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter(xformatter))

    tini = df_data.index[0]
    tend = tini + timedelta(days=1) - dt

    trange_plot = pd.date_range(start=tini, end=tend, freq=dt)

    l_cols = [l_cols_loaddgp[0], l_cols_loadp[0]]  # l_cols_loadevp[0]
    l_labels = ['loaddgp', 'loadp']  # 'loadev'

    for ax, col, label in zip(axs, l_cols, l_labels):
        ax.plot(df_data.loc[trange_plot, col], label=label)
        ax.legend()
    plt.tight_layout()
    plt.show()


# Set default scaling
sbase_mva = df_general.loc['sbase_mva', 'value']
n_buses0 = df_buses.shape[0]
l_buses = list(range(1, n_buses0))
n_buses = len(l_buses)
ar_fact_loadp = (df_buses.loc[l_buses, SNAM_LOADP_MW].values / sbase_mva /
                 df_data.loc[:, l_cols_loadp].max().values)


df_data_original = df_data.copy()
L_RATIOS_LOADEVP = [0.1, 0.2, 0.3]
L_RATIOS_SOLAR = [0.5, 1.0, 1.5]
for j in L_RATIOS_SOLAR:
    for i in L_RATIOS_LOADEVP:
        df_data = df_data_original.copy()
        ratio_loadevp = i
        ratio_dgpmax = j / 2
        ratio_loaddgp = j / 2

        scale_data_ratios(
            df_data,
            ar_fact_loadp,
            se_buses_dgs,
            ratio_loadevp=ratio_loadevp,
            ratio_loaddgp=ratio_loaddgp,
            ratio_dgpmax=ratio_dgpmax,
            power_factor=0.9,
            float_eps=FLOAT_EPS
        )

        # Incorporate df_data into pdata
        pdata.df_data = df_data

        # DG sizing: snom = 3 dgpmax
        l_cols_dgpmax = [i for i in df_data.columns if i.startswith(SNAM_DGPMAX)]
        ar_dgpmax_max = df_data.loc[:, l_cols_dgpmax].max(axis=0).values
        ar_buses_dgs = np.asarray([int(re.search('{}([0-9]+)'.format(SNAM_DGPMAX), i).group(1))
                                   for i in l_cols_dgpmax])
        ar_snom_dgs = 2. * ar_dgpmax_max

        df_dgs = pd.DataFrame({'bus': ar_buses_dgs, 'snom': ar_snom_dgs}, index=range(1, n_dgs + 1))

        # Incorporate df_dgs into pdata
        pdata.grid_tables['dgs'] = df_dgs

        # Write pdata to file
        folder_name = join(OFOL, '{0}_ev{1}_pv{2}'.format(CASE_NAME, i, j))
        if not exists(folder_name):
            mkdir(folder_name)
        pdata.write(folder_name, CASE_NAME)
