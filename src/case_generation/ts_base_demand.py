"""
Real data source:
Pecan Street Dataport. (2020). Retrieved October 30, 2020, from
https://www.pecanstreet.org/dataport/

Data file has 1-minute interval data from 25 homes of Austin region. This has 1-year of data with
99% completeness.
"""
import numpy as np
import pandas as pd
from glob import glob
from os.path import dirname, join
from global_definitions import *

THIS_DIR = dirname(__file__)
IFOL = join(THIS_DIR, '../../data_seed/pecan_1min_by_user')


DEF_LOAD_FILE_FIELDS = [SNAM_LOADP_KW, SNAM_DGPMAX_KW]
"""Hyper parameters"""
POWER_FACTOR_MEAN = 0.95


def load_ts_data(ifol, pattern, fields:list, index_col='time'):
    """
    Assumes a folder containing csv files, each one associated to a specyfic data series
    delimited by ',', with fields:
        <index_col> {str yyyy-mm-dd HH:MM:SS}
        <field> {float64}
    It assumes no missing data
    :param ifol: {str} Folder containing the files
    :param pattern: {str}
    :param fields: {list}
    :param index_col: {str}
    :return:
    """
    l_files = glob(join(ifol, pattern))
    n_files = len(l_files)
    se_data = pd.read_csv(join(ifol, l_files[0]), squeeze=True,
                          delimiter=',', index_col=index_col, usecols=[index_col, *fields])
    se_data.index = pd.to_datetime(se_data.index)
    assert isinstance(se_data.index, pd.DatetimeIndex)
    l_cols = [i + '_' + str(j) for i in fields for j in range(n_files)]
    df_data = pd.DataFrame(se_data.index, columns=l_cols)
    df_data[[i + '_' + str(0) for i in fields]] = se_data
    for n in range(1, n_files):
        se_data = pd.read_csv(join(ifol, l_files[n]), squeeze=True,
                              delimiter=',', index_col=index_col, usecols=[index_col, *fields])
        se_data.index = pd.to_datetime(se_data.index)
        assert df_data.index.equals(se_data.index)
        df_data[[i + '_' + str(n) for i in fields]] = se_data.astype('float64')  # [n + 1] because of the assingnt init for bucle

    return df_data


def reactive_power(df_active_power, pf):
    q = np.sqrt((1 - pf ** 2) / pf ** 2) * df_active_power
    return q


def smooth_series(df_data, n_win):
    return df_data.rolling(window=n_win, win_type='gaussian', min_periods=1).mean(std=3)


def proc_nan_data(df_seed: pd.DataFrame, primary_col_prefix: str = SNAM_LOADP_KW, freq='W-MON',
                  n_nan_consecutive=60):
    lb_null_rows = 0.95
    n_rows = df_seed.shape[0]
    assert isinstance(df_seed.index, pd.DatetimeIndex)
    l_cols_primary = [i for i in df_seed.columns if i.startswith(primary_col_prefix)]

    sbl_cols_prim_not_null = (
        df_seed[l_cols_primary].notnull().sum() > np.round(lb_null_rows * n_rows))

    l_cols_primary_not_null = sbl_cols_prim_not_null[sbl_cols_prim_not_null].index.to_list()
    l_relevant_suffix = [int(i.split('_')[-1]) for i in l_cols_primary_not_null]
    l_relevant_cols = [i for i in df_seed.columns if int(i.split('_')[-1]) in l_relevant_suffix]
    df_seed = df_seed.loc[:, l_relevant_cols]

    dt = df_seed.index[1] - df_seed.index[0]
    df_seed = df_seed.asfreq(dt)

    df_view_prim = df_seed[l_cols_primary_not_null]
    for i in df_view_prim.columns:
        df_seed.loc[df_seed[i].isna(), i] = df_seed.shift(n_nan_consecutive).loc[df_seed[i].isna(), i]

    df_seed: pd.DataFrame = df_seed.groupby(pd.Grouper(freq=freq)).filter(
        lambda x: x[l_cols_primary_not_null].isna().sum().sum() == 0)

    assert df_seed.loc[:, l_cols_primary_not_null].isna().sum().sum() == 0
    df_seed.fillna(0., inplace=True)

    return df_seed


def base_demand(ifol, fname_pattern, n=None, l_buses=None, n_win_filter=15,
                seed_field_name=None, rnd_seed=1):
    if seed_field_name is None:
        seed_field_name = DEF_LOAD_FILE_FIELDS
    assert n is not None or l_buses is not None
    if n is None:
        n = len(l_buses)
    else:
        l_buses = range(n)
    l_cols_loadp = [SNAM_LOADP + str(i) for i in l_buses]
    l_cols_loadq = [SNAM_LOADQ + str(i) for i in l_buses]
    l_cols_loaddgp = [SNAM_LOADDGP + str(i) for i in l_buses]

    df_seed = load_ts_data(ifol, fname_pattern, seed_field_name)

    # Fill nan values
    df_seed = proc_nan_data(df_seed, freq='D')
    assert df_seed.isna().sum().sum() == 0

    l_cols_loadp_kw = [i for i in df_seed.columns if i.startswith(SNAM_LOADP_KW)]
    l_cols_dgpmax_kw = [i for i in df_seed.columns if i.startswith(SNAM_DGPMAX_KW)]
    df_seed_loadp = df_seed.loc[:, l_cols_loadp_kw]
    df_seed_loadq = reactive_power(df_seed[l_cols_loadp_kw], POWER_FACTOR_MEAN)
    df_seed_dgpmax = df_seed.loc[:, l_cols_dgpmax_kw]

    # Linear combination (assumes no nan values)
    m = df_seed_loadp.shape[1]
    np.random.seed(rnd_seed)
    weights = np.random.rand(m, n)

    df_p = pd.DataFrame(index=df_seed_loadp.index, columns=l_buses)
    df_p.loc[:, :] = df_seed_loadp.values @ weights
    assert len(l_cols_loadp) == df_p.shape[1]
    df_p.columns = l_cols_loadp
    df_q = df_seed_loadq @ weights
    df_q.columns = l_cols_loadq
    df_loads = pd.concat([df_p, df_q], axis=1)
    df_dgpmax_unctrl = df_seed_dgpmax @ weights
    df_dgpmax_unctrl.columns = l_cols_loaddgp
    # Filter series
    df_loads = smooth_series(df_loads, n_win_filter)
    df_dgpmax_unctrl = smooth_series(df_dgpmax_unctrl, n_win_filter)

    assert df_loads.index.equals(df_dgpmax_unctrl.index)

    return np.concatenate([df_loads, df_dgpmax_unctrl], axis=1)



