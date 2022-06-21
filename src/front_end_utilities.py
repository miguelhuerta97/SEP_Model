import pandas as pd
from datetime import timedelta
from math import ceil
import data_structure as ds
import gc
import numpy as np
from pandas.tseries.frequencies import to_offset

FLOAT_EPS = 1e-7


def down_sample_pdata(pdata: ds.Adn, n_hours: float, n_rh: int, Dt_down: timedelta,
                      idx_ini: int):

    df_data = pdata.df_data

    # Obtain minimum number of windows such that trange is larger than n_hours
    one_hour = timedelta(hours=1)
    Dt_up = df_data.index[1] - df_data.index[0]
    hours_per_win = (Dt_down * n_rh) / one_hour
    n_win = ceil(n_hours / hours_per_win)

    n_t_up = ceil(n_win * hours_per_win * one_hour / Dt_up)

    if n_t_up == df_data.shape[0]:
        tend = df_data.index[-1]
        df_data.loc[tend + Dt_up, :] = df_data.loc[tend, :].values
    # Data down sampling
    idx_end = idx_ini + n_t_up
    tend = df_data.index[idx_end]
    tini = df_data.index[idx_ini]

    trange = df_data.index[idx_ini:idx_end+1]

    trange_down = pd.date_range(start=tini, end=tend, freq=Dt_down)
    df_data_down = pd.DataFrame(index=trange_down, columns=df_data.columns, dtype='float64')
    df_data_down.loc[trange, :] = df_data.loc[trange, :]

    gc.collect()

    df_data_down = df_data_down.interpolate()
    df_data_down = df_data_down.loc[trange_down[:-1], :]

    pdata.df_data = df_data_down
    tini = df_data_down.index[0]
    tend = df_data_down.index[-1]

    pdata.time_config['tini'] = tini
    pdata.time_config['tend'] = tend
    pdata.time_config['dt'] = pd.to_timedelta(trange_down.freq).to_pytimedelta()
    pdata.time_config['n_rh'] = n_rh


def load_df_result(file_name):
    df_result = pd.read_csv(file_name, index_col='time')
    df_result.index = pd.to_datetime(df_result.index)
    df_result = df_result.astype('float64')
    return df_result


def write_df_result(df_result, file_name):
    assert isinstance(df_result.index, pd.DatetimeIndex)
    df_result.to_csv(file_name, index_label='time')


def down_sample_setpoints(df_insample, dt):
    dt_insample = pd.to_timedelta(to_offset(pd.infer_freq(df_insample.index)))
    assert dt_insample is not None
    assert dt < dt_insample
    tidx_append = df_insample.index[-1] + dt_insample
    df_insample.loc[tidx_append, :] = np.nan
    return df_insample.resample(dt).ffill()
