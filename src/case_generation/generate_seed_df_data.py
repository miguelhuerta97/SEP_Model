"""
From pecan street data. Dump valid chunks of load data and uncontrolled residential solar data into
a folder.
"""
import numpy as np
from datetime import timedelta, datetime
from case_generation.ts_base_demand import proc_nan_data, load_ts_data, smooth_series
from os.path import exists, join
from os import mkdir


IFOL = '/home/jp/Data/Dropbox/tesis/data_seed/pecan_1min_by_user_solar_NaNvals'
OFOL = '/home/jp/tesis/data_seed/pecan_1_min_proc_cont_chunks_noisy'

if not exists(OFOL):
    mkdir(OFOL)


df_load = load_ts_data(IFOL, 'load*', ['loadp_kw', 'dgpmax_kw'])
df_load = proc_nan_data(df_load, freq='D')

df_load.loc[(df_load.loc[:, 'dgpmax_kw_2'] > 8000.0), 'dgpmax_kw_2'] = np.nan
df_load.loc[(df_load.loc[:, 'dgpmax_kw_2'] < -8000.0), 'dgpmax_kw_2'] = np.nan
df_load.loc[(df_load.loc[:, 'loadp_kw_2'] > 2000.0), 'loadp_kw_2'] = np.nan
df_load.loc[(df_load.loc[:, 'loadp_kw_2'] < -2000.0), 'loadp_kw_2'] = np.nan

df_load.fillna(method='bfill', inplace=True)

df_load = smooth_series(df_load, 5)


ar_break_points = (df_load.index[1:] - df_load.index[:-1]) != timedelta(minutes=1)
ar_break_points = np.concatenate([np.asarray([False]), ar_break_points])
ar_groups = ar_break_points.cumsum()

grouped = df_load.groupby(ar_groups)

n = 0
for _, v in sorted(grouped, key=lambda x: x[1].shape[0], reverse=True):
    if v.shape[0] > 1440 * 2:

        fname = join(OFOL, 'load_kw_chunk_%d.csv' % n)
        v.to_csv(fname, sep='\t', index_label='time')
        n = n + 1