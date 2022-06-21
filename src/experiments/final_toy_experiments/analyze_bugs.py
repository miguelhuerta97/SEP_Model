import pandas as pd
from os.path import join
from front_end_utilities import load_df_result
import matplotlib.pyplot as plt
from os import mkdir, walk
import re


def key_sort(x):
    return int(re.search('[0-9]+', x).group(0))


ifol = ('/home/jp/tesis/experiments/ieee4bus_lasso/'
                'bilinear_lassodet_sce100_10days_ipopt_cvlim/log_sto_per')


_, _, l_filenames = next(walk(ifol))

l_fn_setpoint = []
l_fn_insample = []
for fname in l_filenames:
    if fname.startswith('df_vars'):
        l_fn_insample.append(fname)
    elif fname.startswith('df_setpoint'):
        l_fn_setpoint.append(fname)

l_fn_setpoint.sort(key=key_sort)
l_fn_insample.sort(key=key_sort)

df_minmax_in = pd.DataFrame(index=range(len(l_fn_setpoint)), columns=['vmin', 'vmax'], dtype='float64')

k = 0
for fn_setpoint_local, fn_insample_local in zip(l_fn_setpoint, l_fn_insample):
    fn_setpoint = join(ifol, fn_setpoint_local)
    fn_insample = join(ifol, fn_insample_local)
    df_setpoint = pd.read_csv(fn_setpoint, index_col='index')
    df_insample = pd.read_csv(fn_insample, index_col='index')

    df_insample['v'] = df_insample['dv'] + df_setpoint.loc[df_insample['bus'], 'v_set'].values
    df_insample['uq'] = df_insample['duq'] + df_setpoint.loc[df_insample['bus'], 'uq_mean'].values
    df_insample['up'] = df_insample['dup'] + df_setpoint.loc[df_insample['bus'], 'up_mean'].values

    # Calculate v
    df_minmax_in.loc[k, 'vmin'] = df_insample.loc[:, 'v'].min()
    df_minmax_in.loc[k, 'vmax'] = df_insample.loc[:, 'v'].max()

    k += 1

print(df_minmax_in.describe())

"""
prefix = 'df_sce_'
ifol_noctrl = '/home/jp/tesis/tests/log_scenarios/noctrl'
ifol_droop = '/home/jp/tesis/tests/log_scenarios/droop'
l_cols = ['loadp1', 'loadq1', 'dgpmax1']
for i in range(1):
    fn_sce_noctrl = join(ifol_noctrl, prefix + '{}.csv'.format(i))
    df_sce_noctrl = pd.read_csv(fn_sce_noctrl, usecols=l_cols)
    fn_sce_droop = join(ifol_droop, prefix + '{}.csv'.format(i))
    df_sce_droop = pd.read_csv(fn_sce_droop, usecols=l_cols)
    print('caca')




ifol_noctrl = '/home/jp/tesis/experiments/toy_benchmark/bilinear_benchmark_sce100_10days_rnd/noctrl'
ifol_droop = ('/home/jp/tesis/experiments/toy_lasso/bilinear_droop_sce100_10days_rnd/'
              'report_toy1bus_rob_kme')
fn_ins_noctrl = join(ifol_noctrl, 'df_ins_noctrl.csv')
fn_ins_droop = join(ifol_droop, 'df_ins_toy1bus_rob_kme.csv')

df_ins_noctrl = load_df_result(fn_ins_noctrl)
df_ins_droop = load_df_result(fn_ins_droop)

l_cols_dgs = [i for i in df_ins_droop.columns if i.startswith('dg')]

df_diff = df_ins_noctrl.loc[:, l_cols_dgs] - df_ins_droop.loc[:, l_cols_dgs]
print(df_diff.abs().max())

print('caca')

plt.plot(df_diff)
plt.show()

"""

